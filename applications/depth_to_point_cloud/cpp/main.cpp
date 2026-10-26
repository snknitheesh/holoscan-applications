// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Depth to Point Cloud: depth images in, organized point clouds out, on the
// GPU. A source (a synthetic depth camera with known ground truth, or a folder
// of PGM depth images) feeds DepthToPointCloudOp, whose clouds a reporter
// checks, measures and optionally writes as PLY files.

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "deproject/image_io.hpp"
#include "deproject/synthetic_depth.hpp"
#include "operators/depth_sources.hpp"
#include "operators/depth_to_point_cloud_op.hpp"
#include "operators/point_cloud_report_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace dpc = depth_to_point_cloud;

constexpr std::string_view kAppName = "depth_to_point_cloud";

enum class Source { kSynthetic, kImages };

struct Options {
  Source source{Source::kSynthetic};
  std::filesystem::path images;
  // The .pgm files in `images`, listed by validate().
  std::vector<std::filesystem::path> image_files;
  bool loop{false};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  // Synthetic image size; validate() sets it to the images' size for --source images.
  bool size_given{false};
  int width{640};
  int height{480};
  bool millimeters{false};
  bool color{true};
  bool intrinsics_input{false};
  bool host_input{false};
  std::optional<dpc::CameraIntrinsics> intrinsics;
  std::optional<float> depth_scale;
  float depth_min{0.1F};
  float depth_max{10.0F};
  std::filesystem::path ply_dir;
  bool verify{true};
  std::uint32_t timeout_seconds{120};
  bool compile_only{false};
  bool help{false};
};

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " requires true or false");
}

template <typename T>
T parse_number(std::string_view value, std::string_view option) {
  T result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires a number, got '" +
                                std::string(value) + "'");
  }
  return result;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  std::optional<float> fx, fy, cx, cy;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const std::string_view value{argv[++i]};
    if (option == "--source") {
      if (value == "synthetic") {
        opts.source = Source::kSynthetic;
      } else if (value == "images") {
        opts.source = Source::kImages;
      } else {
        throw std::invalid_argument("--source must be synthetic or images");
      }
    } else if (option == "--images") {
      opts.images = value;
    } else if (option == "--loop") {
      opts.loop = parse_bool(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
      opts.size_given = true;
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
      opts.size_given = true;
    } else if (option == "--depth-format") {
      if (value == "meters") {
        opts.millimeters = false;
      } else if (value == "millimeters") {
        opts.millimeters = true;
      } else {
        throw std::invalid_argument("--depth-format must be meters or millimeters");
      }
    } else if (option == "--color") {
      opts.color = parse_bool(value, option);
    } else if (option == "--intrinsics-input") {
      opts.intrinsics_input = parse_bool(value, option);
    } else if (option == "--host-input") {
      opts.host_input = parse_bool(value, option);
    } else if (option == "--fx") {
      fx = parse_number<float>(value, option);
    } else if (option == "--fy") {
      fy = parse_number<float>(value, option);
    } else if (option == "--cx") {
      cx = parse_number<float>(value, option);
    } else if (option == "--cy") {
      cy = parse_number<float>(value, option);
    } else if (option == "--depth-scale") {
      opts.depth_scale = parse_number<float>(value, option);
    } else if (option == "--depth-min") {
      opts.depth_min = parse_number<float>(value, option);
    } else if (option == "--depth-max") {
      opts.depth_max = parse_number<float>(value, option);
    } else if (option == "--ply-dir") {
      opts.ply_dir = value;
    } else if (option == "--verify") {
      opts.verify = parse_bool(value, option);
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  const int given = fx.has_value() + fy.has_value() + cx.has_value() + cy.has_value();
  if (given != 0 && given != 4) {
    throw std::invalid_argument("--fx, --fy, --cx and --cy go together");
  }
  if (given == 4) {
    opts.intrinsics = dpc::CameraIntrinsics{*fx, *fy, *cx, *cy};
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Input:\n"
         "  --source synthetic|images  A synthetic depth camera with ground truth, or PGM\n"
         "                             depth images (default: synthetic)\n"
         "  --images DIR               8- or 16-bit PGM depth images, read in name order\n"
         "  --loop true|false          Repeat the images until --count frames (default: false)\n"
         "  --count N                  Frames (default: 100 synthetic, every image once)\n"
         "  --period-ms N              Source period, 1..1000 ms (default: 33)\n"
         "Synthetic camera:\n"
         "  --width N --height N       Image size, 64..4096 x 48..4096 (default: 640x480)\n"
         "  --depth-format meters|millimeters\n"
         "                             float32 meters or uint16 millimeters (default: meters)\n"
         "  --color true|false         Also send an aligned RGB image (default: true)\n"
         "  --intrinsics-input true|false\n"
         "                             Send per-frame intrinsics instead of configuring them\n"
         "                             (default: false)\n"
         "  --host-input true|false    Send images in host instead of GPU memory\n"
         "                             (default: false)\n"
         "Deprojection:\n"
         "  --fx F --fy F --cx C --cy C  Camera intrinsics in pixels (--source images)\n"
         "  --depth-scale S            Raw depth to meters (default: 0.001 for 16-bit\n"
         "                             images and millimeters, 1 for meters)\n"
         "  --depth-min M --depth-max M  Valid depth range in meters (default: 0.1, 10)\n"
         "Output:\n"
         "  --ply-dir DIR              Write each cloud as DIR/frame_NNNNNN.ply\n"
         "  --verify true|false        Check synthetic clouds against ground truth\n"
         "                             (default: true)\n"
         "  --compile-only true|false  Build and check the graph without running it\n"
         "  --timeout-seconds N        Allowance beyond the expected run time (default: 120)\n";
}

// The operator configurations a run uses; validate() checks them all before
// execute() builds the graph from them.
dpc::SyntheticSourceConfig synthetic_source_config(const Options& opts) {
  return dpc::SyntheticSourceConfig{.width = opts.width,
                                    .height = opts.height,
                                    .count = opts.count.value_or(0),
                                    .period = std::chrono::milliseconds(opts.period_ms),
                                    .millimeters = opts.millimeters,
                                    .color = opts.color,
                                    .intrinsics = opts.intrinsics_input,
                                    .device_output = !opts.host_input};
}

dpc::DepthFileSourceConfig file_source_config(const Options& opts) {
  return dpc::DepthFileSourceConfig{.files = opts.image_files,
                                    .width = opts.width,
                                    .height = opts.height,
                                    .count = opts.count.value_or(0),
                                    .loop = opts.loop,
                                    .period = std::chrono::milliseconds(opts.period_ms)};
}

float depth_scale(const Options& opts) {
  if (opts.source == Source::kSynthetic) {
    return opts.millimeters ? 0.001F : 1.0F;
  }
  return opts.depth_scale.value_or(0.001F);
}

dpc::DepthToPointCloudConfig deproject_config(const Options& opts) {
  return dpc::DepthToPointCloudConfig{.intrinsics = opts.intrinsics,
                                      .depth_scale = depth_scale(opts),
                                      .depth_min = opts.depth_min,
                                      .depth_max = opts.depth_max,
                                      .color = opts.color,
                                      .intrinsics_input = opts.intrinsics_input,
                                      .max_width = opts.width,
                                      .max_height = opts.height};
}

void validate(Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  if (opts.source == Source::kImages) {
    if (opts.images.empty()) {
      throw std::invalid_argument("--source images needs --images DIR");
    }
    if (!opts.intrinsics) {
      throw std::invalid_argument("--source images needs --fx, --fy, --cx and --cy");
    }
    if (opts.intrinsics_input || opts.host_input || opts.millimeters || opts.size_given) {
      throw std::invalid_argument(
          "--intrinsics-input, --host-input, --depth-format, --width and --height apply to "
          "synthetic frames only");
    }
    opts.color = false;
    // Read the first image now, so a missing folder or an unreadable image is a
    // usage error rather than a failed run.
    try {
      opts.image_files = dpc::list_pgm_files(opts.images);
      const dpc::DepthImage first = dpc::read_pgm_depth(opts.image_files.front());
      opts.width = first.width;
      opts.height = first.height;
    } catch (const std::runtime_error& error) {
      throw std::invalid_argument(error.what());
    }
    const auto total = static_cast<std::uint32_t>(opts.image_files.size());
    if (opts.count.value_or(0) == 0) {
      opts.count = total;
    }
    dpc::validate_config(file_source_config(opts));
  } else {
    if (!opts.images.empty()) {
      throw std::invalid_argument("--images needs --source images");
    }
    if (opts.intrinsics) {
      throw std::invalid_argument("the synthetic camera sets its own intrinsics; drop --fx..--cy");
    }
    if (opts.depth_scale) {
      throw std::invalid_argument("--depth-scale follows --depth-format for synthetic frames");
    }
    if (!opts.count) {
      opts.count = 100;
    }
    if (*opts.count == 0) {
      throw std::invalid_argument("synthetic runs need --count > 0");
    }
    dpc::validate_config(synthetic_source_config(opts));
    if (!opts.intrinsics_input) {
      opts.intrinsics = dpc::synthetic_frame(opts.width, opts.height, 0, false).intrinsics;
    }
  }
  if (!opts.ply_dir.empty() && !std::filesystem::is_directory(opts.ply_dir)) {
    throw std::invalid_argument("--ply-dir is not a directory: " + opts.ply_dir.string());
  }
  // The operator's own checks, such as the intrinsics and the depth range.
  dpc::validate_config(deproject_config(opts));
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(Options opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const std::uint32_t frames = *opts.count;
  const holoscan::Output<holoscan::schema::ImageT>* depth_out = nullptr;
  const holoscan::Output<holoscan::schema::ImageT>* color_out = nullptr;
  const holoscan::Output<holoscan::Tensor>* intrinsics_out = nullptr;
  std::optional<holoscan::OperatorHandle<dpc::SyntheticDepthSourceOp>> synthetic;
  std::optional<holoscan::OperatorHandle<dpc::DepthFileSourceOp>> files;
  if (opts.source == Source::kSynthetic) {
    synthetic.emplace(
        graph.op<dpc::SyntheticDepthSourceOp>("source", synthetic_source_config(opts)));
    depth_out = &(*synthetic)->depth;
    color_out = &(*synthetic)->color;
    intrinsics_out = &(*synthetic)->intrinsics;
  } else {
    files.emplace(graph.op<dpc::DepthFileSourceOp>("source", file_source_config(opts)));
    depth_out = &(*files)->depth;
  }

  auto cloud = graph.op<dpc::DepthToPointCloudOp>("deproject", deproject_config(opts));
  auto report = std::make_shared<dpc::CloudReport>();
  std::optional<dpc::CloudTruth> truth;
  if (opts.source == Source::kSynthetic && opts.verify) {
    truth = dpc::CloudTruth{.width = opts.width,
                            .height = opts.height,
                            .millimeters = opts.millimeters,
                            .vary_intrinsics = opts.intrinsics_input,
                            .depth_scale = depth_scale(opts),
                            .depth_min = opts.depth_min,
                            .depth_max = opts.depth_max};
  }
  auto reporter = graph.op<dpc::PointCloudReportOp>(
      "report", dpc::PointCloudReportConfig{.report = report,
                                            .expected_frames = frames,
                                            .colors = opts.color,
                                            .truth = truth,
                                            .ply_dir = opts.ply_dir});

  // Lossless edges: every cloud is checked, in order, against its frame. The
  // queue depth is the one each input declares in its operator's setup().
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(*depth_out, cloud->depth, lossless);
  if (opts.color) {
    graph.add_flow(*color_out, cloud->color, lossless);
    graph.add_flow(cloud->colors, reporter->colors, lossless);
  }
  if (opts.intrinsics_input) {
    graph.add_flow(*intrinsics_out, cloud->intrinsics, lossless);
  }
  graph.add_flow(cloud->point_cloud, reporter->point_cloud, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // GPU outputs are on device 0.
  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("deproject", "point_cloud");
  if (opts.color) {
    on_gpu("deproject", "colors");
  }
  if (opts.source == Source::kSynthetic && !opts.host_input) {
    on_gpu("source", "depth");
    if (opts.color) {
      on_gpu("source", "color");
    }
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  if (opts.compile_only) {
    std::cout << kAppName << ": plan=OK\n";
    return 0;
  }
  if (!signals) {
    // Profilers, sanitizers and preloaded libraries can start a thread before
    // main(); the SDK then refuses to take over SIGINT and SIGTERM.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error());
    if (signals.error().code == holoscan::ErrorCode::kInvalidArgument) {
      std::cerr << " (another thread was already running)";
    }
    std::cerr << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(opts.timeout_seconds) +
                        std::chrono::milliseconds(2LL * frames * opts.period_ms);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  const bool timed_out = session.is_running();
  if (timed_out) {
    std::cerr << kAppName << ": timed out\n";
    session.request_stop();
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  // The SDK logs a failed run's diagnostics itself, unless its report or its
  // logger is turned down; show those of any other run, such as the warning a
  // stop request leaves.
  const bool sdk_reported = termination == holoscan::RunTermination::kFailed &&
                            holoscan::automatic_diagnostic_logging_enabled() &&
                            holoscan::log_level() <= holoscan::LogLevel::ERROR;
  if (!session.diagnostics().empty() && !sdk_reported) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  if (signals) {
    if (auto closed = signals->close(); !closed) {
      std::cerr << kAppName << ": warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }

  // A stop request (SIGINT, SIGTERM) ends the run early; that is not a
  // failure, but not a pass either.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      report->frames == frames && report->passed();
  std::cout << std::fixed << std::setprecision(3) << kAppName << ": frames=" << report->frames
            << '/' << frames << " points=" << report->points << " valid=" << report->valid_points
            << " depth_m=[" << report->min_depth << ", " << report->max_depth << ']';
  if (!opts.ply_dir.empty()) {
    std::cout << " ply_files=" << report->ply_files;
  }
  if (report->frame_gaps != 0) {
    std::cout << " frame_gaps=" << report->frame_gaps;
  }
  if (report->verified) {
    std::cout << " truth: wrong_points=" << report->wrong_points
              << " wrong_colors=" << report->wrong_colors << std::scientific << std::setprecision(2)
              << " max_error_m=" << report->max_error_m;
  }
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread; the SDK
  // refuses a process that already runs others.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  auto opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts);
  return execute(std::move(opts), signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
