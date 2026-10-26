// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Point Cloud to Depth: point clouds in, depth images out, on the GPU. A
// source (a synthetic cloud with known ground truth, or a folder of PLY files)
// feeds PointCloudToDepthOp, whose depth and color images a reporter checks,
// measures and optionally writes as PGM and PPM files.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
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
#include <system_error>
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

#include "operators/depth_image_report_op.hpp"
#include "operators/point_cloud_sources.hpp"
#include "operators/point_cloud_to_depth_op.hpp"
#include "projection/cloud_io.hpp"
#include "projection/synthetic_scene.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace pcd = point_cloud_to_depth;

constexpr std::string_view kAppName = "point_cloud_to_depth";

enum class Source { kSynthetic, kFiles };

struct Options {
  Source source{Source::kSynthetic};
  std::filesystem::path clouds;
  // Found by validate() for --source files: the .ply files in `clouds`, the
  // most points in one, and whether they have colors.
  std::vector<std::filesystem::path> cloud_files;
  std::uint64_t file_max_points{};
  bool loop{false};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  bool width_given{false};
  bool height_given{false};
  int width{640};
  int height{480};
  pcd::DepthFormat depth_format{pcd::DepthFormat::kFloat32};
  std::optional<bool> color;
  bool intrinsics_input{false};
  bool host_input{false};
  std::optional<pcd::CameraIntrinsics> intrinsics;
  std::optional<pcd::Extrinsics> extrinsics;
  std::optional<float> depth_scale;
  float depth_min{0.1F};
  float depth_max{10.0F};
  int point_size{1};
  std::filesystem::path image_dir;
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

// Twelve comma-separated numbers: a 3x4 row-major [R | t].
pcd::Extrinsics parse_extrinsics(std::string_view value, std::string_view option) {
  pcd::Extrinsics result;
  std::size_t index = 0;
  while (true) {
    const std::size_t comma = value.find(',');
    if (index == 12) {
      throw std::invalid_argument(std::string(option) + " takes 12 comma-separated numbers");
    }
    result.matrix[index++] = parse_number<float>(value.substr(0, comma), option);
    if (comma == std::string_view::npos) {
      break;
    }
    value.remove_prefix(comma + 1);
  }
  if (index != 12) {
    throw std::invalid_argument(std::string(option) + " takes 12 comma-separated numbers");
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
      } else if (value == "files") {
        opts.source = Source::kFiles;
      } else {
        throw std::invalid_argument("--source must be synthetic or files");
      }
    } else if (option == "--clouds") {
      opts.clouds = value;
    } else if (option == "--loop") {
      opts.loop = parse_bool(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
      opts.width_given = true;
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
      opts.height_given = true;
    } else if (option == "--depth-format") {
      if (value == "float32") {
        opts.depth_format = pcd::DepthFormat::kFloat32;
      } else if (value == "uint16") {
        opts.depth_format = pcd::DepthFormat::kUint16;
      } else {
        throw std::invalid_argument("--depth-format must be float32 or uint16");
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
    } else if (option == "--extrinsics") {
      opts.extrinsics = parse_extrinsics(value, option);
    } else if (option == "--depth-scale") {
      opts.depth_scale = parse_number<float>(value, option);
    } else if (option == "--depth-min") {
      opts.depth_min = parse_number<float>(value, option);
    } else if (option == "--depth-max") {
      opts.depth_max = parse_number<float>(value, option);
    } else if (option == "--point-size") {
      opts.point_size = parse_number<int>(value, option);
    } else if (option == "--image-dir") {
      opts.image_dir = value;
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
    opts.intrinsics = pcd::CameraIntrinsics{*fx, *fy, *cx, *cy};
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Input:\n"
         "  --source synthetic|files   A synthetic cloud with ground truth, or PLY files\n"
         "                             (default: synthetic)\n"
         "  --clouds DIR               Binary little-endian PLY files, read in name order\n"
         "  --loop true|false          Repeat the files until --count clouds (default: false)\n"
         "  --count N                  Clouds (default: 100 synthetic, every file once)\n"
         "  --period-ms N              Source period, 1..1000 ms (default: 33)\n"
         "Synthetic cloud:\n"
         "  --intrinsics-input true|false\n"
         "                             Send per-frame intrinsics instead of configuring them\n"
         "                             (default: false)\n"
         "  --host-input true|false    Send clouds in host instead of GPU memory\n"
         "                             (default: false)\n"
         "Projection:\n"
         "  --width N --height N       Depth image size; synthetic: 64..4096 x 48..4096\n"
         "                             (default: 640x480), files: required, 1..16384\n"
         "  --fx F --fy F --cx C --cy C  Camera intrinsics in pixels (--source files)\n"
         "  --extrinsics R00,R01,R02,TX,R10,R11,R12,TY,R20,R21,R22,TZ\n"
         "                             Row-major [R | t] from the files' frame into the camera\n"
         "                             optical frame (--source files; default: identity)\n"
         "  --depth-format float32|uint16\n"
         "                             float32 meters or uint16 counts (default: float32)\n"
         "  --depth-scale S            Meters per uint16 count (--source files; default: 0.001)\n"
         "  --depth-min M --depth-max M  Valid depth range in meters (default: 0.1, 10)\n"
         "  --point-size 1|3|5         Pixels a side each point covers (default: 1)\n"
         "  --color true|false         Also make a color image (default: true when the\n"
         "                             points have colors)\n"
         "Output:\n"
         "  --image-dir DIR            Write each depth image as DIR/depth_NNNNNN.pgm\n"
         "                             (millimeters) and color image as DIR/color_NNNNNN.ppm\n"
         "  --verify true|false        Check synthetic images against ground truth\n"
         "                             (default: true)\n"
         "  --compile-only true|false  Build and check the graph without running it\n"
         "  --timeout-seconds N        Fail when no image arrives for N seconds plus two\n"
         "                             source periods (default: 120)\n";
}

bool color_enabled(const Options& opts) { return opts.color.value_or(true); }

// The operator configurations a run uses; validate() checks them all before
// execute() builds the graph from them.
pcd::SyntheticCloudSourceConfig synthetic_source_config(const Options& opts) {
  return pcd::SyntheticCloudSourceConfig{
      .width = opts.width,
      .height = opts.height,
      .count = opts.count.value_or(0),
      .period = std::chrono::milliseconds(opts.period_ms),
      .millimeter_depth = opts.depth_format == pcd::DepthFormat::kUint16,
      .color = color_enabled(opts),
      .intrinsics = opts.intrinsics_input,
      .device_output = !opts.host_input};
}

pcd::CloudFileSourceConfig file_source_config(const Options& opts) {
  return pcd::CloudFileSourceConfig{.files = opts.cloud_files,
                                    .max_points = opts.file_max_points,
                                    .color = color_enabled(opts),
                                    .count = opts.count.value_or(0),
                                    .loop = opts.loop,
                                    .period = std::chrono::milliseconds(opts.period_ms)};
}

std::uint32_t max_points(const Options& opts) {
  return opts.source == Source::kSynthetic
             ? pcd::synthetic_cloud_layout(opts.width, opts.height).total()
             : static_cast<std::uint32_t>(
                   std::min<std::uint64_t>(opts.file_max_points, pcd::kMaxPointsLimit));
}

pcd::PointCloudToDepthConfig projection_config(const Options& opts) {
  return pcd::PointCloudToDepthConfig{.width = opts.width,
                                      .height = opts.height,
                                      .intrinsics = opts.intrinsics,
                                      .extrinsics = opts.extrinsics,
                                      .depth_min = opts.depth_min,
                                      .depth_max = opts.depth_max,
                                      .depth_format = opts.depth_format,
                                      .depth_scale = opts.depth_scale.value_or(0.001F),
                                      .point_size = opts.point_size,
                                      .color = color_enabled(opts),
                                      .intrinsics_input = opts.intrinsics_input,
                                      .max_points = max_points(opts)};
}

// Lists the files and reads their headers, so a missing folder or an
// unreadable file is a usage error rather than a failed run.
void inspect_files(Options& opts) {
  try {
    opts.cloud_files = pcd::list_ply_files(opts.clouds);
    std::optional<bool> colored;
    for (const auto& file : opts.cloud_files) {
      const pcd::PlyHeader header = pcd::read_ply_header(file);
      if (colored && *colored != header.color) {
        throw std::invalid_argument("either every PLY file has colors or none does; " +
                                    file.string() + " differs");
      }
      colored = header.color;
      opts.file_max_points = std::max(opts.file_max_points, header.points);
    }
    if (opts.color.value_or(false) && !*colored) {
      throw std::invalid_argument("--color true needs PLY files with colors");
    }
    if (!opts.color) {
      opts.color = *colored;
    }
  } catch (const std::runtime_error& error) {
    throw std::invalid_argument(error.what());
  }
  if (opts.file_max_points == 0) {
    throw std::invalid_argument(opts.clouds.string() + ": the PLY files have no points");
  }
  if (opts.file_max_points > pcd::kMaxPointsLimit) {
    throw std::invalid_argument("a PLY file has more than " + std::to_string(pcd::kMaxPointsLimit) +
                                " points");
  }
}

// Whether the 3x3 part of `transform` is a rotation: R^T R is the identity
// within float rounding and det(R) is positive, not a reflection's -1.
bool is_rotation(const pcd::Extrinsics& transform) {
  const auto r = [&transform](int row, int col) {
    return static_cast<double>(transform.matrix[4 * row + col]);
  };
  constexpr double kTolerance = 1e-3;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double dot = r(0, i) * r(0, j) + r(1, i) * r(1, j) + r(2, i) * r(2, j);
      if (!(std::fabs(dot - (i == j ? 1.0 : 0.0)) <= kTolerance)) {
        return false;
      }
    }
  }
  const double det = r(0, 0) * (r(1, 1) * r(2, 2) - r(1, 2) * r(2, 1)) -
                     r(0, 1) * (r(1, 0) * r(2, 2) - r(1, 2) * r(2, 0)) +
                     r(0, 2) * (r(1, 0) * r(2, 1) - r(1, 1) * r(2, 0));
  return det > 0.0;
}

void validate(Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  if (opts.source == Source::kFiles) {
    if (opts.clouds.empty()) {
      throw std::invalid_argument("--source files needs --clouds DIR");
    }
    if (!opts.width_given || !opts.height_given) {
      throw std::invalid_argument("--source files needs --width and --height");
    }
    if (!opts.intrinsics) {
      throw std::invalid_argument("--source files needs --fx, --fy, --cx and --cy");
    }
    if (opts.intrinsics_input || opts.host_input) {
      throw std::invalid_argument("--intrinsics-input and --host-input apply to synthetic clouds");
    }
    inspect_files(opts);
    if (!opts.count) {
      opts.count = static_cast<std::uint32_t>(opts.cloud_files.size());
    }
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    pcd::validate_config(file_source_config(opts));
  } else {
    if (!opts.clouds.empty()) {
      throw std::invalid_argument("--clouds needs --source files");
    }
    if (opts.intrinsics || opts.extrinsics) {
      throw std::invalid_argument(
          "the synthetic camera sets its own intrinsics and extrinsics; drop --fx..--cy and "
          "--extrinsics");
    }
    if (opts.depth_scale) {
      throw std::invalid_argument("synthetic uint16 depth is in millimeters; drop --depth-scale");
    }
    if (!opts.count) {
      opts.count = 100;
    }
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    pcd::validate_config(synthetic_source_config(opts));
    if (!opts.intrinsics_input) {
      opts.intrinsics = pcd::synthetic_frame(opts.width, opts.height, 0, false).intrinsics;
    }
    // The clouds are in the body frame; the operator moves them back.
    opts.extrinsics = pcd::rigid_inverse(pcd::synthetic_body_from_camera());
  }
  if (opts.depth_scale && opts.depth_format != pcd::DepthFormat::kUint16) {
    throw std::invalid_argument("--depth-scale applies to --depth-format uint16");
  }
  if (!opts.image_dir.empty()) {
    std::error_code error;
    if (!std::filesystem::is_directory(opts.image_dir, error)) {
      throw std::invalid_argument("--image-dir is not a directory: " + opts.image_dir.string() +
                                  (error ? " (" + error.message() + ")" : ""));
    }
  }
  // The operator's own checks, such as the intrinsics and the depth range.
  pcd::validate_config(projection_config(opts));
  // The operator takes any finite [R | t]; the application promises a camera
  // pose, which a scale, a shear or a mirror is not.
  if (opts.source == Source::kFiles && opts.extrinsics && !is_rotation(*opts.extrinsics)) {
    throw std::invalid_argument(
        "--extrinsics must be a rigid transform: R must be a rotation (orthonormal, det(R) = +1)");
  }
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const std::uint32_t frames = *opts.count;
  const bool color = color_enabled(opts);
  const holoscan::Output<holoscan::Tensor>* points_out = nullptr;
  const holoscan::Output<holoscan::Tensor>* colors_out = nullptr;
  const holoscan::Output<holoscan::Tensor>* intrinsics_out = nullptr;
  std::optional<holoscan::OperatorHandle<pcd::SyntheticCloudSourceOp>> synthetic;
  std::optional<holoscan::OperatorHandle<pcd::CloudFileSourceOp>> files;
  if (opts.source == Source::kSynthetic) {
    synthetic.emplace(
        graph.op<pcd::SyntheticCloudSourceOp>("source", synthetic_source_config(opts)));
    points_out = &(*synthetic)->points;
    colors_out = &(*synthetic)->colors;
    intrinsics_out = &(*synthetic)->intrinsics;
  } else {
    files.emplace(graph.op<pcd::CloudFileSourceOp>("source", file_source_config(opts)));
    points_out = &(*files)->points;
    colors_out = &(*files)->colors;
  }

  const pcd::PointCloudToDepthConfig projection = projection_config(opts);
  auto project = graph.op<pcd::PointCloudToDepthOp>("project", projection);
  auto report = std::make_shared<pcd::DepthReport>();
  std::optional<pcd::DepthImageTruth> truth;
  if (opts.source == Source::kSynthetic && opts.verify) {
    truth = pcd::DepthImageTruth{
        .width = opts.width,
        .height = opts.height,
        .scene = pcd::SceneTruth{.millimeter_depth = opts.depth_format == pcd::DepthFormat::kUint16,
                                 .format = projection.depth_format,
                                 .depth_scale = projection.depth_scale,
                                 .depth_min = projection.depth_min,
                                 .depth_max = projection.depth_max,
                                 .point_size = projection.point_size}};
  }
  auto reporter = graph.op<pcd::DepthImageReportOp>(
      "report", pcd::DepthImageReportConfig{.report = report,
                                            .expected_frames = frames,
                                            .depth_format = projection.depth_format,
                                            .depth_scale = projection.depth_scale,
                                            .color = color,
                                            .truth = truth,
                                            .image_dir = opts.image_dir});

  // Lossless edges: every image is checked, in order, against its frame. The
  // queue depth is the one each input declares in its operator's setup().
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(*points_out, project->points, lossless);
  if (color) {
    graph.add_flow(*colors_out, project->colors, lossless);
    graph.add_flow(project->color, reporter->color, lossless);
  }
  if (opts.intrinsics_input) {
    graph.add_flow(*intrinsics_out, project->intrinsics, lossless);
  }
  graph.add_flow(project->depth, reporter->depth, lossless);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // GPU outputs are on device 0.
  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("project", "depth");
  if (color) {
    on_gpu("project", "color");
  }
  if (opts.source == Source::kSynthetic && !opts.host_input) {
    on_gpu("source", "points");
    if (color) {
      on_gpu("source", "colors");
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
  // The run times out when no image arrives for this long, however long it
  // takes in all: checking large images against the truth is slow, but it
  // goes on. Even 2^32 - 1 seconds fit the clock's nanoseconds, and the
  // period is at most 1000 ms (validate()), so the sum cannot overflow.
  const auto patience = std::chrono::steady_clock::duration(
      std::chrono::seconds(opts.timeout_seconds) + std::chrono::milliseconds(2 * opts.period_ms));
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  bool timed_out = false;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint32_t reported = report->frames.load(); reported != seen) {
      seen = reported;
      last_progress = now;
    } else if (now - last_progress >= patience) {
      timed_out = true;
      break;
    }
  }
  if (timed_out) {
    std::cerr << kAppName << ": timed out: no image for " << opts.timeout_seconds << " s\n";
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
  // Without a truth to check them against, images in which no point landed
  // most likely mean wrong intrinsics or extrinsics; they do not pass.
  const bool all_empty = !report->verified && report->frames != 0 && report->filled_pixels == 0;
  if (all_empty) {
    std::cerr << kAppName
              << ": warning: no point reached the image; check the intrinsics, the extrinsics "
                 "and the depth range\n";
  }
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      report->frames == frames && report->passed() && !all_empty;
  std::cout << std::fixed << std::setprecision(3) << kAppName
            << ": frames=" << report->frames.load() << '/' << frames << " pixels=" << report->pixels
            << " filled=" << report->filled_pixels << " depth_m=[" << report->min_depth_m << ", "
            << report->max_depth_m << ']';
  if (!opts.image_dir.empty()) {
    std::cout << " files_written=" << report->files_written;
  }
  if (report->frame_gaps != 0) {
    std::cout << " frame_gaps=" << report->frame_gaps;
  }
  if (report->verified) {
    std::cout << " truth: wrong_depths=" << report->wrong_depths
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
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
