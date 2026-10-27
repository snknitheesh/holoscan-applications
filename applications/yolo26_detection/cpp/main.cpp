// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// YOLO26 Detection: images in, object detections out, on the GPU. A source
// (synthetic frames with known detections, JPEG and PPM files, or a webcam)
// feeds Yolo26DetectionOp, whose detections a reporter checks, prints, times
// and optionally draws onto the images, which it writes as PPM files. A
// HoloViz window can show the frames with their detections drawn over them.

#include <algorithm>
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
#if YOLO26_DETECTION_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>

#include "operators/detection_overlay_op.hpp"
#endif

#include "detection/image_files.hpp"
#include "detection/matching.hpp"
#include "detection/stub_model.hpp"
#include "detection/synthetic_scene.hpp"
#include "operators/detection_report_op.hpp"
#include "operators/image_sources.hpp"
#include "operators/yolo26_detection_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace yd = yolo26_detection;

constexpr std::string_view kAppName = "yolo26_detection";
constexpr std::string_view kStubModel = "stub";

enum class Source { kSynthetic, kFiles, kCamera };
enum class DisplayMode { kAuto, kOn, kOff };

struct Options {
  Source source{Source::kSynthetic};
  std::filesystem::path images;
  // Found by validate() for --source files.
  std::vector<std::filesystem::path> image_files;
  std::filesystem::path device{"/dev/video0"};
  bool loop{false};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  std::optional<int> width;
  std::optional<int> height;
  yd::PixelFormat encoding{yd::PixelFormat::kRgb8};
  bool host_input{false};
  std::string model;
  int input_size{640};
  float score_threshold{0.25F};
  int max_detections{300};
  std::optional<int> num_classes;
  bool fp16{false};
  std::filesystem::path labels_file;
  std::vector<std::string> labels;
  std::filesystem::path expect_file;
  std::vector<yd::Expectation> expectations;
  float expect_iou{0.8F};
  std::filesystem::path image_dir;
  std::optional<bool> verify;
  std::uint32_t timeout_seconds{120};
  std::uint32_t startup_timeout_seconds{900};
  DisplayMode display{DisplayMode::kAuto};
  bool compile_only{false};
  bool help{false};
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool display_available() {
  return env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
}

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

yd::PixelFormat parse_encoding(std::string_view value) {
  if (value == "rgb8") return yd::PixelFormat::kRgb8;
  if (value == "bgr8") return yd::PixelFormat::kBgr8;
  if (value == "rgba8") return yd::PixelFormat::kRgba8;
  if (value == "bgra8") return yd::PixelFormat::kBgra8;
  throw std::invalid_argument("--encoding must be rgb8, bgr8, rgba8 or bgra8");
}

Options parse_args(int argc, char** argv) {
  Options opts;
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
      } else if (value == "camera") {
        opts.source = Source::kCamera;
      } else {
        throw std::invalid_argument("--source must be synthetic, files or camera");
      }
    } else if (option == "--images") {
      opts.images = value;
    } else if (option == "--device") {
      opts.device = value;
    } else if (option == "--loop") {
      opts.loop = parse_bool(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--encoding") {
      opts.encoding = parse_encoding(value);
    } else if (option == "--host-input") {
      opts.host_input = parse_bool(value, option);
    } else if (option == "--model") {
      opts.model = value;
    } else if (option == "--input-size") {
      opts.input_size = parse_number<int>(value, option);
    } else if (option == "--score-threshold") {
      opts.score_threshold = parse_number<float>(value, option);
    } else if (option == "--max-detections") {
      opts.max_detections = parse_number<int>(value, option);
    } else if (option == "--num-classes") {
      opts.num_classes = parse_number<int>(value, option);
    } else if (option == "--fp16") {
      opts.fp16 = parse_bool(value, option);
    } else if (option == "--labels") {
      opts.labels_file = value;
    } else if (option == "--expect") {
      opts.expect_file = value;
    } else if (option == "--expect-iou") {
      opts.expect_iou = parse_number<float>(value, option);
    } else if (option == "--image-dir") {
      opts.image_dir = value;
    } else if (option == "--verify") {
      opts.verify = parse_bool(value, option);
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--startup-timeout-seconds") {
      opts.startup_timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--display") {
      opts.display = value == "auto"             ? DisplayMode::kAuto
                     : parse_bool(value, option) ? DisplayMode::kOn
                                                 : DisplayMode::kOff;
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " --model PATH|stub [options]\n"
         "Model:\n"
         "  --model PATH|stub          An end-to-end YOLO26 .onnx export (its TensorRT engine is\n"
         "                             built once and cached beside it) or a TensorRT engine;\n"
         "                             'stub' writes and uses a stand-in model with known\n"
         "                             detections, which needs no download\n"
         "  --input-size N             The model's input side, a multiple of 32 (default: 640)\n"
         "  --score-threshold F        Minimum score in [0, 1] (default: 0.25)\n"
         "  --max-detections N         Detections kept per image (default: 300)\n"
         "  --num-classes N            The model's classes (default: 80; stub: 2)\n"
         "  --fp16 true|false          Build the engine with FP16 allowed (default: false)\n"
         "  --labels FILE              Class names, one per line, for printing\n"
         "Input:\n"
         "  --source synthetic|files|camera\n"
         "                             Synthetic frames, image files, or a V4L2 webcam\n"
         "                             (default: synthetic)\n"
         "  --images PATH              A JPEG or PPM file, or a folder of them, in name order\n"
         "  --device PATH              The webcam (default: /dev/video0)\n"
         "  --loop true|false          Repeat the files until --count frames (default: false)\n"
         "  --count N                  Frames (default: 100 synthetic, every file once,\n"
         "                             camera until Ctrl+C)\n"
         "  --period-ms N              Source period, 1..1000 ms (default: 33)\n"
         "  --encoding rgb8|bgr8|rgba8|bgra8\n"
         "                             Image encoding (default: rgb8; files: rgb8 or bgr8)\n"
         "  --width N --height N       Synthetic frame size (default: 1280x720), or a YUYV\n"
         "                             size the camera offers (default: 640x480)\n"
         "  --host-input true|false    Send synthetic frames in host memory (default: false)\n"
         "Output:\n"
         "  --verify true|false        Check synthetic frames against their known detections\n"
         "                             (default: true with --model stub)\n"
         "  --expect FILE              Lines of image,class_id,x1,y1,x2,y2: detections the\n"
         "                             named files must yield\n"
         "  --expect-iou F             Overlap an expected box needs (default: 0.8)\n"
         "  --image-dir DIR            Write each frame with its boxes as\n"
         "                             DIR/detections_NNNNNN.ppm\n"
         "  --display auto|true|false  Show the frames with their detections in a HoloViz\n"
         "                             window, for synthetic frames and the camera (default:\n"
         "                             auto, on for the camera when a display is available)\n"
         "  --compile-only true|false  Build and check the graph without running it\n"
         "  --timeout-seconds N        Fail when no result arrives for N seconds plus two\n"
         "                             source periods (default: 120)\n"
         "  --startup-timeout-seconds N\n"
         "                             The same before the first result, which waits for the\n"
         "                             engine build (default: 900)\n";
}

bool is_stub(const Options& opts) { return opts.model == kStubModel; }

// Synthetic frames, or the camera's once validate() has set its size.
int frame_width(const Options& opts) { return opts.width.value_or(1280); }
int frame_height(const Options& opts) { return opts.height.value_or(720); }

yd::SyntheticImageSourceConfig synthetic_source_config(const Options& opts) {
  return yd::SyntheticImageSourceConfig{.width = frame_width(opts),
                                        .height = frame_height(opts),
                                        .count = opts.count.value_or(0),
                                        .period = std::chrono::milliseconds(opts.period_ms),
                                        .format = opts.encoding,
                                        .device_output = !opts.host_input};
}

yd::ImageFileSourceConfig file_source_config(const Options& opts) {
  return yd::ImageFileSourceConfig{.files = opts.image_files,
                                   .count = opts.count.value_or(0),
                                   .loop = opts.loop,
                                   .period = std::chrono::milliseconds(opts.period_ms),
                                   .bgr = opts.encoding == yd::PixelFormat::kBgr8};
}

yd::CameraSourceConfig camera_source_config(const Options& opts) {
  return yd::CameraSourceConfig{.device = opts.device,
                                .width = frame_width(opts),
                                .height = frame_height(opts),
                                .count = opts.count.value_or(0)};
}

// The detector's settings; the stub's model path is filled in by execute().
yd::Yolo26DetectionConfig detection_config(const Options& opts) {
  return yd::Yolo26DetectionConfig{
      .model = is_stub(opts) ? std::filesystem::path("yolo26_stub.onnx")
                             : std::filesystem::path(opts.model),
      .input_size = opts.input_size,
      .score_threshold = opts.score_threshold,
      .max_detections = opts.max_detections,
      .num_classes = opts.num_classes.value_or(is_stub(opts) ? yd::kStubClasses : 80),
      .fp16 = opts.fp16};
}

void validate(Options& opts) {
  if (opts.model.empty()) {
    throw std::invalid_argument("--model is required: a YOLO26 .onnx file, an engine, or stub");
  }
  if (!is_stub(opts)) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(opts.model, error)) {
      throw std::invalid_argument("--model: no such file: " + opts.model +
                                  "; see the README to prepare the model, or use --model stub");
    }
  } else if (opts.num_classes && *opts.num_classes != yd::kStubClasses) {
    throw std::invalid_argument("the stub model has 2 classes; drop --num-classes");
  }
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0 ||
      opts.startup_timeout_seconds == 0) {
    throw std::invalid_argument(
        "--period-ms must be 1..1000, and --timeout-seconds and --startup-timeout-seconds "
        "positive");
  }
  if (opts.source == Source::kFiles) {
    if (opts.images.empty()) {
      throw std::invalid_argument("--source files needs --images PATH");
    }
    if (opts.width || opts.height || opts.host_input) {
      throw std::invalid_argument("--width, --height and --host-input apply to synthetic frames");
    }
    if (opts.encoding != yd::PixelFormat::kRgb8 && opts.encoding != yd::PixelFormat::kBgr8) {
      throw std::invalid_argument("image files are decoded as rgb8 or bgr8");
    }
    if (opts.verify.value_or(false)) {
      throw std::invalid_argument("--verify checks synthetic frames; use --expect for files");
    }
    opts.verify = false;
    try {
      opts.image_files = yd::list_image_files(opts.images);
      for (const auto& file : opts.image_files) {
        (void)yd::read_image_info(file);
      }
    } catch (const std::runtime_error& error) {
      throw std::invalid_argument(error.what());
    }
    if (!opts.count) {
      opts.count = static_cast<std::uint32_t>(opts.image_files.size());
    }
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    yd::validate_config(file_source_config(opts));
  } else if (opts.source == Source::kCamera) {
    if (!opts.images.empty() || opts.loop) {
      throw std::invalid_argument("--images and --loop need --source files");
    }
    if (!opts.expect_file.empty() || opts.verify.value_or(false)) {
      throw std::invalid_argument("--expect and --verify do not apply to the camera");
    }
    if (opts.host_input || opts.encoding != yd::PixelFormat::kRgb8) {
      throw std::invalid_argument("the camera gives rgb8 frames in host memory");
    }
    if (opts.count && *opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    opts.verify = false;
    opts.width = opts.width.value_or(640);
    opts.height = opts.height.value_or(480);
    yd::check_camera(camera_source_config(opts));
  } else {
    if (!opts.images.empty() || opts.loop) {
      throw std::invalid_argument("--images and --loop need --source files");
    }
    if (!opts.expect_file.empty()) {
      throw std::invalid_argument("--expect applies to image files");
    }
    if (opts.verify.value_or(false) && !is_stub(opts)) {
      throw std::invalid_argument(
          "--verify needs --model stub: only its detections of synthetic frames are known");
    }
    opts.verify = opts.verify.value_or(is_stub(opts));
    if (!opts.count) {
      opts.count = 100;
    }
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive");
    }
    yd::validate_config(synthetic_source_config(opts));
  }
  if (!(opts.expect_iou > 0.0F && opts.expect_iou <= 1.0F)) {
    throw std::invalid_argument("--expect-iou must be in (0, 1]");
  }
  try {
    if (!opts.labels_file.empty()) {
      opts.labels = yd::read_labels(opts.labels_file);
    }
    if (!opts.expect_file.empty()) {
      opts.expectations = yd::read_expectations(opts.expect_file);
    }
  } catch (const std::runtime_error& error) {
    throw std::invalid_argument(error.what());
  }
  if (!opts.image_dir.empty()) {
    std::error_code error;
    if (!std::filesystem::is_directory(opts.image_dir, error)) {
      throw std::invalid_argument("--image-dir is not a directory: " + opts.image_dir.string() +
                                  (error ? " (" + error.message() + ")" : ""));
    }
  }
  yd::validate_config(detection_config(opts));

  // The display: on for the camera when a display is available, unless set.
  if (opts.display == DisplayMode::kAuto) {
    opts.display = YOLO26_DETECTION_ENABLE_VISUALIZATION && opts.source == Source::kCamera &&
                           display_available()
                       ? DisplayMode::kOn
                       : DisplayMode::kOff;
  }
  if (opts.display == DisplayMode::kOn) {
    if (opts.source == Source::kFiles || opts.encoding != yd::PixelFormat::kRgb8) {
      throw std::invalid_argument("--display shows rgb8 synthetic frames and the camera");
    }
    if (!YOLO26_DETECTION_ENABLE_VISUALIZATION) {
      throw std::invalid_argument("--display needs a build with visualization enabled");
    }
    if (!display_available() && !opts.compile_only) {
      throw std::invalid_argument(
          "the display needs a display server; set DISPLAY or add --display false");
    }
  }
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  yd::Yolo26DetectionConfig detection = detection_config(opts);
  if (is_stub(opts)) {
    // Written to the working directory, where its engine is cached beside it.
    detection.model = yd::write_stub_model(std::filesystem::current_path(), opts.input_size);
  }

  holoscan::Graph graph{std::string(kAppName)};
  // Frames the run expects; 0 for a camera that runs until stopped.
  const std::uint32_t frames = opts.count.value_or(0);
  const holoscan::Output<holoscan::schema::ImageT>* image_out = nullptr;
  if (opts.source == Source::kSynthetic) {
    image_out =
        &graph.op<yd::SyntheticImageSourceOp>("source", synthetic_source_config(opts))->image;
  } else if (opts.source == Source::kFiles) {
    image_out = &graph.op<yd::ImageFileSourceOp>("source", file_source_config(opts))->image;
  } else {
    image_out = &graph.op<yd::CameraSourceOp>("source", camera_source_config(opts))->image;
  }
  auto detector = graph.op<yd::Yolo26DetectionOp>("detector", detection);

  auto report = std::make_shared<yd::DetectionReport>();
  std::optional<yd::SyntheticTruth> truth;
  if (opts.source == Source::kSynthetic && opts.verify.value_or(false)) {
    truth = yd::SyntheticTruth{
        .width = frame_width(opts), .height = frame_height(opts), .input_size = opts.input_size};
  }
  auto reporter = graph.op<yd::DetectionReportOp>(
      "report", yd::DetectionReportConfig{.report = report,
                                          .expected_frames = frames,
                                          .max_detections = opts.max_detections,
                                          .truth = truth,
                                          .files = opts.source == Source::kFiles
                                                       ? opts.image_files
                                                       : std::vector<std::filesystem::path>{},
                                          .expectations = opts.expectations,
                                          .expect_iou = opts.expect_iou,
                                          .labels = opts.labels,
                                          .print = opts.source == Source::kFiles,
                                          .image_dir = opts.image_dir});

  // Lossless edges: every frame is detected and reported, in order. The
  // queue depth is the one each input declares in its operator's setup().
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(*image_out, detector->image, lossless);
  graph.add_flow(detector->boxes, reporter->boxes, lossless);
  graph.add_flow(detector->scores, reporter->scores, lossless);
  graph.add_flow(detector->class_ids, reporter->class_ids, lossless);
  graph.add_flow(detector->count, reporter->count, lossless);
  if (!opts.image_dir.empty()) {
    graph.add_flow(*image_out, reporter->image, lossless);
  }

#if YOLO26_DETECTION_ENABLE_VISUALIZATION
  if (opts.display == DisplayMode::kOn) {
    namespace viz = holoscan::visualization;
    const auto width = static_cast<std::uint32_t>(frame_width(opts));
    const auto height = static_cast<std::uint32_t>(frame_height(opts));
    viz::LayerStack layers;
    layers.add_image_layer("video", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                                        .max_width = width,
                                                        .max_height = height});
    // HoloViz blends RGBA image layers by their alpha, so the boxes go over the video.
    layers.add_image_layer("boxes", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                                        .max_width = width,
                                                        .max_height = height});
    layers.add_text_layer(
        "labels",
        viz::TextLayerDesc{.max_glyphs = 2U * static_cast<std::uint32_t>(opts.max_detections) *
                                         yd::kOverlayLabelBytes,
                           .normalized_coordinates = true});
    // Draw when the boxes arrive, over the newest image, which is theirs;
    // triggering on the image would draw the previous frame's boxes.
    const auto max_input_age =
        std::max(4 * std::chrono::milliseconds(opts.period_ms), std::chrono::milliseconds{250});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "YOLO26 Detection", .width = width, .height = height},
        viz::FrameTrigger{viz::LayerTrigger{1}}, max_input_age);
    auto overlay = graph.op<yd::DetectionOverlayOp>(
        "overlay", yd::DetectionOverlayConfig{.width = frame_width(opts),
                                              .height = frame_height(opts),
                                              .max_detections = opts.max_detections,
                                              .labels = opts.labels});
    // Best effort: the display keeps the newest sample and never holds back
    // the detector.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(*image_out, visualizer->image_layer_input(0), latest);
    graph.add_flow(detector->boxes, overlay->boxes, latest);
    graph.add_flow(detector->scores, overlay->scores, latest);
    graph.add_flow(detector->class_ids, overlay->class_ids, latest);
    graph.add_flow(detector->count, overlay->count, latest);
    graph.add_flow(overlay->overlay, visualizer->image_layer_input(1), latest);
    graph.add_flow(overlay->labels, visualizer->tensor_layer_input(2), latest);
  }
#endif
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // GPU outputs are on device 0.
  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  for (const char* port : {"boxes", "scores", "class_ids", "count"}) {
    on_gpu("detector", port);
  }
  if (opts.source == Source::kFiles || (opts.source == Source::kSynthetic && !opts.host_input)) {
    on_gpu("source", "image");
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
  // Build the TensorRT engine now rather than in start(): see prepare_engine().
  yd::prepare_engine(detection);
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // The run times out when no result arrives for this long. Before the first
  // one, the detector may still be building its engine. The sums cannot
  // overflow: the period is at most 1000 ms (validate()).
  const auto margin = std::chrono::milliseconds(2 * opts.period_ms);
  const auto patience =
      std::chrono::steady_clock::duration(std::chrono::seconds(opts.timeout_seconds) + margin);
  const auto startup_patience = std::chrono::steady_clock::duration(
      std::chrono::seconds(opts.startup_timeout_seconds) + margin);
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  std::optional<std::uint32_t> timed_out_after;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint32_t reported = report->frames.load(); reported != seen) {
      seen = reported;
      last_progress = now;
    } else if (now - last_progress >= (seen == 0 ? startup_patience : patience)) {
      timed_out_after = seen == 0 ? opts.startup_timeout_seconds : opts.timeout_seconds;
      break;
    }
  }
  if (timed_out_after) {
    std::cerr << kAppName << ": timed out: no result for " << *timed_out_after << " s\n";
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

  // A stop request (SIGINT, SIGTERM, a closed window) ends the run early;
  // that is not a failure, but not a pass either. For a live camera it is the
  // normal end.
  const bool stopped = !timed_out_after && termination == holoscan::RunTermination::kStopped;
  // Files with nothing to compare against and no detection at all most likely
  // mean a wrong model, input size or threshold; they do not pass.
  const bool none_found = opts.source == Source::kFiles && opts.expectations.empty() &&
                          report->frames != 0 && report->detections == 0;
  if (none_found) {
    std::cerr << kAppName
              << ": warning: no detection in any image; check the model, its input size and the "
                 "score threshold\n";
  }
  const bool passed = !timed_out_after && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      report->frames == frames && report->passed() && !none_found;
  const std::vector<double>& latencies = report->latencies_ms;
  double mean = 0.0;
  for (const double value : latencies) {
    mean += value / static_cast<double>(latencies.size());
  }
  std::cout << std::fixed << std::setprecision(3) << kAppName
            << ": frames=" << report->frames.load();
  if (frames != 0) {
    std::cout << '/' << frames;
  }
  std::cout << " detections=" << report->detections;
  if (!latencies.empty()) {
    std::cout << " latency_ms: mean=" << mean << " p50=" << percentile(latencies, 0.5)
              << " p99=" << percentile(latencies, 0.99)
              << " max=" << *std::max_element(latencies.begin(), latencies.end());
  }
  if (!opts.image_dir.empty()) {
    std::cout << " files_written=" << report->files_written;
  }
  if (report->frame_gaps != 0) {
    std::cout << " frame_gaps=" << report->frame_gaps;
  }
  if (report->verified) {
    std::cout << " truth: missed=" << report->missed << " extra=" << report->extra
              << " max_edge_error_px=" << report->max_edge_error;
  }
  if (report->expected != 0) {
    std::cout << " expected: matched=" << report->matched << '/' << report->expected;
  }
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed || (stopped && frames == 0) ? 0 : 1;
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
