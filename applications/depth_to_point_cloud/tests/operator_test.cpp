// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DepthToPointCloudOp on its own, through the SDK's operator test harness:
// float32 and uint16 depth from host and device memory, with aligned color in
// every encoding and per-frame intrinsics, every point and color checked
// against the synthetic camera; the capture time and frame id carried to the
// outputs; frames whose intrinsics do not match their capture time; and
// unusable inputs and configurations rejected. Also how host input is held
// until its copy is done, and the reporter's frame-id check.

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "deproject/synthetic_depth.hpp"
#include "operators/depth_to_point_cloud_op.hpp"
#include "operators/point_cloud_report_op.hpp"
#include "test_support.hpp"

namespace depth_to_point_cloud {
namespace {

using std::chrono_literals::operator""ms;

constexpr int kWidth = 96;
constexpr int kHeight = 72;

// A tensor of `shape` holding `bytes`, in host or device memory.
holoscan::Tensor make(const std::vector<std::int64_t>& shape, DLDataType dtype, const void* bytes,
                      std::size_t size, bool device) {
  auto tensor = holoscan::make_tensor(
      shape, dtype, std::nullopt,
      device ? holoscan::MemoryStorageType::kDevice : holoscan::MemoryStorageType::kHost,
      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor) {
    throw std::runtime_error(tensor.error().message);
  }
  auto destination = tensor->mutable_data();
  if (!destination) {
    throw std::runtime_error(destination.error().message);
  }
  if (cudaMemcpy(*destination, bytes, size, cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return std::move(*tensor);
}

// Pinned host or device memory whose allocator the test plays: it counts the
// streams it is asked to wait for before reuse (and can refuse to wait), and
// the host waits for the data to be ready. Without `ready_on_stream`, it is
// host-written memory that no stream makes ready.
class TestMemory final : public holoscan::TensorMemoryReference {
 public:
  TestMemory(std::size_t bytes, bool device, bool waits_for_streams, bool ready_on_stream = true)
      : device_(device), waits_(waits_for_streams), ready_on_stream_(ready_on_stream) {
    if ((device ? cudaMalloc(&base_, bytes) : cudaMallocHost(&base_, bytes)) != cudaSuccess) {
      throw std::runtime_error("allocation failed");
    }
  }
  ~TestMemory() override { device_ ? cudaFree(base_) : cudaFreeHost(base_); }
  TestMemory(const TestMemory&) = delete;
  TestMemory& operator=(const TestMemory&) = delete;

  bool set_deallocation_stream(cudaStream_t stream) override {
    if (waits_ && stream != nullptr) {
      streams_.fetch_add(1);
    }
    return waits_;
  }
  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override { return waits_; }
  // The data is written before the tensor is made, so readiness is immediate.
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return ready_on_stream_;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> synchronize_for_host() override {
    host_waits_.fetch_add(1);
    return {};
  }

  [[nodiscard]] std::uint8_t* base() const { return static_cast<std::uint8_t*>(base_); }
  [[nodiscard]] int streams() const { return streams_.load(); }
  [[nodiscard]] int host_waits() const { return host_waits_.load(); }

 private:
  void* base_{};
  bool device_{};
  bool waits_{};
  bool ready_on_stream_{};
  std::atomic<int> streams_{0};
  std::atomic<int> host_waits_{0};
};

// A packed tensor of `shape` at `offset` bytes into `memory`.
holoscan::Tensor wrap(const std::shared_ptr<TestMemory>& memory, std::size_t offset,
                      const std::vector<std::int64_t>& shape, DLDataType dtype, bool device) {
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = shape;
  context->dl_strides.assign(shape.size(), 1);
  for (std::size_t axis = shape.size() - 1; axis-- > 0;) {
    context->dl_strides[axis] = context->dl_strides[axis + 1] * shape[axis + 1];
  }
  DLTensor& tensor = context->tensor.dl_tensor;
  tensor.data = memory->base() + offset;
  tensor.device = DLDevice{device ? kDLCUDA : kDLCUDAHost, 0};
  tensor.ndim = static_cast<std::int32_t>(shape.size());
  tensor.dtype = dtype;
  tensor.shape = context->dl_shape.data();
  tensor.strides = context->dl_strides.data();
  tensor.byte_offset = 0;
  context->memory_reference = memory;
  return holoscan::Tensor(std::move(context));
}

holoscan::schema::ImageT image(holoscan::Tensor tensor, int width, int height,
                               holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT result;
  result.width = width;
  result.height = height;
  result.encoding = encoding;
  result.data = std::make_shared<holoscan::Tensor>(std::move(tensor));
  return result;
}

holoscan::schema::ImageT depth_image(const SyntheticFrame& f, bool millimeters, bool device) {
  const std::size_t n = static_cast<std::size_t>(f.width) * f.height;
  if (millimeters) {
    std::vector<std::uint16_t> mm(n);
    for (int v = 0; v < f.height; ++v)
      for (int u = 0; u < f.width; ++u) mm[v * f.width + u] = synthetic_depth_mm(f, u, v);
    return image(make({f.height, f.width}, {kDLUInt, 16, 1}, mm.data(), n * 2, device), f.width,
                 f.height, holoscan::schema::ImageEncoding_MONO16);
  }
  std::vector<float> m(n);
  for (int v = 0; v < f.height; ++v)
    for (int u = 0; u < f.width; ++u) m[v * f.width + u] = synthetic_depth(f, u, v);
  return image(make({f.height, f.width, 1}, {kDLFloat, 32, 1}, m.data(), n * 4, device), f.width,
               f.height, holoscan::schema::ImageEncoding_MONO32F);
}

// The synthetic camera's color in `encoding`: RGB8, BGR8, RGBA8 or BGRA8.
holoscan::schema::ImageT color_image(const SyntheticFrame& f,
                                     holoscan::schema::ImageEncoding encoding, bool device) {
  const bool bgr = encoding == holoscan::schema::ImageEncoding_BGR8 ||
                   encoding == holoscan::schema::ImageEncoding_BGRA8;
  const int channels = encoding == holoscan::schema::ImageEncoding_RGBA8 ||
                               encoding == holoscan::schema::ImageEncoding_BGRA8
                           ? 4
                           : 3;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(f.width) * f.height * channels);
  for (int v = 0; v < f.height; ++v) {
    for (int u = 0; u < f.width; ++u) {
      std::uint8_t rgb[3];
      synthetic_color(f, u, v, rgb);
      std::uint8_t* p = pixels.data() + (static_cast<std::size_t>(v) * f.width + u) * channels;
      p[0] = bgr ? rgb[2] : rgb[0];
      p[1] = rgb[1];
      p[2] = bgr ? rgb[0] : rgb[2];
      if (channels == 4) {
        p[3] = 255;
      }
    }
  }
  return image(
      make({f.height, f.width, channels}, {kDLUInt, 8, 1}, pixels.data(), pixels.size(), device),
      f.width, f.height, encoding);
}

// Checks a captured [H, W, 3] cloud (and colors) against frame `f`.
template <typename Snapshot>
void check_cloud(const Snapshot& cloud, const Snapshot* colors, const SyntheticFrame& f,
                 bool millimeters, const DepthToPointCloudConfig& config) {
  VERIFY(cloud.shape() == (std::vector<std::int64_t>{f.height, f.width, 3}));
  std::vector<float> xyz(static_cast<std::size_t>(f.width) * f.height * 3);
  VERIFY(cloud.bytes().size() == xyz.size() * sizeof(float));
  if (cloud.bytes().size() != xyz.size() * sizeof(float)) return;
  std::memcpy(xyz.data(), cloud.bytes().data(), cloud.bytes().size());
  int wrong = 0;
  int wrong_colors = 0;
  for (int v = 0; v < f.height; ++v) {
    for (int u = 0; u < f.width; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * f.width + u;
      const float z = millimeters ? synthetic_depth_mm(f, u, v) * config.depth_scale
                                  : synthetic_depth(f, u, v) * config.depth_scale;
      const bool valid = z > 0 && z >= config.depth_min && z <= config.depth_max;
      const float* p = xyz.data() + 3 * i;
      if (!valid) {
        const auto is_invalid = [&config](float value) {
          return std::isnan(config.invalid_value) ? std::isnan(value)
                                                  : value == config.invalid_value;
        };
        wrong += !(is_invalid(p[0]) && is_invalid(p[1]) && is_invalid(p[2]));
      } else {
        const CameraIntrinsics& k = f.intrinsics;
        wrong +=
            std::hypot(p[0] - (u - k.cx) * z / k.fx, p[1] - (v - k.cy) * z / k.fy, p[2] - z) > 1e-4;
      }
      if (colors != nullptr) {
        std::uint8_t rgb[3];
        synthetic_color(f, u, v, rgb);
        const auto* got = reinterpret_cast<const std::uint8_t*>(colors->bytes().data()) + 3 * i;
        wrong_colors += got[0] != rgb[0] || got[1] != rgb[1] || got[2] != rgb[2];
      }
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " points differ from the truth");
  VERIFY_MSG(wrong_colors == 0, std::to_string(wrong_colors) + " colors differ from the truth");
}

DepthToPointCloudConfig base_config(bool millimeters) {
  DepthToPointCloudConfig config;
  config.intrinsics = synthetic_frame(kWidth, kHeight, 0, false).intrinsics;
  config.depth_scale = millimeters ? 0.001F : 1.0F;
  config.depth_min = 0.1F;
  config.depth_max = 10.0F;
  config.max_width = kWidth;
  config.max_height = kHeight;
  return config;
}

// The capture time of frame `frame` in the tests' frames.
std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// Checks that a cloud carries frame `frame`'s capture time and `frame_id`.
template <typename Snapshot>
void check_provenance(const Snapshot& cloud, std::uint32_t frame, std::uint64_t frame_id) {
  const holoscan::SampleMetadata& metadata = cloud.sample_metadata();
  VERIFY_MSG(metadata.capture_timestamp_ns == capture_time(frame).count(),
             "frame " + std::to_string(frame) + " carries capture time " +
                 std::to_string(metadata.capture_timestamp_ns));
  VERIFY_MSG(
      metadata.frame_id == frame_id,
      "frame " + std::to_string(frame) + " carries frame id " + std::to_string(metadata.frame_id));
}

void check_depth_only(bool millimeters, bool device, float invalid_value = std::nanf("")) {
  DepthToPointCloudConfig config = base_config(millimeters);
  config.invalid_value = invalid_value;
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  std::vector<holoscan::testing::TestSample<holoscan::schema::ImageT>> samples;
  for (std::uint32_t frame = 0; frame < 2; ++frame) {
    holoscan::testing::TestSample<holoscan::schema::ImageT> sample = holoscan::testing::sample_at(
        depth_image(synthetic_frame(kWidth, kHeight, frame, false), millimeters, device),
        capture_time(frame));
    sample.metadata.frame_id = 41U + frame;
    samples.push_back(std::move(sample));
  }
  h.push_samples("depth", std::move(samples));
  if (device) {
    h.input_source_device("depth", holoscan::DeviceId{0});
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::Tensor>("point_cloud");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<holoscan::Tensor>("point_cloud");
  VERIFY(clouds.size() == 2);
  for (std::size_t frame = 0; frame < clouds.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_cloud<std::decay_t<decltype(clouds[frame])>>(
        clouds[frame], nullptr, synthetic_frame(kWidth, kHeight, number, false), millimeters,
        config);
    check_provenance(clouds[frame], number, 41U + number);
  }
}

holoscan::Tensor intrinsics_tensor(const SyntheticFrame& f, bool device) {
  const std::array<float, 4> k{f.intrinsics.fx, f.intrinsics.fy, f.intrinsics.cx, f.intrinsics.cy};
  return make({4}, {kDLFloat, 32, 1}, k.data(), sizeof(k), device);
}

DepthToPointCloudConfig color_and_intrinsics_config() {
  DepthToPointCloudConfig config = base_config(true);
  config.color = true;
  config.intrinsics.reset();
  config.intrinsics_input = true;
  return config;
}

// Color and per-frame intrinsics arrive aligned with the depth.
void check_color_and_intrinsics(holoscan::schema::ImageEncoding encoding, bool device) {
  const DepthToPointCloudConfig config = color_and_intrinsics_config();
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("depth", depth_image(f, true, device)),
        holoscan::testing::input("color", color_image(f, encoding, device)),
        holoscan::testing::input("intrinsics", intrinsics_tensor(f, device))));
  }
  if (device) {
    for (const char* port : {"depth", "color", "intrinsics"}) {
      h.input_source_device(port, holoscan::DeviceId{0});
    }
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.output_device("colors", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::Tensor>("point_cloud");
  h.capture_snapshot<holoscan::Tensor>("colors");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<holoscan::Tensor>("point_cloud");
  const auto colors = result.snapshots<holoscan::Tensor>("colors");
  VERIFY(clouds.size() == 3 && colors.size() == 3);
  for (std::size_t frame = 0; frame < clouds.size() && frame < colors.size(); ++frame) {
    const auto number = static_cast<std::uint32_t>(frame);
    check_cloud(clouds[frame], &colors[frame], synthetic_frame(kWidth, kHeight, number, true), true,
                config);
    check_provenance(clouds[frame], number, 0U);
    check_provenance(colors[frame], number, 0U);
  }
}

// Intrinsics pair with depth by exact capture time: a frame whose intrinsics
// carry another time never completes, and gives no cloud, while the frames
// around it still do.
void check_unmatched_intrinsics() {
  const DepthToPointCloudConfig config = color_and_intrinsics_config();
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  for (std::uint32_t frame = 0; frame < 3; ++frame) {
    const SyntheticFrame f = synthetic_frame(kWidth, kHeight, frame, true);
    // Frame 1's intrinsics are a millisecond late.
    const auto intrinsics_time =
        capture_time(frame) + std::chrono::milliseconds(frame == 1 ? 1 : 0);
    h.push_frame(holoscan::testing::frame_at(
        capture_time(frame), holoscan::testing::input("depth", depth_image(f, true, false)),
        holoscan::testing::input("color",
                                 color_image(f, holoscan::schema::ImageEncoding_RGB8, false))));
    h.push_frame(holoscan::testing::frame_at(
        intrinsics_time, holoscan::testing::input("intrinsics", intrinsics_tensor(f, false))));
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  h.output_device("colors", holoscan::DeviceId{0});
  h.capture_snapshot<holoscan::Tensor>("point_cloud");
  h.capture_snapshot<holoscan::Tensor>("colors");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto clouds = result.snapshots<holoscan::Tensor>("point_cloud");
  const auto colors = result.snapshots<holoscan::Tensor>("colors");
  VERIFY_MSG(clouds.size() == 2 && colors.size() == 2,
             std::to_string(clouds.size()) + " clouds and " + std::to_string(colors.size()) +
                 " color images from 2 matched frames");
  const std::array<std::uint32_t, 2> matched{0U, 2U};
  for (std::size_t i = 0; i < clouds.size() && i < colors.size() && i < matched.size(); ++i) {
    check_cloud(clouds[i], &colors[i], synthetic_frame(kWidth, kHeight, matched[i], true), true,
                config);
    check_provenance(clouds[i], matched[i], 0U);
    check_provenance(colors[i], matched[i], 0U);
  }
}

// Sleeps on `stream`, so that work queued after it is still pending a while.
void stall(cudaStream_t stream) {
  VERIFY(cudaLaunchHostFunc(
             stream, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(300)); },
             nullptr) == cudaSuccess);
}

// Host input is held until the stream that copies it is done: by its allocator
// when that can wait for a stream, without blocking; otherwise by waiting.
void check_host_input_held() {
  cudaStream_t stream{};
  VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  {
    auto memory = std::make_shared<TestMemory>(64, false, true);
    stall(stream);
    VERIFY(
        detail::hold_until_done(wrap(memory, 0, {64}, {kDLUInt, 8, 1}, false), stream).has_value());
    VERIFY_MSG(memory->streams() == 1, "the allocator is asked to wait for the stream");
    VERIFY_MSG(cudaStreamQuery(stream) == cudaErrorNotReady, "the host does not wait");
    VERIFY(cudaStreamSynchronize(stream) == cudaSuccess);
  }
  {
    auto memory = std::make_shared<TestMemory>(64, false, false);
    stall(stream);
    VERIFY(
        detail::hold_until_done(wrap(memory, 0, {64}, {kDLUInt, 8, 1}, false), stream).has_value());
    VERIFY_MSG(cudaStreamQuery(stream) == cudaSuccess,
               "the stream is done when the allocator cannot wait for it");
  }
  VERIFY(cudaStreamDestroy(stream) == cudaSuccess);
}

// The test harness copies staged inputs into its own storage, so a tensor whose
// memory the test controls reaches the operator only through a graph: this
// source sends the given frames as they are, one per tick, each with the
// given color image and intrinsics when there are any.
struct GivenFramesConfig {
  std::vector<holoscan::schema::ImageT> depth;
  std::vector<holoscan::schema::ImageT> color;
  std::vector<holoscan::Tensor> intrinsics;
};

class GivenFramesSourceOp final : public holoscan::Operator<> {
 public:
  explicit GivenFramesSourceOp(GivenFramesConfig config) : config_(std::move(config)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    const holoscan::PayloadOptions image_payload{
        .schema = holoscan::schema_identity<holoscan::schema::ImageT>()};
    spec.output(depth, "depth").payload_options(image_payload);
    if (!config_.color.empty()) {
      spec.output(color, "color").payload_options(image_payload);
    }
    if (!config_.intrinsics.empty()) {
      spec.output(intrinsics, "intrinsics");
    }
  }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(5)});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= config_.depth.size()) {
      return {};
    }
    // A frame keeps its capture time until all its parts are out.
    if (!capture_) {
      capture_ = context.activation_time();
    }
    const holoscan::EmitOptions options{.capture_time = *capture_, .frame_id = next_ + 1U};
    // Sends one part unless sent already; false when the consumer is busy.
    const auto send = [&](bool& sent, auto&& emit) -> holoscan::expected<bool, holoscan::Error> {
      if (sent) {
        return true;
      }
      auto result = emit();
      if (!result) {
        if (result.error().code == holoscan::ErrorCode::kBackpressured) {
          return false;
        }
        return holoscan::make_unexpected(std::move(result).error());
      }
      sent = true;
      return true;
    };
    const std::size_t i = next_;
    for (auto part : {0, 1, 2}) {
      holoscan::expected<bool, holoscan::Error> done = true;
      if (part == 0 && !config_.intrinsics.empty()) {
        done =
            send(sent_intrinsics_, [&] { return intrinsics.emit(config_.intrinsics[i], options); });
      } else if (part == 1 && !config_.color.empty()) {
        done = send(sent_color_, [&] { return color.emit(config_.color[i], options); });
      } else if (part == 2) {
        done = send(sent_depth_, [&] { return depth.emit(config_.depth[i], options); });
      }
      if (!done) {
        return holoscan::make_unexpected(std::move(done).error());
      }
      if (!*done) {
        return {};
      }
    }
    ++next_;
    capture_.reset();
    sent_intrinsics_ = sent_color_ = sent_depth_ = false;
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> depth;
  holoscan::Output<holoscan::schema::ImageT> color;
  holoscan::Output<holoscan::Tensor> intrinsics;

 private:
  GivenFramesConfig config_;
  std::uint64_t next_{};
  std::optional<holoscan::TimePoint> capture_;
  bool sent_intrinsics_{};
  bool sent_color_{};
  bool sent_depth_{};
};

// Runs `frames` through the operator into the reporter. Returns the reporter's
// report, and in `failure` the run's diagnostics when it did not complete.
std::shared_ptr<CloudReport> run_graph(const DepthToPointCloudConfig& config,
                                       GivenFramesConfig frames, std::string& failure) {
  holoscan::Graph graph{"depth_to_point_cloud_test"};
  const auto count = static_cast<std::uint32_t>(frames.depth.size());
  auto source = graph.op<GivenFramesSourceOp>("source", std::move(frames));
  auto cloud = graph.op<DepthToPointCloudOp>("deproject", config);
  auto report = std::make_shared<CloudReport>();
  auto reporter = graph.op<PointCloudReportOp>(
      "report",
      PointCloudReportConfig{.report = report, .expected_frames = count, .colors = config.color});
  graph.add_flow(source->depth, cloud->depth);
  if (config.color) {
    graph.add_flow(source->color, cloud->color);
    graph.add_flow(cloud->colors, reporter->colors);
  }
  if (config.intrinsics_input) {
    graph.add_flow(source->intrinsics, cloud->intrinsics);
  }
  graph.add_flow(cloud->point_cloud, reporter->point_cloud);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions options;
  for (const char* port : {"point_cloud", "colors"}) {
    if (std::string(port) == "point_cloud" || config.color) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "deproject", .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
  const auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    failure = holoscan::format_diagnostics(plan.diagnostics(), plan.graph());
    return report;
  }
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  if (session.termination() != holoscan::RunTermination::kRequestedCompleted) {
    failure = "run did not complete: " +
              holoscan::format_diagnostics(session.diagnostics(), plan.graph());
  }
  return report;
}

// The synthetic frame's depth in millimeters, at `offset` bytes into new
// pinned host or device memory.
std::pair<holoscan::schema::ImageT, std::shared_ptr<TestMemory>> given_depth(
    const SyntheticFrame& f, bool device, std::size_t offset = 0) {
  std::vector<std::uint16_t> mm(static_cast<std::size_t>(f.width) * f.height);
  for (int v = 0; v < f.height; ++v)
    for (int u = 0; u < f.width; ++u) mm[v * f.width + u] = synthetic_depth_mm(f, u, v);
  // Pinned depth is host-written: the runtime does not order its release.
  auto memory = std::make_shared<TestMemory>(mm.size() * 2U + offset, device, true, device);
  if (cudaMemcpy(memory->base() + offset, mm.data(), mm.size() * 2U, cudaMemcpyDefault) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return {image(wrap(memory, offset, {f.height, f.width}, {kDLUInt, 16, 1}, device), f.width,
                f.height, holoscan::schema::ImageEncoding_MONO16),
          memory};
}

// The frame's intrinsics at `offset` bytes into new pinned host or device memory.
std::pair<holoscan::Tensor, std::shared_ptr<TestMemory>> given_intrinsics(const SyntheticFrame& f,
                                                                          bool device,
                                                                          std::size_t offset = 0) {
  const std::array<float, 4> k{f.intrinsics.fx, f.intrinsics.fy, f.intrinsics.cx, f.intrinsics.cy};
  auto memory = std::make_shared<TestMemory>(sizeof(k) + offset, device, true);
  if (cudaMemcpy(memory->base() + offset, k.data(), sizeof(k), cudaMemcpyDefault) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
  return {wrap(memory, offset, {4}, {kDLFloat, 32, 1}, device), memory};
}

// Depth in pinned host memory reaches the operator as it is; the operator
// asks the memory to wait for the stream that copies it to the GPU.
void check_pinned_depth_held_by_operator() {
  const auto [depth, memory] = given_depth(synthetic_frame(kWidth, kHeight, 0, false), false);
  std::string failure;
  const auto report = run_graph(base_config(true), GivenFramesConfig{.depth = {depth}}, failure);
  VERIFY_MSG(failure.empty(), failure);
  VERIFY(report->frames == 1);
  VERIFY_MSG(memory->streams() > 0, "the operator asks the pinned depth's memory to wait");
}

// Intrinsics in pinned host memory are read by the host, so the operator waits
// for their producer on the host, not only on its stream.
void check_pinned_intrinsics_waited_for() {
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, true);
  const auto [depth, depth_memory] = given_depth(f, false);
  const auto [k, k_memory] = given_intrinsics(f, false);
  DepthToPointCloudConfig config = base_config(true);
  config.intrinsics.reset();
  config.intrinsics_input = true;
  std::string failure;
  const auto report =
      run_graph(config, GivenFramesConfig{.depth = {depth}, .intrinsics = {k}}, failure);
  VERIFY_MSG(failure.empty(), failure);
  VERIFY(report->frames == 1);
  VERIFY_MSG(k_memory->host_waits() > 0, "the host waits for pinned intrinsics before reading");
}

// Device data the kernel cannot load its elements from in place is rejected.
void check_misaligned_device_data() {
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, true);
  const auto expect_failure = [](const DepthToPointCloudConfig& config, GivenFramesConfig frames,
                                 const std::string& what) {
    std::string failure;
    static_cast<void>(run_graph(config, std::move(frames), failure));
    VERIFY_MSG(failure.find(what) != std::string::npos, "rejected with '" + what + "': " + failure);
  };
  {
    const auto [depth, memory] = given_depth(f, true, 1);
    expect_failure(base_config(true), GivenFramesConfig{.depth = {depth}},
                   "depth data must be aligned");
  }
  {
    const auto [depth, depth_memory] = given_depth(f, true);
    const auto [k, k_memory] = given_intrinsics(f, true, 2);
    DepthToPointCloudConfig config = base_config(true);
    config.intrinsics.reset();
    config.intrinsics_input = true;
    expect_failure(config, GivenFramesConfig{.depth = {depth}, .intrinsics = {k}},
                   "intrinsics data must be aligned");
  }
  {
    const auto [depth, depth_memory] = given_depth(f, true);
    holoscan::schema::ImageT bgra = color_image(f, holoscan::schema::ImageEncoding_BGRA8, false);
    const std::size_t bytes = static_cast<std::size_t>(f.width) * f.height * 4U;
    auto memory = std::make_shared<TestMemory>(bytes + 2U, true, true);
    VERIFY(cudaMemcpy(memory->base() + 2, bgra.data->data(), bytes, cudaMemcpyDefault) ==
           cudaSuccess);
    bgra.data = std::make_shared<holoscan::Tensor>(
        wrap(memory, 2, {f.height, f.width, 4}, {kDLUInt, 8, 1}, true));
    DepthToPointCloudConfig config = base_config(true);
    config.color = true;
    expect_failure(config, GivenFramesConfig{.depth = {depth}, .color = {bgra}},
                   "color data must be aligned");
  }
}

// Runs the operator on one unusable input and checks the failure names `what`.
void check_rejects(const DepthToPointCloudConfig& config, holoscan::schema::ImageT depth,
                   std::optional<holoscan::schema::ImageT> color, const std::string& what) {
  auto h = holoscan::testing::create_operator_test<DepthToPointCloudOp>(config);
  if (color) {
    h.push_frame(holoscan::testing::frame_at(100ms, holoscan::testing::input("depth", depth),
                                             holoscan::testing::input("color", *color)));
  } else {
    h.push("depth", depth);
  }
  h.output_device("point_cloud", holoscan::DeviceId{0});
  if (config.color) {
    h.output_device("colors", holoscan::DeviceId{0});
  }
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), what + " is rejected");
  VERIFY_MSG(result.describe_failure().find(what) != std::string::npos,
             "failure names '" + what + "': " + result.describe_failure());
}

void check_rejected_inputs() {
  const SyntheticFrame f = synthetic_frame(kWidth, kHeight, 0, false);
  const DepthToPointCloudConfig config = base_config(false);
  // A color image where depth belongs.
  check_rejects(config, color_image(f, holoscan::schema::ImageEncoding_RGB8, false), std::nullopt,
                "unsupported depth encoding");
  // A depth image larger than the outputs are sized for.
  DepthToPointCloudConfig small = config;
  small.max_width = kWidth / 2;
  check_rejects(small, depth_image(f, false, false), std::nullopt, "exceeds the configured");
  // A color image of another size.
  DepthToPointCloudConfig with_color = config;
  with_color.color = true;
  const SyntheticFrame other = synthetic_frame(kWidth / 2, kHeight / 2, 0, false);
  check_rejects(with_color, depth_image(f, false, false),
                color_image(other, holoscan::schema::ImageEncoding_RGB8, false),
                "color image size must match");
}

void check_rejected_config() {
  VERIFY(testing::throws<std::invalid_argument>([] {
    DepthToPointCloudOp op(DepthToPointCloudConfig{});  // Neither intrinsics nor their input.
  }));
  VERIFY(testing::throws<std::invalid_argument>([] {
    DepthToPointCloudConfig config = base_config(false);
    config.intrinsics->fx = 0.0F;
    DepthToPointCloudOp op(config);
  }));
  VERIFY(testing::throws<std::invalid_argument>([] {
    DepthToPointCloudConfig config = base_config(false);
    config.depth_max = config.depth_min;
    DepthToPointCloudOp op(config);
  }));
  VERIFY(testing::throws<std::invalid_argument>([] {
    DepthToPointCloudConfig config = base_config(false);
    config.depth_scale = 0.0F;
    DepthToPointCloudOp op(config);
  }));
  VERIFY(testing::throws<std::invalid_argument>([] {
    // Configured intrinsics that per-frame ones would make unused.
    DepthToPointCloudConfig config = base_config(false);
    config.intrinsics_input = true;
    DepthToPointCloudOp op(config);
  }));
}

// The reporter counts a cloud whose frame id does not follow the previous one.
void check_report_frame_gaps(const std::vector<std::uint64_t>& frame_ids,
                             std::uint32_t expected_gaps) {
  auto report = std::make_shared<CloudReport>();
  auto h = holoscan::testing::create_operator_test<PointCloudReportOp>(
      PointCloudReportConfig{.report = report});
  const std::vector<float> xyz(static_cast<std::size_t>(4) * 2 * 3, 1.0F);
  std::vector<holoscan::testing::TestSample<holoscan::Tensor>> samples;
  for (std::size_t i = 0; i < frame_ids.size(); ++i) {
    holoscan::testing::TestSample<holoscan::Tensor> sample = holoscan::testing::sample_at(
        make({2, 4, 3}, {kDLFloat, 32, 1}, xyz.data(), xyz.size() * sizeof(float), false),
        capture_time(static_cast<std::uint32_t>(i)));
    sample.metadata.frame_id = frame_ids[i];
    samples.push_back(std::move(sample));
  }
  h.push_samples("point_cloud", std::move(samples));
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  VERIFY(report->frames == frame_ids.size());
  VERIFY_MSG(report->frame_gaps == expected_gaps, std::to_string(report->frame_gaps) +
                                                      " frame gaps, expected " +
                                                      std::to_string(expected_gaps));
  VERIFY(report->passed() == (expected_gaps == 0));
}

}  // namespace
}  // namespace depth_to_point_cloud

int main() {
  namespace dpc = depth_to_point_cloud;
  return dpc::testing::run({
      {"float32 depth from host memory", [] { dpc::check_depth_only(false, false); }},
      {"float32 depth from device memory", [] { dpc::check_depth_only(false, true); }},
      {"uint16 depth from host memory", [] { dpc::check_depth_only(true, false); }},
      {"uint16 depth from device memory", [] { dpc::check_depth_only(true, true); }},
      {"a finite invalid value", [] { dpc::check_depth_only(false, true, -1.0F); }},
      {"RGB8 color and per-frame intrinsics, host",
       [] { dpc::check_color_and_intrinsics(holoscan::schema::ImageEncoding_RGB8, false); }},
      {"BGR8 color and per-frame intrinsics, device",
       [] { dpc::check_color_and_intrinsics(holoscan::schema::ImageEncoding_BGR8, true); }},
      {"RGBA8 color and per-frame intrinsics, host",
       [] { dpc::check_color_and_intrinsics(holoscan::schema::ImageEncoding_RGBA8, false); }},
      {"BGRA8 color and per-frame intrinsics, device",
       [] { dpc::check_color_and_intrinsics(holoscan::schema::ImageEncoding_BGRA8, true); }},
      {"intrinsics with another capture time give no cloud", dpc::check_unmatched_intrinsics},
      {"host input is held until its copy is done", dpc::check_host_input_held},
      {"pinned depth is held by the operator", dpc::check_pinned_depth_held_by_operator},
      {"pinned intrinsics are waited for on the host", dpc::check_pinned_intrinsics_waited_for},
      {"misaligned device data is rejected", dpc::check_misaligned_device_data},
      {"unusable inputs are rejected", dpc::check_rejected_inputs},
      {"unusable configurations are rejected", dpc::check_rejected_config},
      {"the reporter accepts consecutive frame ids",
       [] { dpc::check_report_frame_gaps({1, 2, 3}, 0); }},
      {"the reporter counts a gap in frame ids",
       [] { dpc::check_report_frame_gaps({1, 2, 4, 5}, 1); }},
  });
}
