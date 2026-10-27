// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Yolo26DetectionOp on its own, through the SDK's operator test harness, with
// the stub model, so through TensorRT without downloaded weights: synthetic
// frames in every encoding, from GPU, host and pinned host memory, wide and
// tall, checked against their known detections; channels mislabeled, which
// swaps the classes; the score threshold and the capacity; an image size
// that changes; the capture time and frame id on every output; and images,
// settings and models it must reject.
//
// Usage: yolo26_operator_test DIR (a folder for the stub models and engines)

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "detection/letterbox.hpp"
#include "detection/matching.hpp"
#include "detection/stub_model.hpp"
#include "detection/synthetic_scene.hpp"
#include "operators/yolo26_detection_op.hpp"
#include "test_support.hpp"
#if YOLO26_DETECTION_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>

#include "operators/detection_overlay_op.hpp"
#endif

namespace yolo26_detection {
namespace {

using holoscan::schema::ImageEncoding;
using holoscan::testing::TensorSnapshot;

std::filesystem::path g_directory;

constexpr int kCapacity = 20;
const char* const kOutputs[] = {"boxes", "scores", "class_ids", "count"};

enum class Memory { kDevice, kHost, kPinned };

// A tensor of `dtype` holding `bytes`: in plain host memory, or copied into
// pinned host or GPU memory.
std::shared_ptr<holoscan::Tensor> image_tensor(std::vector<std::uint8_t> bytes,
                                               std::vector<std::int64_t> shape, Memory memory,
                                               DLDataType dtype = {kDLUInt, 8, 1}) {
  if (memory == Memory::kDevice) {
    // From the SDK's allocator, whose memory says when it is ready to read.
    auto tensor =
        holoscan::make_tensor(shape, dtype, std::nullopt, holoscan::MemoryStorageType::kDevice,
                              std::make_shared<holoscan::UnboundedAllocator>());
    if (!tensor) {
      throw std::runtime_error(tensor.error().message);
    }
    auto destination = tensor->mutable_data();
    if (!destination || cudaMemcpy(*destination, bytes.data(), bytes.size(),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("cannot copy the image to the GPU");
    }
    return std::make_shared<holoscan::Tensor>(std::move(*tensor));
  }
  struct Storage {
    DLManagedTensor managed{};
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
    void* pinned{};
    ~Storage() {
      if (pinned != nullptr) cudaFreeHost(pinned);
    }
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = std::move(shape);
  storage->bytes = std::move(bytes);
  void* data = storage->bytes.data();
  DLDeviceType type = kDLCPU;
  if (memory == Memory::kPinned) {
    if (cudaMallocHost(&storage->pinned, storage->bytes.size()) != cudaSuccess) {
      throw std::runtime_error("cudaMallocHost failed");
    }
    std::memcpy(storage->pinned, storage->bytes.data(), storage->bytes.size());
    data = storage->pinned;
    type = kDLCUDAHost;
  }
  storage->managed.dl_tensor = {.data = data,
                                .device = {type, 0},
                                .ndim = static_cast<std::int32_t>(storage->shape.size()),
                                .dtype = dtype,
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* tensor) {
    delete static_cast<Storage*>(tensor->manager_ctx);
  };
  return std::make_shared<holoscan::Tensor>(&storage.release()->managed);
}

ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
    default:
      return holoscan::schema::ImageEncoding_RGB8;
  }
}

// `scene` drawn in `format` and labeled `label` (by default, as drawn).
holoscan::schema::ImageT scene_image(const SyntheticScene& scene, PixelFormat format,
                                     Memory memory = Memory::kHost,
                                     std::optional<ImageEncoding> label = std::nullopt) {
  const int n = channels(format);
  const std::size_t pitch = static_cast<std::size_t>(scene.width) * n;
  std::vector<std::uint8_t> pixels(pitch * scene.height);
  render_scene(scene, format, pixels.data(), pitch);
  holoscan::schema::ImageT image;
  image.width = scene.width;
  image.height = scene.height;
  image.encoding = label.value_or(encoding_of(format));
  image.data = image_tensor(std::move(pixels), {scene.height, scene.width, n}, memory);
  return image;
}

Yolo26DetectionConfig stub_config(int input_size = 640) {
  return Yolo26DetectionConfig{.model = write_stub_model(g_directory, input_size),
                               .input_size = input_size,
                               .max_detections = kCapacity,
                               .num_classes = kStubClasses};
}

// The harness stages the images in GPU memory, `max_bytes` for the largest.
void stage_on_gpu(holoscan::testing::OperatorTestHarness<Yolo26DetectionOp>& h,
                  std::uint64_t max_bytes) {
  h.input_source_tensor_spec("image",
                             holoscan::TensorOutputSpec{
                                 .representation =
                                     holoscan::TensorRepresentation{
                                         .memory_kind = holoscan::MemoryKind::kCudaDevice,
                                         .dtype = DLDataType{kDLUInt, 8, 1},
                                         .rank = 3U,
                                         .strides = holoscan::TensorStridePolicy::kContiguousOnly,
                                     },
                                 .bounds = holoscan::tensor_bounds(max_bytes),
                                 .storage = holoscan::TensorOutputStorage::kRuntimePool,
                             });
  h.input_source_device("image", holoscan::DeviceId{0});
}

void declare_outputs(holoscan::testing::OperatorTestHarness<Yolo26DetectionOp>& h) {
  for (const char* name : kOutputs) {
    h.output_device(name, holoscan::DeviceId{0});
    h.capture_snapshot<holoscan::Tensor>(name);
  }
}

template <typename T>
std::vector<T> values(const TensorSnapshot& snapshot, const std::vector<std::int64_t>& shape) {
  VERIFY(snapshot.shape() == shape);
  std::vector<T> out(snapshot.bytes().size() / sizeof(T));
  VERIFY(snapshot.bytes().size() == out.size() * sizeof(T));
  std::memcpy(out.data(), snapshot.bytes().data(), out.size() * sizeof(T));
  return out;
}

struct Frame {
  std::vector<BoxDetection> detections;
  holoscan::SampleMetadata metadata;
};

// Each frame's detections from the four outputs, after checking their shapes,
// that slots past the count are zero, and that all four carry one capture
// time and frame id.
std::vector<Frame> frames_of(const holoscan::testing::TestResult& result, int capacity) {
  const auto boxes = result.snapshots<holoscan::Tensor>("boxes");
  const auto scores = result.snapshots<holoscan::Tensor>("scores");
  const auto class_ids = result.snapshots<holoscan::Tensor>("class_ids");
  const auto counts = result.snapshots<holoscan::Tensor>("count");
  VERIFY(boxes.size() == counts.size() && scores.size() == counts.size() &&
         class_ids.size() == counts.size());
  std::vector<Frame> frames;
  for (std::size_t f = 0;
       f < counts.size() && f < boxes.size() && f < scores.size() && f < class_ids.size(); ++f) {
    const auto b = values<float>(boxes[f], {capacity, 4});
    const auto s = values<float>(scores[f], {capacity});
    const auto c = values<std::int32_t>(class_ids[f], {capacity});
    const auto n = values<std::int32_t>(counts[f], {1});
    VERIFY(boxes[f].dtype().code == kDLFloat && class_ids[f].dtype().code == kDLInt);
    Frame frame;
    frame.metadata = counts[f].sample_metadata();
    for (const TensorSnapshot* other : {&boxes[f], &scores[f], &class_ids[f]}) {
      VERIFY(other->sample_metadata().capture_timestamp_ns == frame.metadata.capture_timestamp_ns &&
             other->sample_metadata().frame_id == frame.metadata.frame_id);
    }
    if (n.empty() || b.size() != static_cast<std::size_t>(capacity) * 4U || s.size() != c.size()) {
      continue;
    }
    VERIFY(n[0] >= 0 && n[0] <= capacity);
    for (int i = 0; i < capacity; ++i) {
      const auto k = static_cast<std::size_t>(i);
      if (i < n[0]) {
        frame.detections.push_back(
            BoxDetection{b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3], s[k], c[k]});
      } else {
        VERIFY_MSG(b[4 * k] == 0 && b[4 * k + 1] == 0 && b[4 * k + 2] == 0 && b[4 * k + 3] == 0 &&
                       s[k] == 0 && c[k] == 0,
                   "slot " + std::to_string(i) + " past the count is not zero");
      }
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

void check_truth(const Frame& frame, const SyntheticScene& scene, int input_size) {
  const float scale = make_letterbox(scene.width, scene.height, input_size).scale;
  const TruthMatch match =
      match_truth(frame.detections, scene_truth(scene), 2.0F / scale + 1.0F, 1e-6F);
  VERIFY_MSG(match.missed == 0 && match.extra == 0,
             std::to_string(scene.width) + "x" + std::to_string(scene.height) + ": missed " +
                 std::to_string(match.missed) + ", extra " + std::to_string(match.extra) +
                 ", edge error " + std::to_string(match.max_edge_error));
}

std::chrono::nanoseconds capture_time(std::uint32_t frame) {
  return std::chrono::milliseconds(100 * (frame + 1));
}

// Frames 0..2 of a width x height scene in `format` from `memory`.
void check_detections(PixelFormat format, Memory memory, int width, int height, int input_size) {
  auto h = holoscan::testing::create_operator_test<Yolo26DetectionOp>(stub_config(input_size));
  for (std::uint32_t f = 0; f < 3; ++f) {
    holoscan::testing::TestSample<holoscan::schema::ImageT> sample{
        .data = scene_image(make_scene(width, height, f * 7U), format, memory),
        .capture_coordinate = capture_time(f)};
    sample.metadata.frame_id = 1U + f;
    h.push_samples<holoscan::schema::ImageT>("image", {std::move(sample)});
  }
  if (memory == Memory::kDevice) {
    stage_on_gpu(h, static_cast<std::uint64_t>(width) * height * channels(format));
  }
  declare_outputs(h);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto frames = frames_of(result, kCapacity);
  VERIFY(frames.size() == 3);
  for (std::uint32_t f = 0; f < frames.size(); ++f) {
    check_truth(frames[f], make_scene(width, height, f * 7U), input_size);
    VERIFY(frames[f].metadata.capture_timestamp_ns == capture_time(f).count());
    VERIFY(frames[f].metadata.frame_id == 1U + f);
  }
}

// BGR pixels labeled RGB8 put the blue rectangle in the red plane: the stub
// then finds class 0 where the blue one is, and class 1 where the red one is.
void check_mislabeled_channels() {
  auto h = holoscan::testing::create_operator_test<Yolo26DetectionOp>(stub_config());
  const SyntheticScene scene = make_scene(1280, 720, 3);
  h.push("image", scene_image(scene, PixelFormat::kBgr8, Memory::kHost,
                              holoscan::schema::ImageEncoding_RGB8));
  declare_outputs(h);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto frames = frames_of(result, kCapacity);
  VERIFY(frames.size() == 1);
  if (frames.size() != 1) return;
  std::vector<BoxDetection> swapped = scene_truth(scene);
  std::swap(swapped[0].class_id, swapped[1].class_id);
  std::swap(swapped[0].score, swapped[1].score);
  const TruthMatch match = match_truth(frames[0].detections, swapped, 5.0F, 1e-6F);
  VERIFY_MSG(match.missed == 0 && match.extra == 0, "the classes did not swap");
}

// Only the detections at or above the threshold, at most the capacity, best
// first.
void check_threshold_and_capacity() {
  const SyntheticScene scene = make_scene(1280, 720, 0);
  struct Case {
    float threshold;
    int capacity;
    std::vector<std::int32_t> classes;  // Of the first detections.
    int count;
  };
  for (const Case& c : {Case{0.85F, 5, {0}, 1}, Case{0.95F, 5, {}, 0}, Case{0.25F, 1, {0}, 1},
                        Case{0.0F, kStubRows, {0, 1}, kStubRows}}) {
    Yolo26DetectionConfig config = stub_config();
    config.score_threshold = c.threshold;
    config.max_detections = c.capacity;
    auto h = holoscan::testing::create_operator_test<Yolo26DetectionOp>(config);
    h.push("image", scene_image(scene, PixelFormat::kRgb8));
    declare_outputs(h);
    const auto result = h.run();
    VERIFY_MSG(result.ok(), result.describe_failure());
    if (!result.ok()) continue;
    const auto frames = frames_of(result, c.capacity);
    VERIFY(frames.size() == 1);
    if (frames.size() != 1) continue;
    const auto& d = frames[0].detections;
    VERIFY_MSG(static_cast<int>(d.size()) == c.count, "threshold " + std::to_string(c.threshold) +
                                                          ": " + std::to_string(d.size()) +
                                                          " detections");
    for (std::size_t i = 0; i < c.classes.size() && i < d.size(); ++i) {
      VERIFY(d[i].class_id == c.classes[i] && d[i].score == kStubScores[c.classes[i]]);
    }
    for (std::size_t i = 1; i < d.size(); ++i) {
      VERIFY(d[i].score <= d[i - 1].score);
      VERIFY(d[i].score >= c.threshold);
    }
  }
}

// The letterbox follows each image's size.
void check_size_change() {
  auto h = holoscan::testing::create_operator_test<Yolo26DetectionOp>(stub_config());
  const std::vector<SyntheticScene> scenes{make_scene(1280, 720, 1), make_scene(333, 500, 2),
                                           make_scene(1280, 720, 3), make_scene(64, 48, 4)};
  for (const SyntheticScene& scene : scenes) {
    h.push("image", scene_image(scene, PixelFormat::kRgba8, Memory::kDevice));
  }
  stage_on_gpu(h, 1280ULL * 720ULL * 4ULL);
  declare_outputs(h);
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return;
  const auto frames = frames_of(result, kCapacity);
  VERIFY(frames.size() == scenes.size());
  for (std::size_t i = 0; i < frames.size() && i < scenes.size(); ++i) {
    check_truth(frames[i], scenes[i], 640);
  }
}

// A run with `image` fails, saying `message`.
void check_run_fails(Yolo26DetectionConfig config, holoscan::schema::ImageT image,
                     const std::string& message) {
  auto h = holoscan::testing::create_operator_test<Yolo26DetectionOp>(std::move(config));
  h.push("image", std::move(image));
  declare_outputs(h);
  const auto result = h.run();
  VERIFY_MSG(!result.ok(), "expected a failure: " + message);
  VERIFY_MSG(result.describe_failure().find(message) != std::string::npos,
             "expected '" + message + "' in: " + result.describe_failure());
}

void check_rejects_images() {
  const SyntheticScene scene = make_scene(320, 240, 0);
  holoscan::schema::ImageT mono = scene_image(scene, PixelFormat::kRgb8);
  mono.encoding = holoscan::schema::ImageEncoding_MONO8;
  check_run_fails(stub_config(), mono, "unsupported image encoding MONO8");
  // RGBA pixels labeled RGB8: four channels where three belong.
  check_run_fails(
      stub_config(),
      scene_image(scene, PixelFormat::kRgba8, Memory::kHost, holoscan::schema::ImageEncoding_RGB8),
      "uint8 [height, width, channels]");
}

void check_rejects_models() {
  const SyntheticScene scene = make_scene(320, 240, 0);
  // A model exported for another input size.
  Yolo26DetectionConfig other_size = stub_config(640);
  other_size.model = write_stub_model(g_directory, 320);
  check_run_fails(other_size, scene_image(scene, PixelFormat::kRgb8),
                  "export it for this input size");
  Yolo26DetectionConfig missing = stub_config();
  missing.model = g_directory / "no_such_model.onnx";
  check_run_fails(missing, scene_image(scene, PixelFormat::kRgb8), "cannot load");
}

void check_rejects_settings() {
  const auto rejects = [](auto edit) {
    Yolo26DetectionConfig config{.model = "model.onnx"};
    edit(config);
    return testing::throws<std::invalid_argument>([&] { Yolo26DetectionOp op(config); });
  };
  VERIFY(rejects([](auto& c) { c.model.clear(); }));
  VERIFY(rejects([](auto& c) { c.input_size = 100; }));
  VERIFY(rejects([](auto& c) { c.input_size = 0; }));
  VERIFY(rejects([](auto& c) { c.input_size = 4128; }));
  VERIFY(rejects([](auto& c) { c.score_threshold = -0.1F; }));
  VERIFY(rejects([](auto& c) { c.score_threshold = 1.5F; }));
  VERIFY(rejects([](auto& c) { c.score_threshold = std::numeric_limits<float>::quiet_NaN(); }));
  VERIFY(rejects([](auto& c) { c.max_detections = 0; }));
  VERIFY(rejects([](auto& c) { c.num_classes = 0; }));
  VERIFY(rejects([](auto& c) { c.build_workspace_bytes = 1024; }));
  VERIFY(!rejects([](auto&) {}));
}

#if YOLO26_DETECTION_ENABLE_VISUALIZATION
// `values` as a host tensor of `shape` and `dtype`.
template <typename T>
holoscan::Tensor host_tensor(const std::vector<T>& values, std::vector<std::int64_t> shape,
                             DLDataType dtype) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return *image_tensor(std::move(bytes), std::move(shape), Memory::kHost, dtype);
}

constexpr std::chrono::nanoseconds kOverlayCaptured = std::chrono::milliseconds(1234);
namespace viz = holoscan::visualization;
constexpr std::size_t kOverlayRowBytes = viz::kTextRunHeaderBytes + kOverlayLabelBytes;

struct OverlayOutput {
  holoscan::testing::SchemaSnapshot<holoscan::schema::ImageT> overlay;
  TensorSnapshot labels;
};

// The first `n` of three detections, in a 200x100 image, through the
// overlay; its two layers, or nothing if it failed.
std::optional<OverlayOutput> run_overlay(const std::vector<float>& boxes,
                                         const std::vector<float>& scores,
                                         const std::vector<std::int32_t>& class_ids,
                                         std::int32_t n) {
  auto h = holoscan::testing::create_operator_test<DetectionOverlayOp>(DetectionOverlayConfig{
      .width = 200, .height = 100, .max_detections = 3, .labels = {"person"}});
  namespace ht = holoscan::testing;
  h.push_frame(ht::frame_at(
      kOverlayCaptured, ht::input("boxes", host_tensor(boxes, {3, 4}, {kDLFloat, 32, 1})),
      ht::input("scores", host_tensor(scores, {3}, {kDLFloat, 32, 1})),
      ht::input("class_ids", host_tensor(class_ids, {3}, {kDLInt, 32, 1})),
      ht::input("count", host_tensor(std::vector<std::int32_t>{n}, {1}, {kDLInt, 32, 1}))));
  h.capture_snapshot<holoscan::schema::ImageT>("overlay");
  h.capture_snapshot<holoscan::Tensor>("labels");
  const auto result = h.run();
  VERIFY_MSG(result.ok(), result.describe_failure());
  if (!result.ok()) return std::nullopt;
  return OverlayOutput{result.snapshot<holoscan::schema::ImageT>("overlay"),
                       result.snapshot<holoscan::Tensor>("labels")};
}

// The RGBA pixel at (x, y) of a 200x100 overlay.
std::array<std::uint8_t, 4> pixel(const TensorSnapshot& image, int x, int y) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(image.bytes().data()) + (y * 200 + x) * 4;
  return {p[0], p[1], p[2], p[3]};
}

// Row `row` of the labels: its header and its text.
std::pair<viz::TextRunHeader, std::string> label_row(const TensorSnapshot& labels,
                                                     std::size_t row) {
  const auto* p = reinterpret_cast<const char*>(labels.bytes().data()) + row * kOverlayRowBytes;
  viz::TextRunHeader header;
  std::memcpy(&header, p, sizeof(header));
  std::string text(p + viz::kTextRunHeaderBytes, kOverlayLabelBytes);
  text.erase(std::min(text.find('\0'), text.size()));
  return {header, text};
}

// Two detections: a box per detection, 2 px wide in its class's color, and a
// shadowed label above each, named when the class has a name.
void check_overlay() {
  const auto out =
      run_overlay({20, 30, 80, 60, 100, 15, 150, 100, 0, 0, 0, 0}, {0.87F, 0.5F, 0}, {1, 0, 0}, 2);
  if (!out) return;
  for (const auto* metadata : {&out->overlay.sample_metadata(), &out->labels.sample_metadata()}) {
    VERIFY_MSG(metadata->capture_timestamp_ns == kOverlayCaptured.count(),
               "overlay layers keep the detections' capture time, not " +
                   std::to_string(metadata->capture_timestamp_ns) + " ns");
  }
  VERIFY(out->overlay.metadata().encoding == holoscan::schema::ImageEncoding_RGBA8);
  const auto& image = out->overlay.tensor(holoscan::schema::ImageTensorFields::data);
  VERIFY(image.shape() == (std::vector<std::int64_t>{100, 200, 4}));
  using Rgba = std::array<std::uint8_t, 4>;
  constexpr Rgba kClass0{230, 25, 75, 255};
  constexpr Rgba kClass1{60, 180, 75, 255};
  constexpr Rgba kClear{0, 0, 0, 0};
  // The first box's edges, 2 px wide inside it, and its clear inside.
  VERIFY(pixel(image, 20, 45) == kClass1 && pixel(image, 21, 45) == kClass1);
  VERIFY(pixel(image, 22, 45) == kClear && pixel(image, 19, 45) == kClear);
  VERIFY(pixel(image, 79, 45) == kClass1 && pixel(image, 78, 45) == kClass1);
  VERIFY(pixel(image, 50, 30) == kClass1 && pixel(image, 50, 59) == kClass1);
  VERIFY(pixel(image, 50, 45) == kClear && pixel(image, 50, 29) == kClear);
  // The second box, in another color, its bottom edge on the image's last rows.
  VERIFY(pixel(image, 100, 50) == kClass0 && pixel(image, 125, 99) == kClass0);
  VERIFY(pixel(image, 10, 10) == kClear);

  VERIFY(out->labels.shape() ==
         (std::vector<std::int64_t>{6, static_cast<std::int64_t>(kOverlayRowBytes)}));
  const auto [shadow, shadow_text] = label_row(out->labels, 0);
  const auto [first, first_text] = label_row(out->labels, 1);
  const auto [second_shadow, second_shadow_text] = label_row(out->labels, 2);
  const auto [second, second_text] = label_row(out->labels, 3);
  VERIFY(first_text == "class 1 0.87" && shadow_text == first_text);
  VERIFY(second_text == "person 0.50" && second_shadow_text == second_text);
  // Above the box, in its color; the shadow black and a little offset.
  VERIFY(first.color == 0xFF4BB43CU && shadow.color == 0xFF000000U);
  VERIFY(std::abs(first.x - 22.0F / 200) < 1e-6F && std::abs(first.y - 10.0F / 100) < 1e-6F);
  VERIFY(shadow.x > first.x && shadow.y > first.y);
  // Without room above the box, just inside its top.
  VERIFY(std::abs(second.y - 17.0F / 100) < 1e-6F && second.color == 0xFF4B19E6U);
  for (std::size_t row = 4; row < 6; ++row) {
    VERIFY(label_row(out->labels, row).second.empty());
  }
}

// No detections: a fully transparent overlay and no labels, so the previous
// frame's boxes vanish.
void check_overlay_empty() {
  const auto out =
      run_overlay({20, 30, 80, 60, 0, 0, 0, 0, 0, 0, 0, 0}, {0.87F, 0, 0}, {1, 0, 0}, 0);
  if (!out) return;
  const auto image = out->overlay.tensor(holoscan::schema::ImageTensorFields::data).bytes();
  VERIFY(image.size() == 200U * 100U * 4U);
  VERIFY(std::all_of(image.begin(), image.end(), [](std::byte b) { return b == std::byte{0}; }));
  const auto labels = out->labels.bytes();
  VERIFY(std::all_of(labels.begin(), labels.end(), [](std::byte b) { return b == std::byte{0}; }));
}
#endif

}  // namespace
}  // namespace yolo26_detection

int main(int argc, char** argv) {
  namespace yd = yolo26_detection;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  yd::g_directory = argv[1];
  std::filesystem::create_directories(yd::g_directory);
  using yd::Memory;
  using yd::PixelFormat;
  std::vector<yd::testing::TestCase> cases{
      {"RGB8 from the GPU, wide",
       [] { yd::check_detections(PixelFormat::kRgb8, Memory::kDevice, 1280, 720, 640); }},
      {"BGR8 from the GPU, tall",
       [] { yd::check_detections(PixelFormat::kBgr8, Memory::kDevice, 333, 1000, 640); }},
      {"RGBA8 from host memory, odd padding",
       [] { yd::check_detections(PixelFormat::kRgba8, Memory::kHost, 1000, 333, 640); }},
      {"BGRA8 from pinned memory, input 320",
       [] { yd::check_detections(PixelFormat::kBgra8, Memory::kPinned, 1920, 1080, 320); }},
      {"small image, upscaled",
       [] { yd::check_detections(PixelFormat::kRgb8, Memory::kHost, 100, 64, 640); }},
      {"mislabeled channels swap the classes", yd::check_mislabeled_channels},
      {"threshold and capacity", yd::check_threshold_and_capacity},
      {"image size changes", yd::check_size_change},
      {"rejects unusable images", yd::check_rejects_images},
      {"rejects unusable models", yd::check_rejects_models},
      {"rejects unusable settings", yd::check_rejects_settings},
  };
#if YOLO26_DETECTION_ENABLE_VISUALIZATION
  cases.push_back({"overlay boxes and labels", yd::check_overlay});
  cases.push_back({"overlay clears without detections", yd::check_overlay_empty});
#endif
  return yd::testing::run(cases);
}
