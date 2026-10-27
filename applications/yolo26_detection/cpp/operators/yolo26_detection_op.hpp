// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Yolo26DetectionOp: YOLO26 object detection on the GPU. An image goes in;
// the detections in it come out, as boxes in the image's pixels with a score
// and a class each, and their count. Inside, a CUDA kernel letterboxes the
// image into the model's input, the SDK's TensorRT inference library
// (HoloInfer) runs the end-to-end model, and a second kernel keeps the rows
// that pass the score threshold and maps them back to the image. Everything
// stays on the GPU, and the outputs are GPU tensors.
//
// The model is an ONNX export of an end-to-end (NMS-free) YOLO26 detector:
// input float32 [1, 3, S, S] RGB in [0, 1], output float32 [1, N, 6] of
// [x1, y1, x2, y2, score, class] in input pixels, best first. Given an .onnx
// file, HoloInfer builds a TensorRT engine for this GPU at start() and caches
// it beside the file, so later runs start in seconds; any other file is taken
// as such an engine.
//
// Each image's four outputs carry its capture time and frame id. They go out
// one after another: should one fail to go out, those before it have been
// delivered alone and compute() returns the error. Pair them by capture time
// (OnAligned), as the application's report does.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "detection/decode.hpp"
#include "detection/letterbox.hpp"

namespace yolo26_detection {

struct Yolo26DetectionConfig {
  // An .onnx model, whose engine is built and cached beside it, or a
  // serialized TensorRT engine built for this GPU.
  std::filesystem::path model;
  // The model's square input side S, a multiple of 32 in
  // kMinInputSize..kMaxInputSize. The model must have been exported for it.
  int input_size{640};
  // Detections need at least this score, in [0, 1].
  float score_threshold{0.25F};
  // Capacity of the outputs, 1..kMaxRows; further detections are dropped,
  // the lowest scores first, and a warning says so once per run.
  int max_detections{300};
  // The model's class count; rows with another class value are dropped.
  int num_classes{80};
  // The letterbox's border, as an 8-bit gray level.
  std::uint8_t pad_value{114};
  // Build the engine with FP16 kernels allowed (ignored for an engine file).
  bool fp16{false};
  // Let HoloInfer replay the inference as a CUDA graph.
  bool cuda_graphs{true};
  // TensorRT's workspace limit while it builds the engine.
  std::size_t build_workspace_bytes{std::size_t{1} << 30U};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const Yolo26DetectionConfig& config);

// Builds and caches the TensorRT engine of an .onnx `config.model` unless one
// is cached already. Call it before running the graph: some TensorRT releases
// (10.13 among them) change the SIGINT and SIGTERM handlers while building,
// which the SDK's managed signal handling accepts only before a run starts.
// start() then loads the cached engine.
// @throws std::runtime_error if the model cannot be built.
void prepare_engine(const Yolo26DetectionConfig& config);

class Yolo26DetectionOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit Yolo26DetectionOp(Yolo26DetectionConfig config);
  ~Yolo26DetectionOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  // Loads the model, building its engine first if it is not cached.
  // @throws std::runtime_error if the model cannot be loaded or does not have
  // the expected input and output.
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // RGB8, BGR8, RGBA8 or BGRA8: uint8 [height, width, channels], pixels
  // packed within each row and rows any pitch apart, in GPU memory on device
  // 0 or in host memory. Sides 1..kMaxImageSide.
  holoscan::Input<holoscan::schema::ImageT> image;
  // float32 [max_detections, 4]: x1, y1, x2, y2 in image pixels, within
  // [0, width] x [0, height], best first; zero beyond `count`.
  holoscan::Output<holoscan::Tensor> boxes;
  // float32 [max_detections]: each detection's score.
  holoscan::Output<holoscan::Tensor> scores;
  // int32 [max_detections]: each detection's class.
  holoscan::Output<holoscan::Tensor> class_ids;
  // int32 [1]: the number of detections.
  holoscan::Output<holoscan::Tensor> count;

 private:
  struct Engine;

  [[nodiscard]] holoscan::expected<const std::uint8_t*, holoscan::Error> stage(
      const holoscan::Tensor& tensor, int width, int height, int channels, std::size_t pitch,
      cudaStream_t stream);
  void release() noexcept;
  // Warns, once per run, if a finished frame had more detections than the
  // outputs hold.
  void warn_if_dropped();

  Yolo26DetectionConfig config_;
  std::unique_ptr<Engine> engine_;
  // A GPU copy of host images, grown as needed.
  void* staging_{};
  std::size_t staging_bytes_{};
  // Recorded after each frame's work. The model's buffers and `staging_` are
  // reused by every frame, but the runtime may give each activation another
  // stream, so the next frame's stream waits for it before touching them.
  cudaEvent_t frame_done_{};
  // Mapped host memory where the decode writes how many rows passed, before
  // the capacity cut; read once frame_done_ has completed.
  std::int32_t* passed_{};
  bool warned_dropped_{};
};

}  // namespace yolo26_detection
