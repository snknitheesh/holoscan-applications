// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DepthToPointCloudOp: deprojects an organized depth image into an organized
// point cloud on the GPU, one CUDA thread per pixel. Optionally it takes an
// aligned color image and emits each point's color, and takes each frame's
// intrinsics as an input instead of configured ones.
//
// With the optional inputs, the operator runs once per depth image, when the
// color image and intrinsics with exactly the same capture time have arrived.
// A camera that sends intrinsics at their own rate, or with other capture
// times, never completes a frame: the depth images are dropped and no cloud
// comes out. Send intrinsics with every depth image, or configure them.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "deproject/deproject.hpp"

namespace depth_to_point_cloud {

struct DepthToPointCloudConfig {
  // Used for every frame. Give these or enable `intrinsics_input`, not both.
  std::optional<CameraIntrinsics> intrinsics;
  // Multiplies raw depth into meters: 0.001 for uint16 millimeters, 1.0 for
  // float32 meters.
  float depth_scale{0.001F};
  // Valid metric depth range; other pixels are written as `invalid_value`.
  float depth_min{0.0F};
  float depth_max{100.0F};
  float invalid_value{std::numeric_limits<float>::quiet_NaN()};
  // Declare the `color` input and the `colors` output.
  bool color{false};
  // Declare the `intrinsics` input: float32 [fx, fy, cx, cy], one with the
  // capture time of every depth image.
  bool intrinsics_input{false};
  // Largest image the outputs are sized for.
  int max_width{1920};
  int max_height{1080};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const DepthToPointCloudConfig& config);

namespace detail {

// Keeps `tensor`'s memory from reuse until the work queued on `stream` so far
// is done: its allocator waits for the stream if it can, otherwise this
// waits. For host memory that the operator copies to the GPU asynchronously.
[[nodiscard]] holoscan::expected<void, holoscan::Error> hold_until_done(
    const holoscan::Tensor& tensor, cudaStream_t stream);

}  // namespace detail

class DepthToPointCloudOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit DepthToPointCloudOp(DepthToPointCloudConfig config);
  ~DepthToPointCloudOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // MONO16 (raw units, scaled by depth_scale) or MONO32F, [H, W] or [H, W, 1],
  // rows packed, in host or CUDA device memory.
  holoscan::Input<holoscan::schema::ImageT> depth;
  // With `color`: RGB8, RGBA8, BGR8 or BGRA8 of the depth image's size.
  holoscan::Input<holoscan::schema::ImageT> color;
  // With `intrinsics_input`: float32 [fx, fy, cx, cy], with the depth image's
  // capture time. In host memory they are checked, and unusable ones fail the
  // frame; in device memory the kernel reads them, and unusable ones make
  // every point invalid.
  holoscan::Input<holoscan::Tensor> intrinsics;
  // float32 [H, W, 3] XYZ in meters in the camera optical frame, on the GPU.
  holoscan::Output<holoscan::Tensor> point_cloud;
  // With `color`: uint8 [H, W, 3] RGB per point, on the GPU.
  holoscan::Output<holoscan::Tensor> colors;

 private:
  // A device copy of a host input, grown as needed.
  struct Staging {
    void* data{};
    std::size_t bytes{};
  };
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_device(
      const holoscan::Tensor& tensor, std::size_t bytes, Staging& staging, cudaStream_t stream);
  void release() noexcept;

  DepthToPointCloudConfig config_;
  Staging depth_staging_;
  Staging color_staging_;
  // Used when the runtime gives this operator no CUDA stream.
  cudaStream_t own_stream_{};
};

}  // namespace depth_to_point_cloud
