// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudToDepthOp: projects a point cloud into a pinhole camera on the
// GPU, producing a depth image in which the nearest point wins each pixel,
// and optionally the color image of those points. It takes each point's
// color, and each frame's intrinsics, as optional inputs.
//
// With the optional inputs, the operator runs once per cloud, when the colors
// and intrinsics with exactly the same capture time have arrived. A sensor
// that sends intrinsics at their own rate, or with other capture times, never
// completes a frame: the clouds are dropped and no image comes out. Send
// intrinsics with every cloud, or configure them.
//
// Each frame's depth image is emitted before its color image. If the color
// emit fails, as into a full queue of a blocking connection, the depth image
// has already gone out alone and compute() returns the error. A consumer that
// needs both pairs them by capture time (OnAligned); connect one that only
// displays them with QueuePolicy::kDropOldest, so that it never pushes back.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "projection/project.hpp"

namespace point_cloud_to_depth {

// Largest image side and point count the operator accepts.
inline constexpr int kMaxImageSide = 16384;
inline constexpr std::uint32_t kMaxPointsLimit = std::uint32_t{1} << 30;

struct PointCloudToDepthConfig {
  // The depth image size in pixels, 1..kMaxImageSide.
  int width{640};
  int height{480};
  // Used for every frame. Give these or enable `intrinsics_input`, not both.
  std::optional<CameraIntrinsics> intrinsics;
  // Maps the input points into the camera optical frame (x right, y down,
  // z forward), as a 3x4 row-major [R | t]; none means they are in it
  // already.
  std::optional<Extrinsics> extrinsics;
  // Points whose camera-frame z is not positive or lies outside
  // [depth_min, depth_max] meters are dropped.
  float depth_min{0.0F};
  float depth_max{100.0F};
  // float32 meters (MONO32F), or uint16 counts of `depth_scale` meters
  // (MONO16) rounded half away from zero, in which a point whose count would
  // be 0 or above 65535 is dropped.
  DepthFormat depth_format{DepthFormat::kFloat32};
  float depth_scale{0.001F};
  // Side of the square of pixels each point covers: 1, 3 or 5.
  int point_size{1};
  // Declare the `colors` input and the `color` output.
  bool color{false};
  // The camera's coordinate frame, written with the cloud's capture time into
  // each image's header.
  std::string frame_id;
  // Declare the `intrinsics` input: float32 [fx, fy, cx, cy], one with the
  // capture time of every cloud.
  bool intrinsics_input{false};
  // Largest cloud accepted, 1..kMaxPointsLimit points; a larger one fails the
  // frame. It bounds the work per frame and the staging of host input.
  std::uint32_t max_points{std::uint32_t{1} << 24};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const PointCloudToDepthConfig& config);

namespace detail {

// Keeps `tensor`'s memory from reuse until the work queued on `stream` so far
// is done: its allocator waits for the stream if it can, otherwise this
// waits. For host memory that the operator copies to the GPU asynchronously.
[[nodiscard]] holoscan::expected<void, holoscan::Error> hold_until_done(
    const holoscan::Tensor& tensor, cudaStream_t stream);

}  // namespace detail

class PointCloudToDepthOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit PointCloudToDepthOp(PointCloudToDepthConfig config);
  ~PointCloudToDepthOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // float32 x, y, z: [N, 3], or an organized [H, W, 3] taken as H * W points;
  // packed, in host, pinned or CUDA device memory on the operator's GPU.
  // Non-finite points are skipped.
  holoscan::Input<holoscan::Tensor> points;
  // With `color`: uint8 RGB per point, the points' shape, packed.
  holoscan::Input<holoscan::Tensor> colors;
  // With `intrinsics_input`: float32 [fx, fy, cx, cy], with the cloud's
  // capture time. In host memory they are checked, and unusable ones fail the
  // frame; in device memory the kernel reads them, and unusable ones leave
  // the image empty.
  holoscan::Input<holoscan::Tensor> intrinsics;
  // [height, width] MONO32F meters or MONO16 counts, 0 where no point
  // projects, on the GPU. The descriptor's header holds the cloud's capture
  // time (0 without one) and `frame_id`; the sample carries the capture time
  // and frame id of the cloud.
  holoscan::Output<holoscan::schema::ImageT> depth;
  // With `color`: [height, width, 3] RGB8 of each pixel's nearest point,
  // black where none projects, on the GPU.
  holoscan::Output<holoscan::schema::ImageT> color;

 private:
  // A device buffer, grown as needed.
  struct DeviceBuffer {
    void* data{};
    std::size_t bytes{};
  };
  [[nodiscard]] holoscan::expected<void*, holoscan::Error> reserve(DeviceBuffer& buffer,
                                                                   std::size_t bytes,
                                                                   cudaStream_t stream);
  [[nodiscard]] holoscan::expected<const void*, holoscan::Error> on_device(
      const holoscan::Tensor& tensor, std::size_t bytes, DeviceBuffer& staging,
      cudaStream_t stream);
  [[nodiscard]] holoscan::expected<cudaStream_t, holoscan::Error> stream_for(
      holoscan::ExecutionContext& context);
  void release() noexcept;

  PointCloudToDepthConfig config_;
  ProjectParams params_;
  DeviceBuffer zbuffer_;
  DeviceBuffer points_staging_;
  DeviceBuffer colors_staging_;
};

}  // namespace point_cloud_to_depth
