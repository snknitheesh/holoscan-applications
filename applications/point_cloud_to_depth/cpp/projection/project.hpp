// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The projection kernels: a point cloud, an optional rigid transform into the
// camera optical frame and pinhole intrinsics in, a z-buffered depth image
// (and optionally the nearest points' colors) out. Nothing here depends on the
// Holoscan SDK, so the kernels build and test on their own.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace point_cloud_to_depth {

// Pinhole intrinsics in pixels; pixel centers are at integer coordinates.
struct CameraIntrinsics {
  float fx{};
  float fy{};
  float cx{};
  float cy{};
};

// A rigid transform as a 3x4 row-major matrix [R | t]: p_out = R * p_in + t.
struct Extrinsics {
  float matrix[12]{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
};

// The inverse of a rigid transform: [R^T | -R^T t].
[[nodiscard]] Extrinsics rigid_inverse(const Extrinsics& transform);

// Element type and unit of the depth image.
enum class DepthFormat : std::uint8_t {
  kFloat32,  // Meters.
  kUint16,   // round(z / depth_scale), half away from zero, in 1..kMaxDepthCount.
};

// The largest uint16 depth. A point whose rounded depth is above it, or is 0,
// cannot be represented and is dropped, as 0 marks an empty pixel.
inline constexpr int kMaxDepthCount = 65535;

// The splat sizes the kernel draws: a square of 1, 3 or 5 pixels a side.
[[nodiscard]] constexpr bool valid_point_size(int point_size) {
  return point_size == 1 || point_size == 3 || point_size == 5;
}

struct ProjectParams {
  // The depth image size in pixels.
  int width{};
  int height{};
  CameraIntrinsics intrinsics;
  // When set, the kernel reads float32 [fx, fy, cx, cy] here, in device
  // memory, instead of using `intrinsics`; the host never sees them. A
  // non-positive or non-finite focal length or a non-finite center then
  // leaves the image empty.
  const float* device_intrinsics{};
  // Maps the input points into the camera optical frame (x right, y down,
  // z forward).
  Extrinsics extrinsics;
  // Points whose camera-frame z is not positive or lies outside
  // [depth_min, depth_max] meters are dropped.
  float depth_min{0.0F};
  float depth_max{100.0F};
  DepthFormat format{DepthFormat::kFloat32};
  // Meters per count of a uint16 depth image.
  float depth_scale{0.001F};
  // Side of the square of pixels each point covers, centered on its pixel.
  int point_size{1};
};

// Number of 64-bit z-buffer entries a width x height image needs.
[[nodiscard]] constexpr std::size_t zbuffer_entries(int width, int height) {
  return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

// Projects `point_count` points, packed float32 x, y, z triplets, into a
// depth image of params.width x params.height packed pixels: the nearest
// point wins each pixel, the lower index on equal depth, so the result does
// not depend on thread scheduling. A point covers the pixels around the one
// nearest its projection (lroundf of u and v), clipped at the image border;
// a point whose own pixel is outside the image is dropped. Empty pixels are
// 0 in `depth` and black in `rgb`.
//
// `depth` receives float32 or uint16 pixels as params.format says. When `rgb`
// is given, it receives the winning point's RGB8 color from `colors` (3 bytes
// per point). `zbuffer` holds zbuffer_entries() 64-bit scratch entries. All
// pointers are device memory, and all work is ordered on `stream`.
//
// Returns cudaErrorInvalidValue for an empty image, a point size other than
// 1, 3 or 5, a missing buffer, a `zbuffer` not aligned to 8 bytes, or `rgb`
// without `colors`.
cudaError_t launch_project(const float* points, const std::uint8_t* colors,
                           std::uint32_t point_count, const ProjectParams& params,
                           std::uint64_t* zbuffer, void* depth, std::uint8_t* rgb,
                           cudaStream_t stream);

}  // namespace point_cloud_to_depth
