// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The deprojection kernel: an organized depth image and pinhole intrinsics in,
// an organized XYZ point cloud (and optionally its colors) out. Nothing here
// depends on the Holoscan SDK, so the kernel builds and tests on its own.

#pragma once

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstdint>

namespace depth_to_point_cloud {

// Element type of the depth image.
enum class DepthType : std::uint8_t { kUint16, kFloat32 };

// Pinhole intrinsics in pixels; pixel centers are at integer coordinates.
struct CameraIntrinsics {
  float fx{};
  float fy{};
  float cx{};
  float cy{};
};

struct DeprojectParams {
  DepthType depth_type{DepthType::kFloat32};
  // Multiplies raw depth into meters, e.g. 0.001 for millimeters in uint16.
  float depth_scale{1.0F};
  CameraIntrinsics intrinsics;
  // When set, the kernel reads float32 [fx, fy, cx, cy] here, in device
  // memory, instead of using `intrinsics`; the host never sees them. A
  // non-positive or non-finite focal length or a non-finite center then makes
  // every point invalid.
  const float* device_intrinsics{};
  // Metric depths outside [depth_min, depth_max], zero, negative or non-finite
  // depths give `invalid_value` for X, Y and Z.
  float depth_min{0.0F};
  float depth_max{100.0F};
  float invalid_value{};
};

// An optional color image aligned with the depth image: 3 or 4 uint8 channels
// per pixel, rows packed. With `bgr`, channels 0 and 2 are swapped on output.
struct ColorInput {
  const void* data{};
  int channels{};
  bool bgr{};
};

// Deprojects `width` x `height` packed depth pixels into `out_xyz` and, when
// both `color.data` and `out_rgb` are given, copies the colors as RGB into
// `out_rgb`. All pointers are device memory; work is ordered on `stream`.
// Returns cudaErrorInvalidValue for an unsupported color layout.
cudaError_t launch_deproject(const void* depth, const DeprojectParams& params, ColorInput color,
                             float3* out_xyz, uchar3* out_rgb, int width, int height,
                             cudaStream_t stream);

}  // namespace depth_to_point_cloud
