// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic depth camera: a tilted plane with a dome on it, small holes of
// missing depth, and a patch too far away to be valid, moving from frame to
// frame. The same functions run on the GPU to render frames and on the host
// to check the point clouds made from them, so the check is exact.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "deproject.hpp"

#if defined(__CUDACC__)
#define DEPTH_TO_POINT_CLOUD_HOST_DEVICE __host__ __device__
#else
#define DEPTH_TO_POINT_CLOUD_HOST_DEVICE
#endif

namespace depth_to_point_cloud {

struct SyntheticFrame {
  int width{};
  int height{};
  std::uint32_t frame{};
  // The plane's per-frame offset in meters. Computed on the host only, so the
  // device and host evaluate bit-identical depths.
  float offset{};
  CameraIntrinsics intrinsics;
};

// Depth that the synthetic camera reports beyond every sensible range.
inline constexpr float kSyntheticFarDepth = 50.0F;

// The frame `frame` of a `width` x `height` synthetic camera. With
// `vary_intrinsics`, the focal length changes from frame to frame, to exercise
// per-frame intrinsics.
inline SyntheticFrame synthetic_frame(int width, int height, std::uint32_t frame,
                                      bool vary_intrinsics) {
  SyntheticFrame f;
  f.width = width;
  f.height = height;
  f.frame = frame;
  f.offset = 0.3F * std::sin(0.05F * static_cast<float>(frame));
  const float focal = 0.8F * static_cast<float>(width) *
                      (vary_intrinsics ? 1.0F + 0.01F * static_cast<float>(frame % 5U) : 1.0F);
  // Slightly unequal focal lengths, so a mix-up of the axes shows.
  f.intrinsics = {focal, 0.97F * focal, (width - 1) / 2.0F, (height - 1) / 2.0F};
  return f;
}

// Metric depth at pixel (u, v): 0 in a hole, kSyntheticFarDepth in the far
// patch, otherwise between about 0.5 and 2.2 meters.
DEPTH_TO_POINT_CLOUD_HOST_DEVICE inline float synthetic_depth(const SyntheticFrame& f, int u,
                                                              int v) {
  const int shift = static_cast<int>(f.frame % 16U);
  if (((u + shift) / 8 + (v / 8) * 3) % 37 == 0) {
    return 0.0F;  // A hole: no depth.
  }
  if (u < f.width / 10 && v < f.height / 10) {
    return kSyntheticFarDepth;
  }
  const float w = static_cast<float>(f.width);
  const float h = static_cast<float>(f.height);
  float z = 1.0F + 0.5F * static_cast<float>(u) / w + 0.4F * static_cast<float>(v) / h + f.offset;
  // A dome toward the camera, centered in the image.
  const float radius = 0.2F * w;
  const float du = static_cast<float>(u) - 0.5F * w;
  const float dv = static_cast<float>(v) - 0.5F * h;
  const float r2 = (du * du + dv * dv) / (radius * radius);
  if (r2 < 1.0F) {
    z -= 0.25F * sqrtf(1.0F - r2);
  }
  return z;
}

// The depth in millimeters, as a 16-bit camera reports it.
DEPTH_TO_POINT_CLOUD_HOST_DEVICE inline std::uint16_t synthetic_depth_mm(const SyntheticFrame& f,
                                                                         int u, int v) {
  return static_cast<std::uint16_t>(synthetic_depth(f, u, v) * 1000.0F + 0.5F);
}

// An RGB color per pixel: red across, green down, blue by frame.
DEPTH_TO_POINT_CLOUD_HOST_DEVICE inline void synthetic_color(const SyntheticFrame& f, int u, int v,
                                                             std::uint8_t rgb[3]) {
  rgb[0] = static_cast<std::uint8_t>(255 * u / f.width);
  rgb[1] = static_cast<std::uint8_t>(255 * v / f.height);
  rgb[2] = static_cast<std::uint8_t>((f.frame * 7U) & 0xFFU);
}

// Renders `f` into device buffers on `stream`: depth as float32 meters or,
// with `millimeters`, uint16 millimeters; color as RGB8 when `rgb` is given.
cudaError_t render_synthetic(const SyntheticFrame& f, bool millimeters, void* depth,
                             std::uint8_t* rgb, cudaStream_t stream);

}  // namespace depth_to_point_cloud
