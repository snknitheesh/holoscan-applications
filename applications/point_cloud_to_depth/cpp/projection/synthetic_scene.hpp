// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic depth camera and the point cloud made from it. The camera sees
// a tilted plane with a dome on it, small holes of missing depth, and a patch
// too far away to be valid, moving from frame to frame. Its depth image is
// deprojected into points, moved into a body frame and shuffled, with points
// that must not reach the depth image mixed in: duplicates farther along their
// pixel's ray, points behind the camera and outside its field of view, and
// non-finite points where the camera has no depth. The same functions run on
// the GPU to make clouds and on the host to make them for the tests and to
// know what the depth image should show.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "project.hpp"

#if defined(__CUDACC__)
#define POINT_CLOUD_TO_DEPTH_HOST_DEVICE __host__ __device__
#else
#define POINT_CLOUD_TO_DEPTH_HOST_DEVICE
#endif

namespace point_cloud_to_depth {

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
POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline float synthetic_depth(const SyntheticFrame& f, int u,
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

// The depth in whole millimeters, as a 16-bit camera reports it.
POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline std::uint16_t synthetic_depth_mm(const SyntheticFrame& f,
                                                                         int u, int v) {
  return static_cast<std::uint16_t>(synthetic_depth(f, u, v) * 1000.0F + 0.5F);
}

// The scene's depth in meters at pixel (u, v). With `millimeter_depth` it is a
// whole number of millimeters, so that a uint16 depth image in millimeters
// reproduces it exactly; otherwise it is synthetic_depth().
POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline float synthetic_scene_depth(const SyntheticFrame& f,
                                                                    bool millimeter_depth, int u,
                                                                    int v) {
  return millimeter_depth ? static_cast<float>(synthetic_depth_mm(f, u, v)) * 0.001F
                          : synthetic_depth(f, u, v);
}

// An RGB color per pixel: red across, green down, blue by frame.
POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline void synthetic_color(const SyntheticFrame& f, int u, int v,
                                                             std::uint8_t rgb[3]) {
  rgb[0] = static_cast<std::uint8_t>(255 * u / f.width);
  rgb[1] = static_cast<std::uint8_t>(255 * v / f.height);
  rgb[2] = static_cast<std::uint8_t>((f.frame * 7U) & 0xFFU);
}

// Where the synthetic camera sits on its body: body x forward is camera z,
// body y left is camera -x, body z up is camera -y, and the camera is 12 cm
// forward, 4 cm right and 35 cm up of the body origin. Maps camera-frame
// points into the body frame.
constexpr Extrinsics synthetic_body_from_camera() {
  return Extrinsics{{0.0F, 0.0F, 1.0F, 0.12F,     // Body x: camera z.
                     -1.0F, 0.0F, 0.0F, -0.04F,   // Body y: camera -x.
                     0.0F, -1.0F, 0.0F, 0.35F}};  // Body z: camera -y.
}

// Every kDuplicateEvery-th pixel gets a second point kDuplicateDepthFactor
// times as far along its ray, which must lose the z-buffer. Doubling is exact,
// so a millimeter depth stays a whole number of millimeters.
inline constexpr std::uint32_t kDuplicateEvery = 10U;
inline constexpr float kDuplicateDepthFactor = 2.0F;
// Every kStrayEvery-th pixel gets a point behind the camera, and the pixel
// kOutsideOffset after it a point one image width to the right, outside the
// field of view.
inline constexpr std::uint32_t kStrayEvery = 50U;
inline constexpr std::uint32_t kOutsideOffset = 25U;

// How many points of each kind a synthetic cloud has. Records are numbered
// in this order before the cloud is shuffled.
struct SyntheticCloudLayout {
  std::uint32_t pixels{};      // One per pixel: on the surface, or non-finite in a hole.
  std::uint32_t duplicates{};  // Pixels 0, kDuplicateEvery, 2 * kDuplicateEvery, ...
  std::uint32_t behind{};      // Pixels 0, kStrayEvery, 2 * kStrayEvery, ...
  std::uint32_t outside{};     // Pixels kOutsideOffset, kOutsideOffset + kStrayEvery, ...

  [[nodiscard]] POINT_CLOUD_TO_DEPTH_HOST_DEVICE std::uint32_t total() const {
    return pixels + duplicates + behind + outside;
  }
};

POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline SyntheticCloudLayout synthetic_cloud_layout(int width,
                                                                                    int height) {
  const auto n = static_cast<std::uint32_t>(width) * static_cast<std::uint32_t>(height);
  SyntheticCloudLayout layout;
  layout.pixels = n;
  layout.duplicates = (n + kDuplicateEvery - 1U) / kDuplicateEvery;
  layout.behind = (n + kStrayEvery - 1U) / kStrayEvery;
  layout.outside = n > kOutsideOffset ? (n - kOutsideOffset + kStrayEvery - 1U) / kStrayEvery : 0U;
  return layout;
}

// Whether pixel `index` (v * width + u) has a duplicate point.
POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline bool has_duplicate(std::uint32_t index) {
  return index % kDuplicateEvery == 0U;
}

// The color of a duplicate of a pixel whose color is `rgb`: its inverse, so a
// duplicate that wins shows in the color image as well as in the depth.
POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline void duplicate_color(std::uint8_t rgb[3]) {
  for (int channel = 0; channel < 3; ++channel) {
    rgb[channel] = static_cast<std::uint8_t>(255U - rgb[channel]);
  }
}

// Record `record` of the cloud of frame `f`, moved by `body_from_camera`
// (synthetic_body_from_camera(), passed in as device code cannot call it),
// and its color.
POINT_CLOUD_TO_DEPTH_HOST_DEVICE inline void synthetic_point(
    const SyntheticFrame& f, const SyntheticCloudLayout& layout, bool millimeter_depth,
    const Extrinsics& body_from_camera, std::uint32_t record, float xyz[3], std::uint8_t rgb[3]) {
  enum class Kind { kSurface, kDuplicate, kBehind, kOutside };
  Kind kind = Kind::kSurface;
  std::uint32_t pixel = record;
  if (record >= layout.pixels + layout.duplicates + layout.behind) {
    kind = Kind::kOutside;
    pixel =
        (record - layout.pixels - layout.duplicates - layout.behind) * kStrayEvery + kOutsideOffset;
  } else if (record >= layout.pixels + layout.duplicates) {
    kind = Kind::kBehind;
    pixel = (record - layout.pixels - layout.duplicates) * kStrayEvery;
  } else if (record >= layout.pixels) {
    kind = Kind::kDuplicate;
    pixel = (record - layout.pixels) * kDuplicateEvery;
  }
  const int u = static_cast<int>(pixel % static_cast<std::uint32_t>(f.width));
  const int v = static_cast<int>(pixel / static_cast<std::uint32_t>(f.width));
  const float depth = synthetic_scene_depth(f, millimeter_depth, u, v);
  const bool hole = !(depth > 0.0F);
  float z = hole ? 1.0F : depth;
  if (kind == Kind::kDuplicate) {
    z *= kDuplicateDepthFactor;
  }
  const CameraIntrinsics& k = f.intrinsics;
  const float column =
      kind == Kind::kOutside ? static_cast<float>(u + f.width) : static_cast<float>(u);
  const float camera[3] = {(column - k.cx) * z / k.fx, (static_cast<float>(v) - k.cy) * z / k.fy,
                           kind == Kind::kBehind ? -z : z};
  const float* m = body_from_camera.matrix;
  for (int row = 0; row < 3; ++row) {
    xyz[row] = m[4 * row] * camera[0] + m[4 * row + 1] * camera[1] + m[4 * row + 2] * camera[2] +
               m[4 * row + 3];
  }
  if (hole) {
    // One non-finite coordinate, a different one from pixel to pixel.
    const std::uint32_t which = pixel % 4U;
#if defined(__CUDA_ARCH__)
    const float nan = __int_as_float(0x7FC00000);
    const float infinity = __int_as_float(0x7F800000);
#else
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
#endif
    xyz[which < 3U ? which : 0U] = which < 3U ? nan : infinity;
  }
  if (kind == Kind::kBehind || kind == Kind::kOutside) {
    rgb[0] = 255U;
    rgb[1] = 0U;
    rgb[2] = 255U;
    return;
  }
  synthetic_color(f, u, v, rgb);
  if (kind == Kind::kDuplicate) {
    duplicate_color(rgb);
  }
}

// The deterministic shuffle of a cloud: record r goes to position
// (multiplier * r + offset) % total, a permutation as multiplier and total are
// coprime.
struct CloudShuffle {
  std::uint64_t multiplier{1U};
  std::uint64_t offset{};
};

// The shuffle of frame `frame`'s cloud of `total` points.
[[nodiscard]] CloudShuffle synthetic_shuffle(std::uint32_t total, std::uint32_t frame);

// The cloud of frame `f` as the GPU makes it, made on the host: packed x, y, z
// in `xyz` and RGB in `rgb`, layout.total() points each.
void synthetic_cloud(const SyntheticFrame& f, bool millimeter_depth, std::vector<float>& xyz,
                     std::vector<std::uint8_t>& rgb);

// Makes the cloud of frame `f` in device buffers on `stream`: points as packed
// float32 x, y, z, and colors as RGB8. Either buffer may be null to skip it.
cudaError_t generate_synthetic_cloud(const SyntheticFrame& f, bool millimeter_depth, float* points,
                                     std::uint8_t* colors, cudaStream_t stream);

}  // namespace point_cloud_to_depth
