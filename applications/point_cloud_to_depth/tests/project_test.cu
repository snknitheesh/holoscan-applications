// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The projection kernels against a CPU reference, pixel for pixel: random
// clouds through a rigid transform, every splat size, float32 and uint16
// depth, colors, ties and the z-buffer's determinism, the valid range,
// non-finite points, points behind the camera, the image border, uint16
// rounding and its limits, odd image sizes, intrinsics read from device
// memory, and the round trip of a depth image through its points. Also the
// synthetic cloud's GPU and host versions, and launch error reporting.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "projection/project.hpp"
#include "projection/synthetic_scene.hpp"
#include "test_support.hpp"

namespace point_cloud_to_depth {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInfinity = std::numeric_limits<float>::infinity();

// Stops the test case on a failed CUDA call, which would leave every later
// check reading garbage.
void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) {
    cuda_check(
        cudaMalloc(reinterpret_cast<void**>(&data_), std::max<std::size_t>(count, 1U) * sizeof(T)),
        "cudaMalloc");
  }
  ~Device() { cudaFree(data_); }
  Device(Device&& other) noexcept : data_(other.data_) { other.data_ = nullptr; }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;
  T* get() const { return data_; }

 private:
  T* data_{};
};

template <typename T>
Device<T> upload(const std::vector<T>& host) {
  Device<T> device(host.size());
  cuda_check(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy");
  return device;
}

template <typename T>
std::vector<T> download(const Device<T>& device, std::size_t count) {
  std::vector<T> host(count);
  cuda_check(cudaMemcpy(host.data(), device.get(), count * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy");
  return host;
}

// A depth image and its colors: float32 meters or uint16 counts as the
// params say, 0 and black where empty.
struct Image {
  std::vector<float> depth_m;
  std::vector<std::uint16_t> depth_counts;
  std::vector<std::uint8_t> rgb;
};

// The reference: every point in order, keeping per pixel the nearest one and
// the lower index among equally near ones.
Image reference_project(const std::vector<float>& points, const std::vector<std::uint8_t>& colors,
                        const ProjectParams& p, const CameraIntrinsics& k) {
  const std::size_t pixels = static_cast<std::size_t>(p.width) * p.height;
  std::vector<float> best_z(pixels, kInfinity);
  std::vector<std::size_t> best_point(pixels, SIZE_MAX);
  const float* m = p.extrinsics.matrix;
  const bool k_usable = k.fx > 0 && k.fy > 0 && std::isfinite(k.fx) && std::isfinite(k.fy) &&
                        std::isfinite(k.cx) && std::isfinite(k.cy);
  for (std::size_t i = 0; k_usable && i < points.size() / 3; ++i) {
    const float px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
    if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) continue;
    const float x = m[0] * px + m[1] * py + m[2] * pz + m[3];
    const float y = m[4] * px + m[5] * py + m[6] * pz + m[7];
    const float z = m[8] * px + m[9] * py + m[10] * pz + m[11];
    if (!std::isfinite(z) || z <= 0 || z < p.depth_min || z > p.depth_max) continue;
    const float u = k.fx * x / z + k.cx;
    const float v = k.fy * y / z + k.cy;
    if (!(u > -1.0F && u < p.width && v > -1.0F && v < p.height)) continue;
    const std::int64_t cu = std::lround(u), cv = std::lround(v);
    if (cu < 0 || cu >= p.width || cv < 0 || cv >= p.height) continue;
    if (p.format == DepthFormat::kUint16) {
      const std::int64_t counts = std::lround(z / p.depth_scale);
      if (counts < 1 || counts > kMaxDepthCount) continue;
    }
    const int r = p.point_size / 2;
    for (std::int64_t row = cv - r; row <= cv + r; ++row) {
      for (std::int64_t col = cu - r; col <= cu + r; ++col) {
        if (row < 0 || row >= p.height || col < 0 || col >= p.width) continue;
        const std::size_t at = static_cast<std::size_t>(row) * p.width + col;
        if (z < best_z[at] || (z == best_z[at] && i < best_point[at])) {
          best_z[at] = z;
          best_point[at] = i;
        }
      }
    }
  }
  Image image;
  image.depth_m.assign(pixels, 0.0F);
  image.depth_counts.assign(pixels, 0U);
  image.rgb.assign(pixels * 3, 0U);
  for (std::size_t at = 0; at < pixels; ++at) {
    if (best_point[at] == SIZE_MAX) continue;
    image.depth_m[at] = best_z[at];
    image.depth_counts[at] = static_cast<std::uint16_t>(std::lround(best_z[at] / p.depth_scale));
    if (!colors.empty()) {
      std::memcpy(&image.rgb[3 * at], &colors[3 * best_point[at]], 3);
    }
  }
  return image;
}

// Runs the kernels on `points` (and `colors` when not empty).
Image gpu_project(const std::vector<float>& points, const std::vector<std::uint8_t>& colors,
                  const ProjectParams& p, cudaError_t* status = nullptr) {
  const std::size_t pixels = static_cast<std::size_t>(p.width) * p.height;
  auto d_points = upload(points);
  auto d_colors = upload(colors);
  Device<std::uint64_t> d_zbuffer(pixels);
  Device<float> d_depth(pixels);  // Large enough for either format.
  Device<std::uint8_t> d_rgb(pixels * 3);
  const cudaError_t launched =
      launch_project(d_points.get(), colors.empty() ? nullptr : d_colors.get(),
                     static_cast<std::uint32_t>(points.size() / 3), p, d_zbuffer.get(),
                     d_depth.get(), colors.empty() ? nullptr : d_rgb.get(), nullptr);
  if (status != nullptr) *status = launched;
  VERIFY(launched == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  Image image;
  if (p.format == DepthFormat::kUint16) {
    image.depth_counts.resize(pixels);
    cuda_check(
        cudaMemcpy(image.depth_counts.data(), d_depth.get(), pixels * 2, cudaMemcpyDeviceToHost),
        "cudaMemcpy");
  } else {
    image.depth_m = download(d_depth, pixels);
  }
  if (!colors.empty()) image.rgb = download(d_rgb, pixels * 3);
  return image;
}

// Checks the depth (bit for bit) and colors against the reference.
void check_same(const Image& got, const Image& want, const ProjectParams& p, bool colors,
                const std::string& what) {
  const std::size_t pixels = static_cast<std::size_t>(p.width) * p.height;
  int wrong = 0, wrong_colors = 0, filled = 0;
  for (std::size_t i = 0; i < pixels; ++i) {
    if (p.format == DepthFormat::kUint16) {
      wrong += got.depth_counts[i] != want.depth_counts[i];
      filled += want.depth_counts[i] != 0;
    } else {
      wrong += std::memcmp(&got.depth_m[i], &want.depth_m[i], sizeof(float)) != 0;
      filled += want.depth_m[i] != 0.0F;
    }
    if (colors) wrong_colors += std::memcmp(&got.rgb[3 * i], &want.rgb[3 * i], 3) != 0;
  }
  VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " depths differ");
  VERIFY_MSG(wrong_colors == 0, what + ": " + std::to_string(wrong_colors) + " colors differ");
  VERIFY_MSG(filled > 0, what + ": the reference image is empty, so it checks nothing");
}

// A rotation of `angle` radians about a unit axis, then a translation.
Extrinsics rigid(float ax, float ay, float az, float angle, float tx, float ty, float tz) {
  const float c = std::cos(angle), s = std::sin(angle), t = 1.0F - c;
  return Extrinsics{{t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay, tx,
                     t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax, ty,
                     t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c, tz}};
}

ProjectParams base_params(int width, int height) {
  ProjectParams p;
  p.width = width;
  p.height = height;
  p.intrinsics = {50.0F, 52.0F, 33.2F, 23.7F};
  p.depth_min = 0.2F;
  p.depth_max = 8.0F;
  return p;
}

// A random cloud: points spread over the view and beyond it, at random depths,
// some repeated exactly (equal depth, other index), and some not finite.
void random_cloud(std::uint32_t seed, std::size_t count, const ProjectParams& p,
                  std::vector<float>& points, std::vector<std::uint8_t>& colors) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> lateral(-1.2F, 1.2F);
  std::uniform_real_distribution<float> depth(-1.0F, 10.0F);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> pick(0, 99);
  const Extrinsics to_input = rigid_inverse(p.extrinsics);
  points.clear();
  colors.clear();
  for (std::size_t i = 0; i < count; ++i) {
    float camera[3];
    const int kind = pick(rng);
    if (kind < 5 && i > 0) {
      // An exact repeat of an earlier point: a tie the lower index wins.
      const std::size_t earlier = std::uniform_int_distribution<std::size_t>(0, i - 1)(rng);
      points.insert(points.end(), points.begin() + 3 * earlier, points.begin() + 3 * earlier + 3);
      for (int c = 0; c < 3; ++c) colors.push_back(static_cast<std::uint8_t>(byte(rng)));
      continue;
    }
    const float z = depth(rng);
    camera[0] = lateral(rng) * z;
    camera[1] = lateral(rng) * z;
    camera[2] = z;
    const float* m = to_input.matrix;
    for (int row = 0; row < 3; ++row) {
      points.push_back(m[4 * row] * camera[0] + m[4 * row + 1] * camera[1] +
                       m[4 * row + 2] * camera[2] + m[4 * row + 3]);
    }
    if (kind >= 97) points[points.size() - 1 - (kind - 97)] = kind == 99 ? kInfinity : kNaN;
    for (int c = 0; c < 3; ++c) colors.push_back(static_cast<std::uint8_t>(byte(rng)));
  }
}

void check_random_clouds(DepthFormat format, int point_size, float depth_min = 0.2F,
                         float depth_max = 8.0F) {
  ProjectParams p = base_params(67, 49);  // Not a multiple of the block size.
  p.format = format;
  p.point_size = point_size;
  p.depth_min = depth_min;
  p.depth_max = depth_max;
  p.depth_scale = 0.002F;
  p.extrinsics = rigid(0.48F, 0.6F, 0.64F, 0.3F, 0.25F, -0.1F, 0.4F);
  std::vector<float> points;
  std::vector<std::uint8_t> colors;
  random_cloud(7U + static_cast<std::uint32_t>(point_size), 20000, p, points, colors);
  const Image want = reference_project(points, colors, p, p.intrinsics);
  check_same(gpu_project(points, colors, p), want, p, true,
             std::string(format == DepthFormat::kUint16 ? "uint16" : "float32") + " point size " +
                 std::to_string(point_size));
}

// Many points crowd a tiny image: the nearest point wins, the lower index on
// equal depth, and repeated runs agree bit for bit.
void check_zbuffer_determinism() {
  ProjectParams p;
  p.width = 4;
  p.height = 3;
  p.intrinsics = {1.0F, 1.0F, 0.0F, 0.0F};
  p.depth_max = 100.0F;
  std::mt19937 rng(11U);
  std::uniform_int_distribution<int> pixel_u(0, 3), pixel_v(0, 2), step(1, 40);
  std::vector<float> points;
  std::vector<std::uint8_t> colors;
  for (std::uint32_t i = 0; i < 200000U; ++i) {
    // Depths on a coarse grid, so that many points tie.
    const float z = 0.25F * static_cast<float>(step(rng));
    points.insert(points.end(),
                  {static_cast<float>(pixel_u(rng)) * z, static_cast<float>(pixel_v(rng)) * z, z});
    colors.insert(colors.end(), {static_cast<std::uint8_t>(i & 0xFFU),
                                 static_cast<std::uint8_t>((i >> 8) & 0xFFU),
                                 static_cast<std::uint8_t>((i >> 16) & 0xFFU)});
  }
  const Image first = gpu_project(points, colors, p);
  check_same(first, reference_project(points, colors, p, p.intrinsics), p, true, "crowded image");
  for (int run = 0; run < 5; ++run) {
    const Image again = gpu_project(points, colors, p);
    VERIFY_MSG(std::memcmp(again.depth_m.data(), first.depth_m.data(), 4 * 12) == 0 &&
                   again.rgb == first.rgb,
               "run " + std::to_string(run) + " differs from the first");
  }
  // Every pixel holds depth 0.25, the nearest, from the first point there.
  for (int i = 0; i < 12; ++i) {
    VERIFY_MSG(first.depth_m[i] == 0.25F, "pixel " + std::to_string(i) + " has the nearest depth");
  }
}

// One point per pixel along a row of a width x 1 image with fx = 1, cx = 0 and
// z = 1, so a point's u is its x exactly.
std::vector<float> row_points(const std::vector<float>& xs, float z = 1.0F) {
  std::vector<float> points;
  for (const float x : xs) points.insert(points.end(), {x * z, 0.0F, z});
  return points;
}

ProjectParams row_params(int width) {
  ProjectParams p;
  p.width = width;
  p.height = 1;
  p.intrinsics = {1.0F, 1.0F, 0.0F, 0.0F};
  p.depth_min = 0.0F;
  p.depth_max = 1.0e6F;
  return p;
}

// The pixel nearest u is lroundf(u), half away from zero; the image covers
// u in (-0.5, width - 0.5).
void check_image_border() {
  ProjectParams p = row_params(4);
  // u = -0.5 rounds to -1 and 3.5 to 4: both outside. Far values, which
  // lroundf could not represent, are dropped too.
  const std::vector<float> outside{-0.5F, 3.5F, -1.0F, 4.0F, 1.0e30F, -1.0e30F};
  const Image none = gpu_project(row_points(outside), {}, p);
  for (int i = 0; i < 4; ++i) {
    VERIFY_MSG(none.depth_m[i] == 0.0F, "pixel " + std::to_string(i) + " stays empty");
  }
  // -0.49 rounds to pixel 0, 0.5 to 1, 2.5 to 3, 3.49 to 3.
  const std::vector<float> inside{-0.49F, 0.5F, 2.5F, 3.49F};
  const std::vector<float> depths{1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> points;
  for (std::size_t i = 0; i < inside.size(); ++i) {
    points.insert(points.end(), {inside[i] * depths[i], 0.0F, depths[i]});
  }
  const Image some = gpu_project(points, {}, p);
  VERIFY(some.depth_m[0] == 1.0F);
  VERIFY(some.depth_m[1] == 2.0F);
  VERIFY(some.depth_m[2] == 0.0F);
  VERIFY(some.depth_m[3] == 3.0F);  // Nearer than the point at 4 m.
  check_same(some, reference_project(points, {}, p, p.intrinsics), p, false, "image border");
}

// A square splat is clipped at the border, and a point whose own pixel is
// outside the image draws nothing, however much of its square would be inside.
void check_splat_border() {
  ProjectParams p = row_params(6);
  p.height = 5;
  p.point_size = 5;
  // Pixel (0, 0) and, outside, pixel (-1, 2).
  std::vector<float> points{0.0F, 0.0F, 1.0F, -1.0F, 2.0F, 1.0F};
  const Image image = gpu_project(points, {}, p);
  for (int v = 0; v < 5; ++v) {
    for (int u = 0; u < 6; ++u) {
      const bool covered = u <= 2 && v <= 2;
      VERIFY_MSG((image.depth_m[v * 6 + u] == 1.0F) == covered,
                 "pixel (" + std::to_string(u) + ", " + std::to_string(v) + ")");
    }
  }
  check_same(image, reference_project(points, {}, p, p.intrinsics), p, false, "splat border");
  // Pixel (5, 4), the bottom right corner, and, outside, pixel (6, 2).
  const std::vector<float> far_corner{5.0F, 4.0F, 1.0F, 6.0F, 2.0F, 1.0F};
  const Image clipped = gpu_project(far_corner, {}, p);
  for (int v = 0; v < 5; ++v) {
    for (int u = 0; u < 6; ++u) {
      const bool covered = u >= 3 && v >= 2;
      VERIFY_MSG((clipped.depth_m[v * 6 + u] == 1.0F) == covered,
                 "pixel (" + std::to_string(u) + ", " + std::to_string(v) + ") of the far corner");
    }
  }
}

// Points outside [depth_min, depth_max], behind the camera, at z = 0 or not
// finite are dropped and occlude nothing.
void check_dropped_points() {
  ProjectParams p = row_params(8);
  p.depth_min = 1.0F;
  p.depth_max = 5.0F;
  const std::vector<float> points{
      0.0F,    0.0F, 0.999F,    // Pixel 0: just nearer than depth_min.
      1.0F,    0.0F, 1.0F,      // Pixel 1: at depth_min, kept.
      10.0F,   0.0F, 5.0F,      // Pixel 2: at depth_max, kept.
      15.004F, 0.0F, 5.001F,    // Pixel 3: just beyond depth_max.
      -4.0F,   0.0F, -1.0F,     // Pixel 4 seen from behind: behind the camera.
      kNaN,    0.0F, 2.0F,      // Pixel 5: x not a number.
      12.0F,   kNaN, 2.0F,      // Pixel 6: y not a number.
      14.0F,   0.0F, kInfinity  // Pixel 7 at infinity: infinite z.
  };
  const Image alone = gpu_project(points, {}, p);
  const std::vector<float> kept{0.0F, 1.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (int u = 0; u < 8; ++u) {
    VERIFY_MSG(alone.depth_m[u] == kept[u],
               "pixel " + std::to_string(u) + " holds " + std::to_string(alone.depth_m[u]));
  }
  // Behind each dropped point, a farther kept one shows through.
  std::vector<float> with_background = points;
  for (int u = 0; u < 8; ++u) {
    with_background.insert(with_background.end(), {u * 4.5F, 0.0F, 4.5F});
  }
  const Image image = gpu_project(with_background, {}, p);
  const std::vector<float> shown{4.5F, 1.0F, 4.5F, 4.5F, 4.5F, 4.5F, 4.5F, 4.5F};
  for (int u = 0; u < 8; ++u) {
    VERIFY_MSG(image.depth_m[u] == shown[u],
               "pixel " + std::to_string(u) + " holds " + std::to_string(image.depth_m[u]));
  }
  // A point behind the camera or at z = 0 is dropped even when the range
  // reaches behind it (the kernel does not check the range it is given).
  p.depth_min = -10.0F;
  const Image behind =
      gpu_project({-3.0F, 0.0F, -1.0F, 0.0F, 0.0F, 0.0F, -2.0F, 0.0F, -0.0F}, {}, p);
  for (int u = 0; u < 8; ++u) {
    VERIFY_MSG(behind.depth_m[u] == 0.0F,
               "pixel " + std::to_string(u) + " holds " + std::to_string(behind.depth_m[u]));
  }
  // Without an upper bound, a z that overflows to infinity in the transform,
  // and would project onto cx, is still dropped.
  p.depth_min = 0.0F;
  p.depth_max = kInfinity;
  p.extrinsics.matrix[11] = 3.0e38F;
  const Image unbounded = gpu_project({0.0F, 0.0F, 3.0e38F}, {}, p);
  for (int u = 0; u < 8; ++u) VERIFY(unbounded.depth_m[u] == 0.0F);
}

// uint16 depth is round(z / scale), half away from zero; a point that rounds
// to 0 or beyond 65535 is dropped and occludes nothing.
void check_uint16_limits() {
  ProjectParams p = row_params(8);
  p.format = DepthFormat::kUint16;
  p.depth_scale = 1.0F;  // One count per meter: halves are exact.
  const std::vector<float> depths{2.5F, 3.4999F, 0.5F, 0.49F, 65535.49F, 65535.5F, 70000.0F, 7.0F};
  std::vector<float> points;
  for (int u = 0; u < 8; ++u) points.insert(points.end(), {u * depths[u], 0.0F, depths[u]});
  // A farther point behind the one rounding to 0, at pixel 3.
  points.insert(points.end(), {3.0F * 9.0F, 0.0F, 9.0F});
  const Image image = gpu_project(points, {}, p);
  const std::vector<std::uint16_t> want{3, 3, 1, 9, 65535, 0, 0, 7};
  for (int u = 0; u < 8; ++u) {
    VERIFY_MSG(image.depth_counts[u] == want[u], "pixel " + std::to_string(u) + " holds " +
                                                     std::to_string(image.depth_counts[u]) +
                                                     ", not " + std::to_string(want[u]));
  }
  // Millimeters, the default scale: the largest depth is 65.535 m; a point
  // that rounds above it is dropped, and the one behind shows through.
  p.depth_scale = 0.001F;
  const std::vector<float> meters{65.5349F, 65.535F, 65.5356F, 65.6F, 0.0004F, 0.0006F};
  std::vector<float> mm_points;
  for (int u = 0; u < 6; ++u) mm_points.insert(mm_points.end(), {u * meters[u], 0.0F, meters[u]});
  mm_points.insert(mm_points.end(), {2.0F * 70.0F, 0.0F, 70.0F});    // Behind pixel 2: dropped too.
  mm_points.insert(mm_points.end(), {4.0F * 0.002F, 0.0F, 0.002F});  // Behind pixel 4: shown.
  const Image mm = gpu_project(mm_points, {}, p);
  const std::vector<std::uint16_t> want_mm{65535, 65535, 0, 0, 2, 1};
  for (int u = 0; u < 6; ++u) {
    VERIFY_MSG(mm.depth_counts[u] == want_mm[u],
               "millimeter pixel " + std::to_string(u) + " holds " +
                   std::to_string(mm.depth_counts[u]) + ", not " + std::to_string(want_mm[u]));
  }
  check_same(mm, reference_project(mm_points, {}, p, p.intrinsics), p, false, "millimeters");
}

void check_odd_sizes() {
  for (const auto& [width, height] : {std::pair{1, 1}, std::pair{1, 7}, std::pair{257, 3}}) {
    ProjectParams p = base_params(width, height);
    // Most points land in the image, however small.
    p.intrinsics = {0.8F * width, 0.8F * height, (width - 1) / 2.0F, (height - 1) / 2.0F};
    std::vector<float> points;
    std::vector<std::uint8_t> colors;
    random_cloud(3U, 5000, p, points, colors);
    check_same(gpu_project(points, colors, p), reference_project(points, colors, p, p.intrinsics),
               p, true, std::to_string(width) + "x" + std::to_string(height));
  }
}

// The kernel reads the intrinsics from device memory, ignoring the host ones;
// unusable ones leave the image empty.
void check_device_intrinsics() {
  ProjectParams truth = base_params(67, 49);
  std::vector<float> points;
  std::vector<std::uint8_t> colors;
  random_cloud(5U, 10000, truth, points, colors);
  const std::vector<float> k{truth.intrinsics.fx, truth.intrinsics.fy, truth.intrinsics.cx,
                             truth.intrinsics.cy};
  auto d_k = upload(k);
  ProjectParams p = truth;
  p.intrinsics = {1.0F, 1.0F, 0.0F, 0.0F};  // Must not be used.
  p.device_intrinsics = d_k.get();
  check_same(gpu_project(points, colors, p),
             reference_project(points, colors, truth, truth.intrinsics), truth, true,
             "device intrinsics");
  for (const std::vector<float>& bad :
       std::vector<std::vector<float>>{{0.0F, 52.0F, 33.2F, 23.7F},
                                       {50.0F, -1.0F, 33.2F, 23.7F},
                                       {50.0F, 52.0F, kNaN, 23.7F},
                                       {kInfinity, 52.0F, 33.2F, 23.7F}}) {
    auto d_bad = upload(bad);
    p.device_intrinsics = d_bad.get();
    const Image image = gpu_project(points, colors, p);
    int filled = 0;
    for (const float z : image.depth_m) filled += z != 0.0F;
    VERIFY_MSG(filled == 0, std::to_string(filled) + " pixels filled with unusable intrinsics");
  }
}

// Deprojecting a depth image with some intrinsics and projecting the points
// back with the same ones reproduces it exactly, in any point order.
void check_round_trip(bool millimeters) {
  const SyntheticFrame f = synthetic_frame(96, 72, 3, false);
  ProjectParams p;
  p.width = f.width;
  p.height = f.height;
  p.intrinsics = f.intrinsics;
  p.depth_min = 0.1F;
  p.depth_max = 10.0F;
  p.format = millimeters ? DepthFormat::kUint16 : DepthFormat::kFloat32;
  std::vector<float> points;
  for (int v = 0; v < f.height; ++v) {
    for (int u = 0; u < f.width; ++u) {
      const float z = synthetic_scene_depth(f, millimeters, u, v);
      points.insert(points.end(), {(u - f.intrinsics.cx) * z / f.intrinsics.fx,
                                   (v - f.intrinsics.cy) * z / f.intrinsics.fy, z});
    }
  }
  // Reverse the order: the result must not depend on it.
  std::vector<float> reversed;
  for (std::size_t i = points.size() / 3; i-- > 0;) {
    reversed.insert(reversed.end(), points.begin() + 3 * i, points.begin() + 3 * i + 3);
  }
  for (const auto* cloud : {&points, &reversed}) {
    const Image image = gpu_project(*cloud, {}, p);
    int wrong = 0;
    int filled = 0;
    for (int v = 0; v < f.height; ++v) {
      for (int u = 0; u < f.width; ++u) {
        const std::size_t i = static_cast<std::size_t>(v) * f.width + u;
        const float z = synthetic_scene_depth(f, millimeters, u, v);
        const bool valid = z >= p.depth_min && z <= p.depth_max;
        filled += valid;
        if (millimeters) {
          wrong += image.depth_counts[i] != (valid ? synthetic_depth_mm(f, u, v) : 0U);
        } else {
          wrong += image.depth_m[i] != (valid ? z : 0.0F);
        }
      }
    }
    VERIFY_MSG(wrong == 0, std::to_string(wrong) + " pixels differ after the round trip");
    VERIFY(filled > f.width * f.height / 2);
  }
}

void check_no_points() {
  ProjectParams p = base_params(16, 8);
  Device<std::uint64_t> d_zbuffer(128);
  Device<float> d_depth(128);
  Device<std::uint8_t> d_rgb(384);
  cuda_check(cudaMemset(d_depth.get(), 0x7F, 128 * sizeof(float)), "cudaMemset");
  cuda_check(cudaMemset(d_rgb.get(), 0x7F, 384), "cudaMemset");
  VERIFY(launch_project(nullptr, nullptr, 0, p, d_zbuffer.get(), d_depth.get(), d_rgb.get(),
                        nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  for (const float z : download(d_depth, 128)) VERIFY(z == 0.0F);
  for (const std::uint8_t c : download(d_rgb, 384)) VERIFY(c == 0U);
}

// The GPU makes the same synthetic cloud as the host, bit for bit.
void check_synthetic_cloud(bool millimeters) {
  const SyntheticFrame f = synthetic_frame(80, 60, 6, true);
  std::vector<float> xyz;
  std::vector<std::uint8_t> rgb;
  synthetic_cloud(f, millimeters, xyz, rgb);
  const std::size_t total = synthetic_cloud_layout(80, 60).total();
  VERIFY(xyz.size() == 3 * total && rgb.size() == 3 * total);
  Device<float> d_xyz(xyz.size());
  Device<std::uint8_t> d_rgb(rgb.size());
  VERIFY(generate_synthetic_cloud(f, millimeters, d_xyz.get(), d_rgb.get(), nullptr) ==
         cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  const auto got_xyz = download(d_xyz, xyz.size());
  VERIFY(std::memcmp(got_xyz.data(), xyz.data(), xyz.size() * sizeof(float)) == 0);
  VERIFY(download(d_rgb, rgb.size()) == rgb);
  // The shuffle is a permutation: every surface point appears once.
  int finite = 0;
  for (std::size_t i = 0; i < total; ++i) {
    finite +=
        std::isfinite(xyz[3 * i]) && std::isfinite(xyz[3 * i + 1]) && std::isfinite(xyz[3 * i + 2]);
  }
  const SyntheticCloudLayout layout = synthetic_cloud_layout(80, 60);
  VERIFY_MSG(finite < static_cast<int>(total) && finite > static_cast<int>(layout.pixels),
             std::to_string(finite) + " finite points");
}

// The cloud shuffle is a permutation for every size, the smallest included,
// and is found without searching forever.
void check_shuffle_sizes() {
  for (std::uint32_t total = 1; total <= 64; ++total) {
    for (std::uint32_t frame = 0; frame < 3; ++frame) {
      const CloudShuffle shuffle = synthetic_shuffle(total, frame);
      std::vector<bool> hit(total, false);
      for (std::uint64_t record = 0; record < total; ++record) {
        hit[(shuffle.multiplier * record + shuffle.offset) % total] = true;
      }
      VERIFY_MSG(std::all_of(hit.begin(), hit.end(), [](bool h) { return h; }),
                 "the shuffle of " + std::to_string(total) + " points is a permutation");
    }
  }
}

void check_rejected_arguments() {
  const ProjectParams good = base_params(4, 4);
  auto d_points = upload(std::vector<float>{0.0F, 0.0F, 1.0F});
  auto d_colors = upload(std::vector<std::uint8_t>{1, 2, 3});
  Device<std::uint64_t> d_zbuffer(16);
  Device<float> d_depth(16);
  Device<std::uint8_t> d_rgb(48);
  const auto launch = [&](const ProjectParams& p, const std::uint8_t* colors, std::uint8_t* rgb,
                          std::uint64_t* zbuffer) {
    return launch_project(d_points.get(), colors, 1, p, zbuffer, d_depth.get(), rgb, nullptr);
  };
  for (const int size : {0, 2, 4, 7}) {
    ProjectParams p = good;
    p.point_size = size;
    VERIFY_MSG(launch(p, nullptr, nullptr, d_zbuffer.get()) == cudaErrorInvalidValue,
               "point size " + std::to_string(size) + " is rejected");
  }
  ProjectParams empty = good;
  empty.width = 0;
  VERIFY(launch(empty, nullptr, nullptr, d_zbuffer.get()) == cudaErrorInvalidValue);
  VERIFY(launch(good, nullptr, nullptr, nullptr) == cudaErrorInvalidValue);
  VERIFY(launch(good, nullptr, d_rgb.get(), d_zbuffer.get()) == cudaErrorInvalidValue);
  // atomicMin needs each 64-bit z-buffer entry aligned to 8 bytes.
  Device<std::uint64_t> d_spare(17);
  VERIFY(launch(good, nullptr, nullptr,
                reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::uint8_t*>(d_spare.get()) +
                                                 4)) == cudaErrorInvalidValue);
  VERIFY(launch(good, d_colors.get(), d_rgb.get(), d_zbuffer.get()) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
}

// A launch reports its own outcome: an error some earlier call left behind is
// neither blamed on it nor left pending after it.
void check_launch_errors() {
  const ProjectParams p = base_params(4, 4);
  auto d_points = upload(std::vector<float>{0.0F, 0.0F, 1.0F});
  Device<std::uint64_t> d_zbuffer(16);
  Device<float> d_depth(16);
  void* unused = nullptr;
  VERIFY(cudaMalloc(&unused, std::size_t{1} << 60) == cudaErrorMemoryAllocation);
  VERIFY(launch_project(d_points.get(), nullptr, 1, p, d_zbuffer.get(), d_depth.get(), nullptr,
                        nullptr) == cudaSuccess);
  VERIFY(cudaGetLastError() == cudaSuccess);
  VERIFY(cudaMalloc(&unused, std::size_t{1} << 60) == cudaErrorMemoryAllocation);
  Device<float> d_cloud(3 * synthetic_cloud_layout(64, 48).total());
  VERIFY(generate_synthetic_cloud(synthetic_frame(64, 48, 0, false), false, d_cloud.get(), nullptr,
                                  nullptr) == cudaSuccess);
  VERIFY(cudaGetLastError() == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
}

void check_rigid_inverse() {
  const Extrinsics forward = rigid(0.48F, 0.6F, 0.64F, 0.3F, 0.25F, -0.1F, 0.4F);
  const Extrinsics inverse = rigid_inverse(forward);
  const float* a = forward.matrix;
  const float* b = inverse.matrix;
  // inverse * forward is the identity.
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      float value = b[4 * row] * a[col] + b[4 * row + 1] * a[4 + col] + b[4 * row + 2] * a[8 + col];
      if (col == 3) value += b[4 * row + 3];
      const float want = col == row ? 1.0F : 0.0F;
      VERIFY_MSG(std::fabs(value - want) < 1e-6F,
                 "inverse * forward at (" + std::to_string(row) + ", " + std::to_string(col) + ")");
    }
  }
}

}  // namespace
}  // namespace point_cloud_to_depth

int main() {
  namespace pcd = point_cloud_to_depth;
  using pcd::DepthFormat;
  return pcd::testing::run({
      {"float32, point size 1", [] { pcd::check_random_clouds(DepthFormat::kFloat32, 1); }},
      {"float32, point size 3", [] { pcd::check_random_clouds(DepthFormat::kFloat32, 3); }},
      {"float32, point size 5", [] { pcd::check_random_clouds(DepthFormat::kFloat32, 5); }},
      {"uint16, point size 1", [] { pcd::check_random_clouds(DepthFormat::kUint16, 1); }},
      {"uint16, point size 5", [] { pcd::check_random_clouds(DepthFormat::kUint16, 5); }},
      {"uint16, point size 3, narrow range",
       [] { pcd::check_random_clouds(DepthFormat::kUint16, 3, 1.5F, 4.0F); }},
      {"the z-buffer is deterministic", pcd::check_zbuffer_determinism},
      {"the image border", pcd::check_image_border},
      {"splats at the image border", pcd::check_splat_border},
      {"dropped points", pcd::check_dropped_points},
      {"uint16 rounding and limits", pcd::check_uint16_limits},
      {"odd image sizes", pcd::check_odd_sizes},
      {"intrinsics in device memory", pcd::check_device_intrinsics},
      {"float32 depth round trip", [] { pcd::check_round_trip(false); }},
      {"uint16 depth round trip", [] { pcd::check_round_trip(true); }},
      {"no points", pcd::check_no_points},
      {"synthetic cloud, meters", [] { pcd::check_synthetic_cloud(false); }},
      {"synthetic cloud, millimeters", [] { pcd::check_synthetic_cloud(true); }},
      {"the cloud shuffle at every size", pcd::check_shuffle_sizes},
      {"rejected arguments", pcd::check_rejected_arguments},
      {"launch errors", pcd::check_launch_errors},
      {"rigid inverse", pcd::check_rigid_inverse},
  });
}
