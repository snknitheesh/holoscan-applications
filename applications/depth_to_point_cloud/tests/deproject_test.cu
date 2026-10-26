// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The deprojection kernel against a CPU reference: float32 and uint16 depth,
// holes, the valid range, non-finite depth, both color layouts and the
// channel swap, image sizes not a multiple of the block size, intrinsics read
// from device memory, the rejected color layouts, and launch error reporting.

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "deproject/deproject.hpp"
#include "deproject/synthetic_depth.hpp"
#include "test_support.hpp"

namespace depth_to_point_cloud {
namespace {

template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) {
    cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
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
  cudaMemcpy(device.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
  return device;
}

template <typename T>
std::vector<T> download(const Device<T>& device, std::size_t count) {
  std::vector<T> host(count);
  cudaMemcpy(host.data(), device.get(), count * sizeof(T), cudaMemcpyDeviceToHost);
  return host;
}

// Checks every point against the reference formula.
void check_points(const std::vector<float>& depth_m, const std::vector<float3>& got, int width,
                  const DeprojectParams& p, const std::string& what) {
  int wrong = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const int u = static_cast<int>(i) % width;
    const int v = static_cast<int>(i) / width;
    const float z = depth_m[i];
    const bool valid = z > 0.0F && std::isfinite(z) && z >= p.depth_min && z <= p.depth_max;
    const float3 o = got[i];
    if (!valid) {
      wrong += !(std::isnan(o.x) && std::isnan(o.y) && std::isnan(o.z));
      continue;
    }
    const float x = (u - p.intrinsics.cx) * z / p.intrinsics.fx;
    const float y = (v - p.intrinsics.cy) * z / p.intrinsics.fy;
    wrong += std::fabs(o.x - x) > 1e-5F || std::fabs(o.y - y) > 1e-5F || std::fabs(o.z - z) > 1e-6F;
  }
  VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " points differ");
}

DeprojectParams params(DepthType type, float scale) {
  DeprojectParams p;
  p.depth_type = type;
  p.depth_scale = scale;
  p.intrinsics = {50.0F, 52.0F, 31.5F, 23.5F};
  p.depth_min = 0.1F;
  p.depth_max = 10.0F;
  p.invalid_value = std::numeric_limits<float>::quiet_NaN();
  return p;
}

void check_float_depth() {
  // 67 x 49: not a multiple of the 256-thread block.
  const int w = 67, h = 49, n = w * h;
  std::vector<float> depth(n);
  for (int i = 0; i < n; ++i) depth[i] = 0.5F + 0.001F * static_cast<float>(i % 997);
  depth[10] = 0.0F;                                     // A hole.
  depth[11] = -1.0F;                                    // Negative.
  depth[12] = std::numeric_limits<float>::quiet_NaN();  // Not a number.
  depth[13] = std::numeric_limits<float>::infinity();   // Infinite.
  depth[14] = 0.05F;                                    // Below the range.
  depth[15] = 50.0F;                                    // Above the range.
  const DeprojectParams p = params(DepthType::kFloat32, 1.0F);
  auto d_depth = upload(depth);
  Device<float3> d_xyz(n);
  VERIFY(launch_deproject(d_depth.get(), p, {}, d_xyz.get(), nullptr, w, h, nullptr) ==
         cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  check_points(depth, download(d_xyz, n), w, p, "float32");
}

// The kernel reads the intrinsics from device memory, ignoring the host ones.
void check_device_intrinsics() {
  const int w = 67, h = 49, n = w * h;
  std::vector<float> depth(n);
  for (int i = 0; i < n; ++i) depth[i] = 0.5F + 0.001F * static_cast<float>(i % 997);
  depth[10] = 0.0F;
  const DeprojectParams truth = params(DepthType::kFloat32, 1.0F);
  const std::vector<float> k{truth.intrinsics.fx, truth.intrinsics.fy, truth.intrinsics.cx,
                             truth.intrinsics.cy};
  auto d_k = upload(k);
  DeprojectParams p = truth;
  p.intrinsics = {1.0F, 1.0F, 0.0F, 0.0F};  // Must not be used.
  p.device_intrinsics = d_k.get();
  auto d_depth = upload(depth);
  Device<float3> d_xyz(n);
  VERIFY(launch_deproject(d_depth.get(), p, {}, d_xyz.get(), nullptr, w, h, nullptr) ==
         cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  check_points(depth, download(d_xyz, n), w, truth, "device intrinsics");
}

// Device intrinsics the host never checks: unusable ones make every point invalid.
void check_unusable_device_intrinsics() {
  const int w = 16, h = 8, n = w * h;
  const std::vector<float> depth(n, 2.0F);
  auto d_depth = upload(depth);
  for (const std::vector<float>& k : std::vector<std::vector<float>>{
           {0.0F, 52.0F, 31.5F, 23.5F},
           {50.0F, -1.0F, 31.5F, 23.5F},
           {50.0F, 52.0F, std::numeric_limits<float>::quiet_NaN(), 23.5F},
           {std::numeric_limits<float>::infinity(), 52.0F, 31.5F, 23.5F}}) {
    auto d_k = upload(k);
    DeprojectParams p = params(DepthType::kFloat32, 1.0F);
    p.device_intrinsics = d_k.get();
    Device<float3> d_xyz(n);
    VERIFY(launch_deproject(d_depth.get(), p, {}, d_xyz.get(), nullptr, w, h, nullptr) ==
           cudaSuccess);
    VERIFY(cudaDeviceSynchronize() == cudaSuccess);
    int valid = 0;
    for (const float3& o : download(d_xyz, n)) {
      valid += !(std::isnan(o.x) && std::isnan(o.y) && std::isnan(o.z));
    }
    VERIFY_MSG(valid == 0, std::to_string(valid) + " points valid with unusable intrinsics");
  }
}

// A launch reports its own outcome: an error some earlier call left behind is
// neither blamed on it nor left pending after it.
void check_launch_errors() {
  const int w = 4, h = 2, n = w * h;
  auto d_depth = upload(std::vector<float>(n, 2.0F));
  Device<float3> d_xyz(n);
  Device<float> d_render(n);
  void* unused = nullptr;
  VERIFY(cudaMalloc(&unused, std::size_t{1} << 60) == cudaErrorMemoryAllocation);
  VERIFY(launch_deproject(d_depth.get(), params(DepthType::kFloat32, 1.0F), {}, d_xyz.get(),
                          nullptr, w, h, nullptr) == cudaSuccess);
  VERIFY(cudaGetLastError() == cudaSuccess);
  VERIFY(cudaMalloc(&unused, std::size_t{1} << 60) == cudaErrorMemoryAllocation);
  VERIFY(render_synthetic(synthetic_frame(w, h, 0, false), false, d_render.get(), nullptr,
                          nullptr) == cudaSuccess);
  VERIFY(cudaGetLastError() == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
}

void check_uint16_depth() {
  const int w = 64, h = 48, n = w * h;
  std::vector<std::uint16_t> raw(n);
  std::vector<float> depth_m(n);
  for (int i = 0; i < n; ++i) {
    raw[i] = static_cast<std::uint16_t>(i % 7 == 0 ? 0 : 400 + (i % 4000));
    depth_m[i] = static_cast<float>(raw[i]) * 0.001F;
  }
  const DeprojectParams p = params(DepthType::kUint16, 0.001F);
  auto d_depth = upload(raw);
  Device<float3> d_xyz(n);
  VERIFY(launch_deproject(d_depth.get(), p, {}, d_xyz.get(), nullptr, w, h, nullptr) ==
         cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  check_points(depth_m, download(d_xyz, n), w, p, "uint16");
}

void check_colors(int channels, bool bgr) {
  const int w = 33, h = 17, n = w * h;
  std::vector<float> depth(n, 2.0F);
  std::vector<std::uint8_t> color(static_cast<std::size_t>(n) * channels);
  for (std::size_t i = 0; i < color.size(); ++i) color[i] = static_cast<std::uint8_t>(i * 7);
  const DeprojectParams p = params(DepthType::kFloat32, 1.0F);
  auto d_depth = upload(depth);
  auto d_color = upload(color);
  Device<float3> d_xyz(n);
  Device<uchar3> d_rgb(n);
  VERIFY(launch_deproject(d_depth.get(), p, {d_color.get(), channels, bgr}, d_xyz.get(),
                          d_rgb.get(), w, h, nullptr) == cudaSuccess);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  const auto rgb = download(d_rgb, n);
  int wrong = 0;
  for (int i = 0; i < n; ++i) {
    const std::uint8_t* c = color.data() + static_cast<std::size_t>(i) * channels;
    const std::uint8_t r = bgr ? c[2] : c[0];
    const std::uint8_t b = bgr ? c[0] : c[2];
    wrong += rgb[i].x != r || rgb[i].y != c[1] || rgb[i].z != b;
  }
  VERIFY_MSG(wrong == 0, std::to_string(channels) + " channels" + (bgr ? " bgr" : "") + ": " +
                             std::to_string(wrong) + " colors differ");
}

void check_rejected_color() {
  const DeprojectParams p = params(DepthType::kFloat32, 1.0F);
  Device<float> d_depth(4);
  Device<float3> d_xyz(4);
  Device<uchar3> d_rgb(4);
  Device<std::uint8_t> d_color(8);
  VERIFY(launch_deproject(d_depth.get(), p, {d_color.get(), 2, false}, d_xyz.get(), d_rgb.get(), 2,
                          2, nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_deproject(d_depth.get(), p, {}, d_xyz.get(), d_rgb.get(), 2, 2, nullptr) ==
         cudaErrorInvalidValue);
  VERIFY(launch_deproject(d_depth.get(), p, {}, d_xyz.get(), nullptr, 0, 0, nullptr) ==
         cudaSuccess);
}

}  // namespace
}  // namespace depth_to_point_cloud

int main() {
  using depth_to_point_cloud::testing::TestCase;
  namespace dpc = depth_to_point_cloud;
  return dpc::testing::run({
      {"float32 depth", dpc::check_float_depth},
      {"uint16 depth", dpc::check_uint16_depth},
      {"RGB8 colors", [] { dpc::check_colors(3, false); }},
      {"BGR8 colors", [] { dpc::check_colors(3, true); }},
      {"RGBA8 colors", [] { dpc::check_colors(4, false); }},
      {"BGRA8 colors", [] { dpc::check_colors(4, true); }},
      {"intrinsics in device memory", dpc::check_device_intrinsics},
      {"unusable intrinsics in device memory", dpc::check_unusable_device_intrinsics},
      {"rejected color layouts", dpc::check_rejected_color},
      {"launch errors", dpc::check_launch_errors},
  });
}
