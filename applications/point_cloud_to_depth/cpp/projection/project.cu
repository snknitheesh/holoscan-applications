// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "project.hpp"

namespace point_cloud_to_depth {
namespace {

constexpr int kBlock = 256;
// An empty z-buffer entry: larger than every key a point can write.
constexpr std::uint64_t kEmptyKey = ~std::uint64_t{0};
// atomicMin has a 64-bit overload for unsigned long long only.
static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));  // NOLINT(runtime/int)

__device__ bool usable(const CameraIntrinsics& k) {
  return k.fx > 0.0F && k.fy > 0.0F && isfinite(k.fx) && isfinite(k.fy) && isfinite(k.cx) &&
         isfinite(k.cy);
}

// One thread per point, grid-stride. Each point writes a 64-bit key into the
// z-buffer with atomicMin: the float bits of its depth (positive, so their
// unsigned order is the depth order) above its index. The smallest key is
// the nearest point, and the lower index among equally near ones.
__global__ void project_kernel(const float* __restrict__ points, std::uint32_t count,
                               ProjectParams p, std::uint64_t* __restrict__ zbuffer) {
  CameraIntrinsics k = p.intrinsics;
  if (p.device_intrinsics != nullptr) {
    k = CameraIntrinsics{p.device_intrinsics[0], p.device_intrinsics[1], p.device_intrinsics[2],
                         p.device_intrinsics[3]};
  }
  // Checked on the host for host intrinsics; device ones are only seen here.
  if (!usable(k)) {
    return;
  }
  const float* m = p.extrinsics.matrix;
  const int radius = p.point_size / 2;
  const auto width = static_cast<float>(p.width);
  const auto height = static_cast<float>(p.height);
  const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const float px = points[3 * i];
    const float py = points[3 * i + 1];
    const float pz = points[3 * i + 2];
    if (!(isfinite(px) && isfinite(py) && isfinite(pz))) {
      continue;
    }
    const float x = m[0] * px + m[1] * py + m[2] * pz + m[3];
    const float y = m[4] * px + m[5] * py + m[6] * pz + m[7];
    const float z = m[8] * px + m[9] * py + m[10] * pz + m[11];
    // Also rejects a NaN or infinite z, and points behind the camera.
    if (!(z > 0.0F && z >= p.depth_min && z <= p.depth_max && isfinite(z))) {
      continue;
    }
    const float u = k.fx * x / z + k.cx;
    const float v = k.fy * y / z + k.cy;
    // Bounded before rounding, which is undefined for values beyond a long;
    // this also rejects non-finite coordinates.
    if (!(u > -1.0F && u < width && v > -1.0F && v < height)) {
      continue;
    }
    const int pixel_u = static_cast<int>(lroundf(u));
    const int pixel_v = static_cast<int>(lroundf(v));
    if (pixel_u < 0 || pixel_u >= p.width || pixel_v < 0 || pixel_v >= p.height) {
      continue;
    }
    if (p.format == DepthFormat::kUint16) {
      // Rounds to 1..kMaxDepthCount, half away from zero, as the resolve does.
      const float counts = z / p.depth_scale;
      if (!(counts >= 0.5F && counts < static_cast<float>(kMaxDepthCount) + 0.5F)) {
        continue;
      }
    }
    const std::uint64_t key =
        (static_cast<std::uint64_t>(__float_as_uint(z)) << 32) | static_cast<std::uint32_t>(i);
    // The square around the pixel, clipped at the image border.
    const int first_row = pixel_v > radius ? pixel_v - radius : 0;
    const int last_row = pixel_v + radius < p.height ? pixel_v + radius : p.height - 1;
    const int first_col = pixel_u > radius ? pixel_u - radius : 0;
    const int last_col = pixel_u + radius < p.width ? pixel_u + radius : p.width - 1;
    for (int row = first_row; row <= last_row; ++row) {
      for (int col = first_col; col <= last_col; ++col) {
        std::uint64_t* entry = &zbuffer[static_cast<std::size_t>(row) * p.width + col];
        atomicMin(reinterpret_cast<unsigned long long*>(entry), key);  // NOLINT(runtime/int)
      }
    }
  }
}

// One thread per pixel: the winning key gives the depth and the point whose
// color the pixel takes.
__global__ void resolve_kernel(const std::uint64_t* __restrict__ zbuffer,
                               const std::uint8_t* __restrict__ colors, ProjectParams p,
                               void* __restrict__ depth, std::uint8_t* __restrict__ rgb) {
  const std::size_t n = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const std::uint64_t key = zbuffer[i];
    const bool filled = key != kEmptyKey;
    const float z = filled ? __uint_as_float(static_cast<unsigned int>(key >> 32)) : 0.0F;
    if (p.format == DepthFormat::kUint16) {
      static_cast<std::uint16_t*>(depth)[i] =
          filled ? static_cast<std::uint16_t>(lroundf(z / p.depth_scale)) : std::uint16_t{0};
    } else {
      static_cast<float*>(depth)[i] = z;
    }
    if (rgb != nullptr) {
      const std::uint64_t point = key & 0xFFFFFFFFULL;
      for (int channel = 0; channel < 3; ++channel) {
        rgb[3 * i + channel] = filled ? colors[3 * point + channel] : std::uint8_t{0};
      }
    }
  }
}

}  // namespace

Extrinsics rigid_inverse(const Extrinsics& transform) {
  const float* m = transform.matrix;
  Extrinsics inverse;
  float* r = inverse.matrix;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      r[4 * row + col] = m[4 * col + row];
    }
    r[4 * row + 3] = -(r[4 * row] * m[3] + r[4 * row + 1] * m[7] + r[4 * row + 2] * m[11]);
  }
  return inverse;
}

cudaError_t launch_project(const float* points, const std::uint8_t* colors,
                           std::uint32_t point_count, const ProjectParams& params,
                           std::uint64_t* zbuffer, void* depth, std::uint8_t* rgb,
                           cudaStream_t stream) {
  // atomicMin on a z-buffer entry needs it 8-byte aligned.
  if (params.width < 1 || params.height < 1 || !valid_point_size(params.point_size) ||
      zbuffer == nullptr ||
      reinterpret_cast<std::uintptr_t>(zbuffer) % alignof(std::uint64_t) != 0U ||
      depth == nullptr || (point_count != 0 && points == nullptr) ||
      (rgb != nullptr && point_count != 0 && colors == nullptr)) {
    return cudaErrorInvalidValue;
  }
  const std::size_t pixels = zbuffer_entries(params.width, params.height);
  // Clear an error an earlier, unrelated call left behind, so the checks
  // below report this launch only and leave nothing pending.
  static_cast<void>(cudaGetLastError());
  // All bits set: every entry holds kEmptyKey.
  if (const cudaError_t status =
          cudaMemsetAsync(zbuffer, 0xFF, pixels * sizeof(std::uint64_t), stream);
      status != cudaSuccess) {
    return status;
  }
  if (point_count != 0) {
    const auto blocks =
        static_cast<unsigned int>((static_cast<std::uint64_t>(point_count) + kBlock - 1U) / kBlock);
    project_kernel<<<blocks, kBlock, 0, stream>>>(points, point_count, params, zbuffer);
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return status;
    }
  }
  const auto blocks = static_cast<unsigned int>((pixels + kBlock - 1U) / kBlock);
  resolve_kernel<<<blocks, kBlock, 0, stream>>>(zbuffer, colors, params, depth, rgb);
  return cudaGetLastError();
}

}  // namespace point_cloud_to_depth
