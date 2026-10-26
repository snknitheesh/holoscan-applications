// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "deproject.hpp"

namespace depth_to_point_cloud {
namespace {

constexpr int kBlock = 256;

// One thread per pixel, grid-stride; reads and writes are coalesced.
template <typename T>
__global__ void deproject_kernel(const T* __restrict__ depth, DeprojectParams p,
                                 const uchar3* __restrict__ color3,
                                 const uchar4* __restrict__ color4, bool bgr,
                                 float3* __restrict__ out_xyz, uchar3* __restrict__ out_rgb,
                                 int width, int height) {
  const int n = width * height;
  CameraIntrinsics k = p.intrinsics;
  if (p.device_intrinsics != nullptr) {
    k = CameraIntrinsics{p.device_intrinsics[0], p.device_intrinsics[1], p.device_intrinsics[2],
                         p.device_intrinsics[3]};
  }
  // Checked on the host for host intrinsics; device ones are only seen here.
  const bool usable = k.fx > 0.0F && k.fy > 0.0F && isfinite(k.fx) && isfinite(k.fy) &&
                      isfinite(k.cx) && isfinite(k.cy);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const int u = i % width;
    const int v = i / width;
    const float z = static_cast<float>(depth[i]) * p.depth_scale;
    // A positive-depth check (rather than raw != 0) rejects zero, negative and
    // NaN depths alike for both element types.
    const bool valid = usable && z > 0.0F && isfinite(z) && z >= p.depth_min && z <= p.depth_max;
    out_xyz[i] = valid ? make_float3((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z)
                       : make_float3(p.invalid_value, p.invalid_value, p.invalid_value);
    if (out_rgb != nullptr) {
      uchar3 c;
      if (color4 != nullptr) {
        const uchar4 c4 = color4[i];
        c = make_uchar3(c4.x, c4.y, c4.z);
      } else {
        c = color3[i];
      }
      out_rgb[i] = bgr ? make_uchar3(c.z, c.y, c.x) : c;
    }
  }
}

}  // namespace

cudaError_t launch_deproject(const void* depth, const DeprojectParams& params, ColorInput color,
                             float3* out_xyz, uchar3* out_rgb, int width, int height,
                             cudaStream_t stream) {
  if (width <= 0 || height <= 0) {
    return cudaSuccess;
  }
  const uchar3* color3 = nullptr;
  const uchar4* color4 = nullptr;
  if (out_rgb != nullptr) {
    if (color.data == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (color.channels == 4) {
      color4 = static_cast<const uchar4*>(color.data);
    } else if (color.channels == 3) {
      color3 = static_cast<const uchar3*>(color.data);
    } else {
      return cudaErrorInvalidValue;  // Only 3- or 4-channel uint8 color.
    }
  }
  const int n = width * height;
  const int grid = (n + kBlock - 1) / kBlock;
  // Clear an error an earlier, unrelated call left behind, so the check after
  // the launch reports this launch only and leaves nothing pending.
  static_cast<void>(cudaGetLastError());
  switch (params.depth_type) {
    case DepthType::kUint16:
      deproject_kernel<std::uint16_t>
          <<<grid, kBlock, 0, stream>>>(static_cast<const std::uint16_t*>(depth), params, color3,
                                        color4, color.bgr, out_xyz, out_rgb, width, height);
      break;
    case DepthType::kFloat32:
      deproject_kernel<float><<<grid, kBlock, 0, stream>>>(static_cast<const float*>(depth), params,
                                                           color3, color4, color.bgr, out_xyz,
                                                           out_rgb, width, height);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace depth_to_point_cloud
