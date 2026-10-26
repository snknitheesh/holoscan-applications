// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "synthetic_depth.hpp"

namespace depth_to_point_cloud {
namespace {

constexpr int kBlock = 256;

__global__ void render_kernel(SyntheticFrame f, bool millimeters, void* depth, std::uint8_t* rgb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= f.width * f.height) {
    return;
  }
  const int u = i % f.width;
  const int v = i / f.width;
  if (millimeters) {
    static_cast<std::uint16_t*>(depth)[i] = synthetic_depth_mm(f, u, v);
  } else {
    static_cast<float*>(depth)[i] = synthetic_depth(f, u, v);
  }
  if (rgb != nullptr) {
    synthetic_color(f, u, v, rgb + 3 * i);
  }
}

}  // namespace

cudaError_t render_synthetic(const SyntheticFrame& f, bool millimeters, void* depth,
                             std::uint8_t* rgb, cudaStream_t stream) {
  const int n = f.width * f.height;
  if (n <= 0) {
    return cudaSuccess;
  }
  // Clear an error an earlier, unrelated call left behind, so the check after
  // the launch reports this launch only and leaves nothing pending.
  static_cast<void>(cudaGetLastError());
  render_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(f, millimeters, depth, rgb);
  return cudaGetLastError();
}

}  // namespace depth_to_point_cloud
