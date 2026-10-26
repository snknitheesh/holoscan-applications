// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <numeric>
#include <vector>

#include "synthetic_scene.hpp"

namespace point_cloud_to_depth {
namespace {

constexpr int kBlock = 256;
// A multiplier far from 0 and 1 modulo any cloud size, so that neighbors in
// the image end up far apart in the cloud.
constexpr std::uint64_t kShuffleMultiplier = 2654435761U;
constexpr std::uint64_t kShuffleOffsetStep = 7919U;

__global__ void cloud_kernel(SyntheticFrame f, SyntheticCloudLayout layout, bool millimeter_depth,
                             Extrinsics body_from_camera, CloudShuffle shuffle, float* points,
                             std::uint8_t* colors) {
  const std::uint32_t total = layout.total();
  const std::uint32_t record = blockIdx.x * blockDim.x + threadIdx.x;
  if (record >= total) {
    return;
  }
  float xyz[3];
  std::uint8_t rgb[3];
  synthetic_point(f, layout, millimeter_depth, body_from_camera, record, xyz, rgb);
  const std::uint64_t at = (shuffle.multiplier * record + shuffle.offset) % total;
  for (int axis = 0; axis < 3; ++axis) {
    if (points != nullptr) {
      points[3 * at + axis] = xyz[axis];
    }
    if (colors != nullptr) {
      colors[3 * at + axis] = rgb[axis];
    }
  }
}

}  // namespace

CloudShuffle synthetic_shuffle(std::uint32_t total, std::uint32_t frame) {
  // No multiplier of 2 or more is coprime with 1 or 2 modulo them.
  if (total <= 2U) {
    return {};
  }
  std::uint64_t multiplier = kShuffleMultiplier % total;
  while (multiplier < 2U || std::gcd(multiplier, std::uint64_t{total}) != 1U) {
    multiplier = (multiplier + 1U) % total;
  }
  return CloudShuffle{.multiplier = multiplier, .offset = (frame * kShuffleOffsetStep) % total};
}

void synthetic_cloud(const SyntheticFrame& f, bool millimeter_depth, std::vector<float>& xyz,
                     std::vector<std::uint8_t>& rgb) {
  const SyntheticCloudLayout layout = synthetic_cloud_layout(f.width, f.height);
  const std::uint32_t total = layout.total();
  const CloudShuffle shuffle = synthetic_shuffle(total, f.frame);
  xyz.assign(static_cast<std::size_t>(total) * 3U, 0.0F);
  rgb.assign(static_cast<std::size_t>(total) * 3U, 0U);
  for (std::uint32_t record = 0; record < total; ++record) {
    const std::uint64_t at = (shuffle.multiplier * record + shuffle.offset) % total;
    synthetic_point(f, layout, millimeter_depth, synthetic_body_from_camera(), record, &xyz[3 * at],
                    &rgb[3 * at]);
  }
}

cudaError_t generate_synthetic_cloud(const SyntheticFrame& f, bool millimeter_depth, float* points,
                                     std::uint8_t* colors, cudaStream_t stream) {
  const SyntheticCloudLayout layout = synthetic_cloud_layout(f.width, f.height);
  const std::uint32_t total = layout.total();
  if (total == 0U || (points == nullptr && colors == nullptr)) {
    return cudaSuccess;
  }
  // Clear an error an earlier, unrelated call left behind, so the check after
  // the launch reports this launch only and leaves nothing pending.
  static_cast<void>(cudaGetLastError());
  cloud_kernel<<<(total + kBlock - 1U) / kBlock, kBlock, 0, stream>>>(
      f, layout, millimeter_depth, synthetic_body_from_camera(), synthetic_shuffle(total, f.frame),
      points, colors);
  return cudaGetLastError();
}

}  // namespace point_cloud_to_depth
