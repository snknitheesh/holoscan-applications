// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "letterbox.hpp"

namespace yolo26_detection {
namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

// Per-launch constants, passed by value.
struct LetterboxKernelParams {
  const std::uint8_t* image;
  std::size_t pitch;
  int width;
  int height;
  int channels;
  // Offsets of red and blue within a pixel; green is always 1.
  int red;
  int blue;
  int size;
  int pad_left;
  int pad_top;
  int resized_width;
  int resized_height;
  // Source pixels per resized pixel, along each axis.
  float x_ratio;
  float y_ratio;
  float pad;  // pad_value / 255
};

// One output pixel per thread, written to all three planes.
__global__ void letterbox_kernel(LetterboxKernelParams p, float* output) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= p.size || v >= p.size) {
    return;
  }
  const std::size_t plane = static_cast<std::size_t>(p.size) * static_cast<std::size_t>(p.size);
  const std::size_t index = static_cast<std::size_t>(v) * static_cast<std::size_t>(p.size) + u;
  const int x = u - p.pad_left;
  const int y = v - p.pad_top;
  if (x < 0 || y < 0 || x >= p.resized_width || y >= p.resized_height) {
    output[index] = p.pad;
    output[plane + index] = p.pad;
    output[2 * plane + index] = p.pad;
    return;
  }
  // The source point under the center of resized pixel (x, y), clamped to the
  // image as OpenCV does at its edges.
  const float sx = fmaxf((static_cast<float>(x) + 0.5f) * p.x_ratio - 0.5f, 0.0f);
  const float sy = fmaxf((static_cast<float>(y) + 0.5f) * p.y_ratio - 0.5f, 0.0f);
  const int x0 = min(static_cast<int>(sx), p.width - 1);
  const int y0 = min(static_cast<int>(sy), p.height - 1);
  const int x1 = min(x0 + 1, p.width - 1);
  const int y1 = min(y0 + 1, p.height - 1);
  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);
  const std::uint8_t* row0 = p.image + static_cast<std::size_t>(y0) * p.pitch;
  const std::uint8_t* row1 = p.image + static_cast<std::size_t>(y1) * p.pitch;
  const int c0 = x0 * p.channels;
  const int c1 = x1 * p.channels;
  const int offsets[3] = {p.red, 1, p.blue};
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int k = offsets[c];
    const float top =
        (1.0f - fx) * static_cast<float>(row0[c0 + k]) + fx * static_cast<float>(row0[c1 + k]);
    const float bottom =
        (1.0f - fx) * static_cast<float>(row1[c0 + k]) + fx * static_cast<float>(row1[c1 + k]);
    output[c * plane + index] = ((1.0f - fy) * top + fy * bottom) / 255.0f;
  }
}

}  // namespace

Letterbox make_letterbox(int source_width, int source_height, int size) {
  if (source_width < 1 || source_height < 1 || source_width > kMaxImageSide ||
      source_height > kMaxImageSide) {
    throw std::invalid_argument("image sides must be 1.." + std::to_string(kMaxImageSide) +
                                ", got " + std::to_string(source_width) + "x" +
                                std::to_string(source_height));
  }
  if (size < kMinInputSize || size > kMaxInputSize) {
    throw std::invalid_argument("input size must be " + std::to_string(kMinInputSize) + ".." +
                                std::to_string(kMaxInputSize));
  }
  Letterbox result;
  result.source_width = source_width;
  result.source_height = source_height;
  result.size = size;
  const double scale =
      std::min(static_cast<double>(size) / source_width, static_cast<double>(size) / source_height);
  // nearbyint rounds half to even in the default rounding mode, as Python's
  // round() does; the result never exceeds `size`.
  result.resized_width =
      std::clamp(static_cast<int>(std::nearbyint(source_width * scale)), 1, size);
  result.resized_height =
      std::clamp(static_cast<int>(std::nearbyint(source_height * scale)), 1, size);
  // round(d / 2 - 0.1) for the whole padding d: the left or top half, rounded
  // down when d is odd.
  result.pad_left = (size - result.resized_width) / 2;
  result.pad_top = (size - result.resized_height) / 2;
  result.scale = static_cast<float>(scale);
  return result;
}

cudaError_t launch_letterbox(const ImageView& image, const Letterbox& letterbox,
                             std::uint8_t pad_value, float* output, cudaStream_t stream) {
  if (image.data == nullptr || output == nullptr || image.width != letterbox.source_width ||
      image.height != letterbox.source_height ||
      image.pitch < static_cast<std::size_t>(image.width) * channels(image.format)) {
    return cudaErrorInvalidValue;
  }
  const bool bgr = blue_first(image.format);
  const LetterboxKernelParams params{
      .image = image.data,
      .pitch = image.pitch,
      .width = image.width,
      .height = image.height,
      .channels = channels(image.format),
      .red = bgr ? 2 : 0,
      .blue = bgr ? 0 : 2,
      .size = letterbox.size,
      .pad_left = letterbox.pad_left,
      .pad_top = letterbox.pad_top,
      .resized_width = letterbox.resized_width,
      .resized_height = letterbox.resized_height,
      .x_ratio = static_cast<float>(image.width) / static_cast<float>(letterbox.resized_width),
      .y_ratio = static_cast<float>(image.height) / static_cast<float>(letterbox.resized_height),
      .pad = static_cast<float>(pad_value) / 255.0f,
  };
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((letterbox.size + kBlockX - 1) / kBlockX,
                  (letterbox.size + kBlockY - 1) / kBlockY);
  letterbox_kernel<<<grid, block, 0, stream>>>(params, output);
  return cudaGetLastError();
}

}  // namespace yolo26_detection
