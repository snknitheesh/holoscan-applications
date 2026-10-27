// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The letterbox that fits an image into a YOLO model's square input: the image
// is scaled by one factor to fit, keeping its aspect ratio, centered, and the
// rest is filled with a constant gray. The geometry follows the Ultralytics
// predictor, so boxes map back to the image as they do there. No Holoscan.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace yolo26_detection {

// Interleaved 8-bit color layouts the letterbox reads.
enum class PixelFormat : std::uint8_t { kRgb8, kBgr8, kRgba8, kBgra8 };

[[nodiscard]] constexpr int channels(PixelFormat format) noexcept {
  return format == PixelFormat::kRgba8 || format == PixelFormat::kBgra8 ? 4 : 3;
}

// Whether the first channel of a pixel is blue rather than red.
[[nodiscard]] constexpr bool blue_first(PixelFormat format) noexcept {
  return format == PixelFormat::kBgr8 || format == PixelFormat::kBgra8;
}

// Largest image side the letterbox accepts, and the model input sizes.
inline constexpr int kMaxImageSide = 16384;
inline constexpr int kMinInputSize = 32;
inline constexpr int kMaxInputSize = 4096;

// Where an image of source_width x source_height lands in a size x size input:
// scaled by `scale` to resized_width x resized_height, its top-left corner at
// (pad_left, pad_top). A box in input pixels maps back to the image as
// (x - pad_left) / scale, (y - pad_top) / scale.
struct Letterbox {
  int source_width{};
  int source_height{};
  int size{};
  float scale{};
  int resized_width{};
  int resized_height{};
  int pad_left{};
  int pad_top{};
};

// The letterbox of a source_width x source_height image in a size x size
// input: scale = min(size / width, size / height); the resized sides are
// rounded half to even, and the padding is split with the odd pixel, if any,
// on the right or bottom, as the Ultralytics LetterBox does.
// @throws std::invalid_argument for a side outside 1..kMaxImageSide or a size
// outside kMinInputSize..kMaxInputSize.
[[nodiscard]] Letterbox make_letterbox(int source_width, int source_height, int size);

// An image in GPU memory: rows of `width` packed pixels, `pitch` bytes apart.
struct ImageView {
  const std::uint8_t* data{};
  std::size_t pitch{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::kRgb8};
};

// Writes the letterboxed image as float32 [3, size, size] RGB planes, each
// value in [0, 1]: bilinear samples of the image (pixel centers aligned, edges
// clamped, as OpenCV's INTER_LINEAR) divided by 255, and `pad_value` / 255
// outside it. Alpha is ignored. `image` must match the letterbox's source size.
cudaError_t launch_letterbox(const ImageView& image, const Letterbox& letterbox,
                             std::uint8_t pad_value, float* output, cudaStream_t stream);

}  // namespace yolo26_detection
