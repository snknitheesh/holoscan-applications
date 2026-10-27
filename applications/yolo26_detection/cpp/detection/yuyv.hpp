// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// YUYV 4:2:2 (the usual webcam format) to packed RGB8. Header-only and free of
// any SDK, so its test runs anywhere.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace yolo26_detection {

// Converts `width` x `height` YUYV pixels, rows `pitch` bytes apart, to RGB8
// rows packed into `rgb`, with BT.601 limited-range integer coefficients.
// @throws std::invalid_argument if `width` is odd (two pixels share U and V).
inline void yuyv_to_rgb(const std::uint8_t* yuyv, std::size_t pitch, int width, int height,
                        std::uint8_t* rgb) {
  if (width <= 0 || height <= 0 || width % 2 != 0) {
    throw std::invalid_argument("YUYV images need a positive, even width");
  }
  const auto clamp = [](int value) { return static_cast<std::uint8_t>(std::clamp(value, 0, 255)); };
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* in = yuyv + static_cast<std::size_t>(y) * pitch;
    std::uint8_t* out = rgb + static_cast<std::size_t>(y) * width * 3;
    for (int x = 0; x < width; x += 2, in += 4) {
      const int d = in[1] - 128;
      const int e = in[3] - 128;
      for (const int luma : {in[0], in[2]}) {
        const int c = 298 * (luma - 16) + 128;
        *out++ = clamp((c + 409 * e) >> 8);
        *out++ = clamp((c - 100 * d - 208 * e) >> 8);
        *out++ = clamp((c + 516 * d) >> 8);
      }
    }
  }
}

}  // namespace yolo26_detection
