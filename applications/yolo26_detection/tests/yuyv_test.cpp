// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The webcam source's YUYV to RGB8 conversion, on known pixels. Needs no SDK
// or GPU, so it runs on hosted CI.

#include "detection/yuyv.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "test_support.hpp"

namespace yolo26_detection {
namespace {

using Rgb = std::array<int, 3>;

// One YUYV pair (Y0 U Y1 V) as its two RGB pixels.
std::array<Rgb, 2> convert_pair(std::uint8_t y0, std::uint8_t u, std::uint8_t y1, std::uint8_t v) {
  const std::array<std::uint8_t, 4> yuyv{y0, u, y1, v};
  std::array<std::uint8_t, 6> rgb{};
  yuyv_to_rgb(yuyv.data(), yuyv.size(), 2, 1, rgb.data());
  return {Rgb{rgb[0], rgb[1], rgb[2]}, Rgb{rgb[3], rgb[4], rgb[5]}};
}

void check_known_pixels() {
  // Video black and white, and luma beyond them clamped.
  VERIFY((convert_pair(16, 128, 235, 128) == std::array<Rgb, 2>{Rgb{0, 0, 0}, Rgb{255, 255, 255}}));
  VERIFY((convert_pair(0, 128, 255, 128) == std::array<Rgb, 2>{Rgb{0, 0, 0}, Rgb{255, 255, 255}}));
  // BT.601 primaries: both pixels of a pair share U and V.
  VERIFY((convert_pair(81, 90, 81, 240)[1] == Rgb{255, 0, 0}));
  VERIFY((convert_pair(145, 54, 145, 34)[0] == Rgb{0, 255, 1}));
  VERIFY((convert_pair(41, 240, 41, 110)[0] == Rgb{0, 0, 255}));
  // Mid gray.
  VERIFY((convert_pair(126, 128, 126, 128)[0] == Rgb{128, 128, 128}));
}

// Rows are read `pitch` bytes apart and written packed.
void check_pitch() {
  const std::vector<std::uint8_t> yuyv{16,  128, 16,  128, 0, 0, 0, 0,  // Row 0 and padding.
                                       235, 128, 235, 128, 0, 0, 0, 0};
  std::vector<std::uint8_t> rgb(2 * 2 * 3);
  yuyv_to_rgb(yuyv.data(), 8, 2, 2, rgb.data());
  VERIFY((rgb == std::vector<std::uint8_t>{0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255}));
}

void check_rejects_odd_width() {
  std::vector<std::uint8_t> buffer(64);
  VERIFY(testing::throws<std::invalid_argument>(
      [&] { yuyv_to_rgb(buffer.data(), 8, 3, 1, buffer.data()); }));
  VERIFY(testing::throws<std::invalid_argument>(
      [&] { yuyv_to_rgb(buffer.data(), 8, 0, 1, buffer.data()); }));
}

}  // namespace
}  // namespace yolo26_detection

int main() {
  return yolo26_detection::testing::run({
      {"known pixels", yolo26_detection::check_known_pixels},
      {"row pitch", yolo26_detection::check_pitch},
      {"odd widths rejected", yolo26_detection::check_rejects_odd_width},
  });
}
