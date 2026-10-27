// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic frames with known detections for the stub model (stub_model.hpp):
// on a dark gray background, a red rectangle (class 0), a blue one (class 1),
// and a green and a white one that neither class matches. The rectangles move
// from frame to frame and stay inside the image. No Holoscan.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "detection/letterbox.hpp"

namespace yolo26_detection {

// The smallest and largest synthetic images.
inline constexpr int kMinSyntheticWidth = 64;
inline constexpr int kMinSyntheticHeight = 48;
inline constexpr int kMaxSyntheticSide = 4096;

// A detection in image pixels: corners (x2, y2 exclusive), score and class.
struct BoxDetection {
  float x1{};
  float y1{};
  float x2{};
  float y2{};
  float score{};
  std::int32_t class_id{};
};

// A filled rectangle [x0, x1) x [y0, y1) of one RGB color.
struct SceneRect {
  int x0{};
  int y0{};
  int x1{};
  int y1{};
  std::array<std::uint8_t, 3> rgb{};
};

struct SyntheticScene {
  int width{};
  int height{};
  // Red (class 0), blue (class 1), then the green and white distractors.
  std::array<SceneRect, 4> rects{};
};

// Frame `frame` of a width x height scene.
// @throws std::invalid_argument for a size outside the synthetic range.
[[nodiscard]] SyntheticScene make_scene(int width, int height, std::uint32_t frame);

// The stub model's detections of `scene`: the red and the blue rectangle,
// class 0 and 1, with their stub scores, in that order.
[[nodiscard]] std::vector<BoxDetection> scene_truth(const SyntheticScene& scene);

// Draws `scene` into `pixels`, rows of `pitch` bytes in `format`; alpha is 255.
void render_scene(const SyntheticScene& scene, PixelFormat format, std::uint8_t* pixels,
                  std::size_t pitch);

}  // namespace yolo26_detection
