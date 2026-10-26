// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// What a depth image made from a synthetic cloud should show, derived from
// the synthetic camera's depth image rather than from the cloud: each pixel
// holds the nearest depth among the pixels its points' squares cover, from the
// surface points and their duplicates, within the valid range.

#pragma once

#include <cstdint>

#include "project.hpp"
#include "synthetic_scene.hpp"

namespace point_cloud_to_depth {

// The operator settings a checked depth image was made with.
struct SceneTruth {
  // The scene's depths are whole millimeters (see synthetic_scene_depth()).
  // Required for uint16 depth, which then must reproduce them exactly.
  bool millimeter_depth{};
  DepthFormat format{DepthFormat::kFloat32};
  float depth_scale{0.001F};
  float depth_min{};
  float depth_max{};
  int point_size{1};
};

// Largest difference, in meters, a float32 depth `z` may have from the truth:
// 1 um plus 1 um per meter. The cloud's trip into the body frame and back
// moves points by a few float32 roundings, about 0.1 um per meter.
[[nodiscard]] constexpr double max_depth_error(double z) { return 1e-6 + 1e-6 * z; }

struct TruthCheck {
  // Pixels empty where a point belongs, filled where none does, or with
  // another depth than the nearest point's.
  std::uint64_t wrong_depths{};
  // Filled pixels whose color is not that of a point at their depth.
  std::uint64_t wrong_colors{};
  // Largest difference of a float32 depth from the truth, in meters.
  double max_error_m{};
};

// Checks a frame.width x frame.height depth image (float32 meters or uint16,
// as truth.format says, 0 where empty) and, when `rgb` is given, its RGB8
// color image against frame `frame` of the synthetic camera. A point within
// max_depth_error() of depth_min or depth_max may land on either side of it, so
// either outcome passes there.
[[nodiscard]] TruthCheck check_against_scene(const SyntheticFrame& frame, const SceneTruth& truth,
                                             const void* depth, const std::uint8_t* rgb);

}  // namespace point_cloud_to_depth
