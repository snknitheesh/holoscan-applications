// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "depth_truth.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace point_cloud_to_depth {
namespace {

// Where a point's depth falls relative to the valid range.
enum class Range : std::uint8_t {
  kOut,   // Dropped.
  kIn,    // Kept.
  kEdge,  // So close to depth_min or depth_max that it may go either way.
};

// A point the depth image may show at a pixel.
struct Candidate {
  float z{};
  std::uint8_t rgb[3]{};
  Range range{Range::kOut};
};

class SceneChecker {
 public:
  SceneChecker(const SyntheticFrame& frame, const SceneTruth& truth, const void* depth)
      : frame_(frame),
        truth_(truth),
        depth_f32_(static_cast<const float*>(depth)),
        depth_u16_(static_cast<const std::uint16_t*>(depth)),
        scene_(static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height)) {
    for (int v = 0; v < frame.height; ++v) {
      for (int u = 0; u < frame.width; ++u) {
        scene_[index(u, v)] = synthetic_scene_depth(frame, truth.millimeter_depth, u, v);
      }
    }
  }

  // The points of pixel (u, v): its surface point and its duplicate. Returns
  // how many it has.
  int candidates(int u, int v, Candidate out[2]) const {
    const float z = scene_[index(u, v)];
    if (!(z > 0.0F)) {
      return 0;  // A hole: its point is not finite.
    }
    out[0].z = z;
    synthetic_color(frame_, u, v, out[0].rgb);
    out[0].range = classify(z);
    if (!has_duplicate(static_cast<std::uint32_t>(index(u, v)))) {
      return 1;
    }
    out[1] = out[0];
    out[1].z = z * kDuplicateDepthFactor;
    duplicate_color(out[1].rgb);
    out[1].range = classify(out[1].z);
    return 2;
  }

  [[nodiscard]] bool filled(std::size_t pixel) const {
    return truth_.format == DepthFormat::kUint16 ? depth_u16_[pixel] != 0U
                                                 : depth_f32_[pixel] != 0.0F;
  }

  // Whether the image's depth at `pixel` is that of a point at depth `z`.
  [[nodiscard]] bool matches(std::size_t pixel, float z) const {
    if (truth_.format == DepthFormat::kUint16) {
      return depth_u16_[pixel] == std::lround(z / truth_.depth_scale);
    }
    return std::fabs(static_cast<double>(depth_f32_[pixel]) - z) <= max_depth_error(z);
  }

  [[nodiscard]] double error_m(std::size_t pixel, float z) const {
    return truth_.format == DepthFormat::kUint16
               ? 0.0
               : std::fabs(static_cast<double>(depth_f32_[pixel]) - z);
  }

  [[nodiscard]] std::size_t index(int u, int v) const {
    return static_cast<std::size_t>(v) * static_cast<std::size_t>(frame_.width) +
           static_cast<std::size_t>(u);
  }

 private:
  [[nodiscard]] Range classify(float z) const {
    if (truth_.format == DepthFormat::kUint16) {
      const float counts = z / truth_.depth_scale;
      if (!(counts >= 0.5F && counts < static_cast<float>(kMaxDepthCount) + 0.5F)) {
        return Range::kOut;
      }
    }
    if (std::fabs(static_cast<double>(z) - truth_.depth_min) <= max_depth_error(z) ||
        std::fabs(static_cast<double>(z) - truth_.depth_max) <= max_depth_error(z)) {
      return Range::kEdge;
    }
    return z >= truth_.depth_min && z <= truth_.depth_max ? Range::kIn : Range::kOut;
  }

  const SyntheticFrame& frame_;
  const SceneTruth& truth_;
  const float* depth_f32_;
  const std::uint16_t* depth_u16_;
  std::vector<float> scene_;
};

}  // namespace

TruthCheck check_against_scene(const SyntheticFrame& frame, const SceneTruth& truth,
                               const void* depth, const std::uint8_t* rgb) {
  const SceneChecker checker(frame, truth, depth);
  const int radius = truth.point_size / 2;
  TruthCheck result;
  std::vector<Candidate> window;
  for (int v = 0; v < frame.height; ++v) {
    for (int u = 0; u < frame.width; ++u) {
      // Every point whose square covers (u, v).
      window.clear();
      for (int row = std::max(v - radius, 0); row <= std::min(v + radius, frame.height - 1);
           ++row) {
        for (int col = std::max(u - radius, 0); col <= std::min(u + radius, frame.width - 1);
             ++col) {
          Candidate found[2];
          const int count = checker.candidates(col, row, found);
          window.insert(window.end(), found, found + count);
        }
      }
      float nearest = std::numeric_limits<float>::infinity();
      for (const Candidate& candidate : window) {
        if (candidate.range == Range::kIn) {
          nearest = std::min(nearest, candidate.z);
        }
      }
      const bool must_fill = std::isfinite(nearest);
      const std::size_t pixel = checker.index(u, v);
      const bool filled = checker.filled(pixel);
      bool depth_right = false;
      if (!filled) {
        depth_right = !must_fill;
      } else if (must_fill && checker.matches(pixel, nearest)) {
        depth_right = true;
        result.max_error_m = std::max(result.max_error_m, checker.error_m(pixel, nearest));
      } else {
        // A nearer point at the edge of the range may have been kept.
        depth_right = std::any_of(window.begin(), window.end(), [&](const Candidate& candidate) {
          return candidate.range == Range::kEdge && candidate.z < nearest &&
                 checker.matches(pixel, candidate.z);
        });
      }
      if (!depth_right) {
        ++result.wrong_depths;
        continue;
      }
      if (rgb == nullptr) {
        continue;
      }
      const std::uint8_t* got = rgb + 3 * pixel;
      const bool color_right = filled
                                   ? std::any_of(window.begin(), window.end(),
                                                 [&](const Candidate& candidate) {
                                                   return candidate.range != Range::kOut &&
                                                          checker.matches(pixel, candidate.z) &&
                                                          std::equal(got, got + 3, candidate.rgb);
                                                 })
                                   : got[0] == 0U && got[1] == 0U && got[2] == 0U;
      if (!color_right) {
        ++result.wrong_colors;
      }
    }
  }
  return result;
}

}  // namespace point_cloud_to_depth
