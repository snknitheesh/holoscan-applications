// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Checks detections against what they should be: the exact detections of a
// synthetic frame, or expected detections listed per image file; and reads
// the files that hold expectations and class names. No Holoscan.

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "detection/synthetic_scene.hpp"

namespace yolo26_detection {

// Intersection over union of two boxes; 0 when either has no area.
[[nodiscard]] float iou(const BoxDetection& a, const BoxDetection& b);

// The largest distance between corresponding edges of two boxes.
[[nodiscard]] float edge_error(const BoxDetection& a, const BoxDetection& b);

struct TruthMatch {
  // Truth detections no detection matched.
  std::uint32_t missed{};
  // Detections that matched no truth detection.
  std::uint32_t extra{};
  // Over the truth detections, the smallest edge error of a detection of the
  // same class: how far off the nearest candidate was, matched or not.
  float max_edge_error{};
};

// Pairs each truth detection with a distinct detection of its class whose
// edges are all within `tolerance` and whose score is within
// `score_tolerance` of the truth's.
[[nodiscard]] TruthMatch match_truth(std::span<const BoxDetection> detections,
                                     std::span<const BoxDetection> truth, float tolerance,
                                     float score_tolerance);

// A detection an image file must yield: its class, and a box that a
// detection of that class must overlap by an IoU of at least the threshold.
struct Expectation {
  std::string image;
  BoxDetection box;
};

// Reads lines of `image,class_id,x1,y1,x2,y2`, where `image` is a file name
// without its directory. Blank lines and lines starting with # are skipped.
// @throws std::runtime_error naming the file and line of a malformed entry.
[[nodiscard]] std::vector<Expectation> read_expectations(const std::filesystem::path& path);

// How many of `expected` a distinct detection of the same class overlaps by
// an IoU of at least `min_iou`.
[[nodiscard]] std::uint32_t count_matched(std::span<const BoxDetection> detections,
                                          std::span<const Expectation> expected, float min_iou);

// One class name per line; a trailing empty line is dropped.
// @throws std::runtime_error if the file cannot be read or holds no name.
[[nodiscard]] std::vector<std::string> read_labels(const std::filesystem::path& path);

}  // namespace yolo26_detection
