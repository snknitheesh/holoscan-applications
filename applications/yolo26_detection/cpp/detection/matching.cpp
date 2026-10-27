// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "matching.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace yolo26_detection {
namespace {

std::string_view trim(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r");
  return text.substr(first, last - first + 1);
}

template <typename T>
bool parse(std::string_view text, T& value) {
  text = trim(text);
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size() && !text.empty();
}

}  // namespace

float iou(const BoxDetection& a, const BoxDetection& b) {
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  if (!(area_a > 0.0F) || !(area_b > 0.0F)) {
    return 0.0F;
  }
  const float w = std::max(0.0F, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float h = std::max(0.0F, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float overlap = w * h;
  return overlap / (area_a + area_b - overlap);
}

float edge_error(const BoxDetection& a, const BoxDetection& b) {
  return std::max({std::fabs(a.x1 - b.x1), std::fabs(a.y1 - b.y1), std::fabs(a.x2 - b.x2),
                   std::fabs(a.y2 - b.y2)});
}

TruthMatch match_truth(std::span<const BoxDetection> detections,
                       std::span<const BoxDetection> truth, float tolerance,
                       float score_tolerance) {
  TruthMatch result;
  std::vector<bool> used(detections.size(), false);
  for (const BoxDetection& t : truth) {
    float nearest = std::numeric_limits<float>::infinity();
    std::size_t match = detections.size();
    for (std::size_t i = 0; i < detections.size(); ++i) {
      const BoxDetection& d = detections[i];
      if (d.class_id != t.class_id) {
        continue;
      }
      const float error = edge_error(d, t);
      nearest = std::min(nearest, error);
      if (!used[i] && match == detections.size() && error <= tolerance &&
          std::fabs(d.score - t.score) <= score_tolerance) {
        match = i;
      }
    }
    result.max_edge_error = std::max(result.max_edge_error, nearest);
    if (match == detections.size()) {
      ++result.missed;
    } else {
      used[match] = true;
    }
  }
  result.extra = static_cast<std::uint32_t>(std::count(used.begin(), used.end(), false));
  return result;
}

std::vector<Expectation> read_expectations(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(path.string() + ": cannot open");
  }
  std::vector<Expectation> result;
  std::string line;
  for (int number = 1; std::getline(in, line); ++number) {
    const std::string_view text = trim(line);
    if (text.empty() || text.front() == '#') {
      continue;
    }
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (true) {
      const std::size_t comma = text.find(',', start);
      fields.push_back(text.substr(start, comma == std::string_view::npos ? comma : comma - start));
      if (comma == std::string_view::npos) {
        break;
      }
      start = comma + 1;
    }
    Expectation e;
    const bool ok = fields.size() == 6 && !trim(fields[0]).empty() &&
                    parse(fields[1], e.box.class_id) && parse(fields[2], e.box.x1) &&
                    parse(fields[3], e.box.y1) && parse(fields[4], e.box.x2) &&
                    parse(fields[5], e.box.y2) && e.box.class_id >= 0 && e.box.x2 > e.box.x1 &&
                    e.box.y2 > e.box.y1;
    if (!ok) {
      throw std::runtime_error(path.string() + ":" + std::to_string(number) +
                               ": expected image,class_id,x1,y1,x2,y2 with x2 > x1, y2 > y1");
    }
    e.image = std::string(trim(fields[0]));
    result.push_back(std::move(e));
  }
  return result;
}

std::uint32_t count_matched(std::span<const BoxDetection> detections,
                            std::span<const Expectation> expected, float min_iou) {
  std::vector<bool> used(detections.size(), false);
  std::uint32_t matched = 0;
  for (const Expectation& e : expected) {
    // The best-overlapping unused detection of the class.
    std::size_t best = detections.size();
    float best_iou = min_iou;
    for (std::size_t i = 0; i < detections.size(); ++i) {
      if (used[i] || detections[i].class_id != e.box.class_id) {
        continue;
      }
      const float overlap = iou(detections[i], e.box);
      if (overlap >= best_iou) {
        best = i;
        best_iou = overlap;
      }
    }
    if (best != detections.size()) {
      used[best] = true;
      ++matched;
    }
  }
  return matched;
}

std::vector<std::string> read_labels(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(path.string() + ": cannot open");
  }
  std::vector<std::string> labels;
  for (std::string line; std::getline(in, line);) {
    labels.emplace_back(trim(line));
  }
  while (!labels.empty() && labels.back().empty()) {
    labels.pop_back();
  }
  if (labels.empty()) {
    throw std::runtime_error(path.string() + ": no class names");
  }
  return labels;
}

}  // namespace yolo26_detection
