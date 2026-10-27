// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The decode kernel against a host decode of the same rows: the score
// threshold and its boundary, invalid and non-finite classes, scores and
// corners, boxes clamped to the image and dropped when nothing is left,
// order, capacity, the count and the zeroed slots after it, the rows kept
// before the capacity cut, with and without counting them, inputs of several
// block-sized chunks, and boxes mapped back through letterboxes with
// padding on either axis.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "detection/decode.hpp"
#include "test_support.hpp"

namespace yolo26_detection {
namespace {

struct Decoded {
  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<std::int32_t> class_ids;
  std::int32_t count{-1};
  // Rows kept before the capacity cut; -1 when not asked for.
  std::int32_t passed{-1};
};

// Decodes on the GPU; `count_passed` asks the kernel for Decoded::passed.
Decoded run_kernel(const std::vector<float>& rows, const DecodeParams& params,
                   bool count_passed = true) {
  float* d_rows = nullptr;
  float* d_boxes = nullptr;
  float* d_scores = nullptr;
  std::int32_t* d_classes = nullptr;
  std::int32_t* d_count = nullptr;
  std::int32_t* d_passed = nullptr;
  const std::size_t capacity = static_cast<std::size_t>(params.capacity);
  if (cudaMalloc(&d_rows, std::max<std::size_t>(rows.size(), 1) * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&d_boxes, capacity * 4 * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&d_scores, capacity * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&d_classes, capacity * sizeof(std::int32_t)) != cudaSuccess ||
      cudaMalloc(&d_count, sizeof(std::int32_t)) != cudaSuccess ||
      cudaMalloc(&d_passed, sizeof(std::int32_t)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }
  cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(float), cudaMemcpyHostToDevice);
  // Garbage in the outputs: the kernel must write every slot.
  cudaMemset(d_boxes, 0x7F, capacity * 4 * sizeof(float));
  cudaMemset(d_scores, 0x7F, capacity * sizeof(float));
  cudaMemset(d_classes, 0x7F, capacity * sizeof(std::int32_t));
  cudaMemset(d_count, 0x7F, sizeof(std::int32_t));
  cudaMemset(d_passed, 0x7F, sizeof(std::int32_t));
  const cudaError_t status = launch_decode(
      d_rows, params,
      DecodeOutputs{d_boxes, d_scores, d_classes, d_count, count_passed ? d_passed : nullptr},
      nullptr);
  Decoded out;
  out.boxes.resize(capacity * 4);
  out.scores.resize(capacity);
  out.class_ids.resize(capacity);
  cudaMemcpy(out.boxes.data(), d_boxes, out.boxes.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(out.scores.data(), d_scores, out.scores.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out.class_ids.data(), d_classes, out.class_ids.size() * sizeof(std::int32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&out.count, d_count, sizeof(std::int32_t), cudaMemcpyDeviceToHost);
  if (count_passed) {
    cudaMemcpy(&out.passed, d_passed, sizeof(std::int32_t), cudaMemcpyDeviceToHost);
  }
  const cudaError_t synced = cudaDeviceSynchronize();
  for (void* p :
       {static_cast<void*>(d_rows), static_cast<void*>(d_boxes), static_cast<void*>(d_scores),
        static_cast<void*>(d_classes), static_cast<void*>(d_count), static_cast<void*>(d_passed)}) {
    cudaFree(p);
  }
  if (status != cudaSuccess || synced != cudaSuccess) {
    throw std::runtime_error("decode kernel failed");
  }
  return out;
}

// The decode as its header describes it, on the host.
Decoded reference(const std::vector<float>& rows, const DecodeParams& p) {
  Decoded out;
  const std::size_t capacity = static_cast<std::size_t>(p.capacity);
  out.boxes.assign(capacity * 4, 0.0F);
  out.scores.assign(capacity, 0.0F);
  out.class_ids.assign(capacity, 0);
  std::size_t kept = 0;
  std::int32_t passed = 0;
  const Letterbox& lb = p.letterbox;
  const auto map = [&](float v, int pad, int limit) {
    return std::fmin(std::fmax((v - static_cast<float>(pad)) / lb.scale, 0.0F),
                     static_cast<float>(limit));
  };
  for (int r = 0; r < p.rows; ++r) {
    const float* row = rows.data() + static_cast<std::size_t>(r) * kRowFields;
    if (!std::isfinite(row[4]) || !(row[4] >= p.score_threshold) || !std::isfinite(row[5])) {
      continue;
    }
    const float c = std::nearbyint(row[5]);
    if (c < 0.0F || c >= static_cast<float>(p.num_classes)) {
      continue;
    }
    if (!std::isfinite(row[0]) || !std::isfinite(row[1]) || !std::isfinite(row[2]) ||
        !std::isfinite(row[3])) {
      continue;
    }
    const float x1 = map(row[0], lb.pad_left, lb.source_width);
    const float y1 = map(row[1], lb.pad_top, lb.source_height);
    const float x2 = map(row[2], lb.pad_left, lb.source_width);
    const float y2 = map(row[3], lb.pad_top, lb.source_height);
    if (!(x2 > x1 && y2 > y1)) {
      continue;
    }
    ++passed;
    if (kept == capacity) {
      continue;
    }
    out.boxes[4 * kept] = x1;
    out.boxes[4 * kept + 1] = y1;
    out.boxes[4 * kept + 2] = x2;
    out.boxes[4 * kept + 3] = y2;
    out.scores[kept] = row[4];
    out.class_ids[kept] = static_cast<std::int32_t>(c);
    ++kept;
  }
  out.count = static_cast<std::int32_t>(kept);
  out.passed = passed;
  return out;
}

// Rows in input pixels with a share of every special case.
std::vector<float> random_rows(int count, int size, std::uint32_t seed) {
  std::mt19937 random(seed);
  std::uniform_real_distribution<float> coordinate(-40.0F, static_cast<float>(size) + 40.0F);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInfinity = std::numeric_limits<float>::infinity();
  std::vector<float> rows;
  for (int i = 0; i < count; ++i) {
    float x1 = coordinate(random);
    float y1 = coordinate(random);
    float x2 = x1 + unit(random) * 200.0F;
    float y2 = y1 + unit(random) * 200.0F;
    float score = unit(random);
    float cls = static_cast<float>(random() % 90U) - 5.0F;  // -5..84, some invalid.
    switch (random() % 16U) {
      case 0:
        score = kNan;
        break;
      case 1:
        score = kInfinity;
        break;
      case 2:
        cls = kNan;
        break;
      case 3:
        cls += 0.4F;
        break;  // Rounds to the class.
      case 4:
        x1 = kNan;
        break;
      case 5:
        y2 = -kInfinity;
        break;
      case 6:
        x2 = x1;
        break;  // No width.
      case 7:
        y1 = y2 + 1.0F;
        break;  // Inverted.
      case 8:
        x1 = -500.0F;
        x2 = -100.0F;
        break;  // Left of the image.
      default:
        break;
    }
    rows.insert(rows.end(), {x1, y1, x2, y2, score, cls});
  }
  return rows;
}

bool same(float a, float b) { return a == b || (std::isnan(a) && std::isnan(b)); }

void check_equal(const Decoded& gpu, const Decoded& cpu, const std::string& what) {
  VERIFY_MSG(gpu.count == cpu.count,
             what + ": count " + std::to_string(gpu.count) + " vs " + std::to_string(cpu.count));
  VERIFY_MSG(gpu.passed == cpu.passed,
             what + ": passed " + std::to_string(gpu.passed) + " vs " + std::to_string(cpu.passed));
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < cpu.boxes.size(); ++i) wrong += !same(gpu.boxes[i], cpu.boxes[i]);
  for (std::size_t i = 0; i < cpu.scores.size(); ++i) {
    wrong += !same(gpu.scores[i], cpu.scores[i]) || gpu.class_ids[i] != cpu.class_ids[i];
  }
  VERIFY_MSG(wrong == 0, what + ": " + std::to_string(wrong) + " values differ");
}

void check_random(int rows, int capacity, int width, int height, float threshold) {
  const Letterbox lb = make_letterbox(width, height, 640);
  const DecodeParams params{.rows = rows,
                            .score_threshold = threshold,
                            .num_classes = 80,
                            .capacity = capacity,
                            .letterbox = lb};
  const auto input = random_rows(rows, 640, static_cast<std::uint32_t>(rows * 7 + capacity));
  const Decoded cpu = reference(input, params);
  const std::string what = std::to_string(rows) + " rows into " + std::to_string(capacity);
  check_equal(run_kernel(input, params), cpu, what);
  // Without counting the passed rows, the kernel may stop at the capacity;
  // the outputs are the same.
  Decoded uncounted = run_kernel(input, params, /*count_passed=*/false);
  uncounted.passed = cpu.passed;
  check_equal(uncounted, cpu, what + " without counting");
  VERIFY_MSG(cpu.count > 0, "the case keeps some rows");
}

// A score equal to the threshold is kept; the next float below it is not.
void check_threshold_boundary() {
  const float threshold = 0.25F;
  const std::vector<float> rows{10,   10, 50, 50, threshold,
                                1,  //
                                10,   10, 50, 50, std::nextafter(threshold, 0.0F),
                                2,    10, 10, 50, 50,
                                1.0F, 3};
  const DecodeParams params{.rows = 3,
                            .score_threshold = threshold,
                            .num_classes = 80,
                            .capacity = 5,
                            .letterbox = make_letterbox(640, 640, 640)};
  const Decoded out = run_kernel(rows, params);
  VERIFY(out.count == 2 && out.class_ids[0] == 1 && out.class_ids[1] == 3);
  VERIFY(out.scores[0] == threshold && out.scores[2] == 0.0F);
}

// Kept rows beyond the capacity are dropped, the last ones first, and counted.
void check_capacity() {
  std::vector<float> rows;
  for (int i = 0; i < 10; ++i) {
    rows.insert(rows.end(),
                {10.0F * i, 0, 10.0F * i + 5, 5, 0.9F - 0.01F * i, static_cast<float>(i)});
  }
  const DecodeParams params{.rows = 10,
                            .score_threshold = 0.25F,
                            .num_classes = 80,
                            .capacity = 3,
                            .letterbox = make_letterbox(640, 640, 640)};
  const Decoded out = run_kernel(rows, params);
  VERIFY(out.count == 3 && out.class_ids[0] == 0 && out.class_ids[1] == 1 && out.class_ids[2] == 2);
  VERIFY(out.passed == 10);
}

void check_empty() {
  const DecodeParams params{.rows = 0,
                            .score_threshold = 0.25F,
                            .num_classes = 80,
                            .capacity = 4,
                            .letterbox = make_letterbox(640, 480, 640)};
  const Decoded out = run_kernel({}, params);
  VERIFY(out.count == 0);
  for (const float v : out.boxes) VERIFY(v == 0.0F);
  for (const std::int32_t v : out.class_ids) VERIFY(v == 0);
}

// Boxes in letterboxes with odd padding above or to the left come back to
// the image: x = (u - pad_left) / scale, y = (v - pad_top) / scale.
void check_mapping() {
  const Letterbox wide = make_letterbox(1000, 333, 640);  // Scale 0.64, 213 rows above.
  const Letterbox tall = make_letterbox(333, 1000, 640);  // 213 columns to the left.
  const std::vector<float> rows_wide{64.0F, 213.0F + 64.0F, 640.0F, 213.0F + 128.0F, 0.9F, 1.0F};
  const std::vector<float> rows_tall{213.0F + 64.0F, 64.0F, 213.0F + 128.0F, 640.0F, 0.9F, 1.0F};
  for (const auto& [lb, rows, expected] :
       {std::tuple{wide, rows_wide, std::array<float, 4>{100, 100, 1000, 200}},
        std::tuple{tall, rows_tall, std::array<float, 4>{100, 100, 200, 1000}}}) {
    const Decoded out = run_kernel(
        rows,
        DecodeParams{
            .rows = 1, .score_threshold = 0.25F, .num_classes = 2, .capacity = 1, .letterbox = lb});
    VERIFY(out.count == 1);
    for (int k = 0; k < 4; ++k) {
      VERIFY_MSG(std::fabs(out.boxes[k] - expected[k]) < 1e-3F,
                 "edge " + std::to_string(k) + " = " + std::to_string(out.boxes[k]));
    }
  }
}

void check_rejects_bad_arguments() {
  float* f = nullptr;
  std::int32_t* i = nullptr;
  cudaMalloc(&f, 64);
  cudaMalloc(&i, 64);
  const DecodeOutputs outputs{f, f, i, i};
  DecodeParams params{.rows = 1,
                      .score_threshold = 0.5F,
                      .num_classes = 1,
                      .capacity = 1,
                      .letterbox = make_letterbox(32, 32, 32)};
  VERIFY(launch_decode(f, params, outputs, nullptr) == cudaSuccess);
  VERIFY(launch_decode(nullptr, params, outputs, nullptr) == cudaErrorInvalidValue);
  params.capacity = 0;
  VERIFY(launch_decode(f, params, outputs, nullptr) == cudaErrorInvalidValue);
  params.capacity = 1;
  params.num_classes = 0;
  VERIFY(launch_decode(f, params, outputs, nullptr) == cudaErrorInvalidValue);
  params.num_classes = 1;
  params.letterbox.scale = 0.0F;
  VERIFY(launch_decode(f, params, outputs, nullptr) == cudaErrorInvalidValue);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  cudaFree(f);
  cudaFree(i);
}

}  // namespace
}  // namespace yolo26_detection

int main() {
  namespace yd = yolo26_detection;
  return yd::testing::run({
      {"300 rows, wide image", [] { yd::check_random(300, 300, 1280, 720, 0.25F); }},
      {"300 rows, tall image, small capacity", [] { yd::check_random(300, 20, 333, 1000, 0.1F); }},
      {"1000 rows in chunks", [] { yd::check_random(1000, 600, 1000, 333, 0.05F); }},
      {"5000 rows, capacity reached early", [] { yd::check_random(5000, 100, 640, 640, 0.0F); }},
      {"threshold boundary", yd::check_threshold_boundary},
      {"capacity", yd::check_capacity},
      {"no rows", yd::check_empty},
      {"letterbox mapping", yd::check_mapping},
      {"rejects bad arguments", yd::check_rejects_bad_arguments},
  });
}
