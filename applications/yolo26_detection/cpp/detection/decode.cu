// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include <cub/block/block_scan.cuh>  // NOLINT(build/include_order)

#include "decode.hpp"

namespace yolo26_detection {
namespace {

constexpr int kThreads = 256;

struct Box {
  float x1;
  float y1;
  float x2;
  float y2;
};

// Whether row `row` is kept, and if so its box in image pixels and its class.
__device__ bool decode_row(const float* row, const DecodeParams& p, Box& box, int& class_id) {
  const float score = row[4];
  if (!(score >= p.score_threshold) || !isfinite(score)) {
    return false;
  }
  const float value = row[5];
  if (!isfinite(value)) {
    return false;
  }
  const float rounded = rintf(value);
  if (rounded < 0.0f || rounded >= static_cast<float>(p.num_classes)) {
    return false;
  }
  class_id = static_cast<int>(rounded);
  const Letterbox& lb = p.letterbox;
  const float width = static_cast<float>(lb.source_width);
  const float height = static_cast<float>(lb.source_height);
  const float left = static_cast<float>(lb.pad_left);
  const float top = static_cast<float>(lb.pad_top);
  // fminf/fmaxf drop a NaN operand, so a NaN corner clamps to the edge it
  // would leave first; the area check below then drops the box.
  box.x1 = fminf(fmaxf((row[0] - left) / lb.scale, 0.0f), width);
  box.y1 = fminf(fmaxf((row[1] - top) / lb.scale, 0.0f), height);
  box.x2 = fminf(fmaxf((row[2] - left) / lb.scale, 0.0f), width);
  box.y2 = fminf(fmaxf((row[3] - top) / lb.scale, 0.0f), height);
  return box.x2 > box.x1 && box.y2 > box.y1 && isfinite(row[0]) && isfinite(row[1]) &&
         isfinite(row[2]) && isfinite(row[3]);
}

// One block walks the rows in chunks, compacting the kept ones with a block
// scan so they keep their order, then zeroes the unused slots. Once the
// outputs are full it stops, unless it must count the rows kept beyond them.
__global__ void decode_kernel(const float* rows, DecodeParams p, DecodeOutputs out) {
  using Scan = cub::BlockScan<int, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int written;
  __shared__ int passed;
  if (threadIdx.x == 0) {
    written = 0;
    passed = 0;
  }
  __syncthreads();
  for (int base = 0; base < p.rows; base += kThreads) {
    const int r = base + static_cast<int>(threadIdx.x);
    Box box{};
    int class_id = 0;
    const bool keep = r < p.rows && decode_row(rows + static_cast<std::int64_t>(r) * kRowFields, p,
                                               box, class_id);
    int slot = 0;
    int kept = 0;
    Scan(scan_storage).ExclusiveSum(keep ? 1 : 0, slot, kept);
    slot += written;
    if (keep && slot < p.capacity) {
      out.boxes[4 * slot + 0] = box.x1;
      out.boxes[4 * slot + 1] = box.y1;
      out.boxes[4 * slot + 2] = box.x2;
      out.boxes[4 * slot + 3] = box.y2;
      out.scores[slot] = rows[static_cast<std::int64_t>(r) * kRowFields + 4];
      out.class_ids[slot] = class_id;
    }
    __syncthreads();  // Every thread has read `written`, and the scan storage is free.
    if (threadIdx.x == 0) {
      written = min(written + kept, p.capacity);
      passed += kept;
    }
    __syncthreads();
    if (written >= p.capacity && out.passed == nullptr) {
      break;
    }
  }
  for (int slot = written + static_cast<int>(threadIdx.x); slot < p.capacity; slot += kThreads) {
    out.boxes[4 * slot + 0] = 0.0f;
    out.boxes[4 * slot + 1] = 0.0f;
    out.boxes[4 * slot + 2] = 0.0f;
    out.boxes[4 * slot + 3] = 0.0f;
    out.scores[slot] = 0.0f;
    out.class_ids[slot] = 0;
  }
  if (threadIdx.x == 0) {
    *out.count = written;
    if (out.passed != nullptr) {
      *out.passed = passed;
    }
  }
}

}  // namespace

cudaError_t launch_decode(const float* rows, const DecodeParams& params,
                          const DecodeOutputs& outputs, cudaStream_t stream) {
  if ((rows == nullptr && params.rows != 0) || params.rows < 0 || params.rows > kMaxRows ||
      params.capacity < 1 || params.capacity > kMaxRows || params.num_classes < 1 ||
      !(params.letterbox.scale > 0.0f) || outputs.boxes == nullptr || outputs.scores == nullptr ||
      outputs.class_ids == nullptr || outputs.count == nullptr) {
    return cudaErrorInvalidValue;
  }
  decode_kernel<<<1, kThreads, 0, stream>>>(rows, params, outputs);
  return cudaGetLastError();
}

}  // namespace yolo26_detection
