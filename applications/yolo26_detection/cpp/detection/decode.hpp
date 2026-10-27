// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Decodes the output of an end-to-end YOLO26 model on the GPU. The model
// emits a fixed number of rows, [x1, y1, x2, y2, score, class], with corners
// in letterboxed input pixels and no suppression left to do. The decode keeps
// the rows that pass the score threshold, maps their corners back to the
// image and compacts them, in model order, into fixed-capacity outputs. No
// Holoscan.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "detection/letterbox.hpp"

namespace yolo26_detection {

// Floats per model output row.
inline constexpr int kRowFields = 6;
// Largest number of model rows and of kept detections.
inline constexpr int kMaxRows = 1 << 20;

struct DecodeParams {
  // Model rows to read.
  int rows{};
  // A row is kept when its score is finite and at least this.
  float score_threshold{};
  // Class values must round to 0..num_classes - 1.
  int num_classes{};
  // Detections the outputs hold; rows kept beyond it are dropped, the
  // lowest-ranked first.
  int capacity{};
  // The input's letterbox, to map corners back to the image.
  Letterbox letterbox;
};

// GPU outputs: `boxes` holds capacity x [x1, y1, x2, y2] in image pixels,
// `scores` and `class_ids` capacity values, and `count` one int32, the
// detections written. Slots from `count` on are zero. `passed`, if given,
// gets the number of rows kept before the capacity cut, so a caller can tell
// that some were dropped; it may be mapped host memory.
struct DecodeOutputs {
  float* boxes{};
  float* scores{};
  std::int32_t* class_ids{};
  std::int32_t* count{};
  std::int32_t* passed{};
};

// Decodes `rows` (params.rows x kRowFields floats in GPU memory) on `stream`.
// A row is kept when its score passes the threshold, its class value is a
// valid class, and its box, mapped back to the image as
// ((x - pad_left) / scale, (y - pad_top) / scale) and clamped to
// [0, width] x [0, height], still has a positive width and height. Kept rows
// keep their order.
cudaError_t launch_decode(const float* rows, const DecodeParams& params,
                          const DecodeOutputs& outputs, cudaStream_t stream);

}  // namespace yolo26_detection
