// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A stand-in for a YOLO26 model, for tests and the `ci` mode, which need no
// downloaded weights. It is a small ONNX network with the tensors of an
// end-to-end YOLO26 export, `images` float32 [1, 3, S, S] in and `output0`
// float32 [1, kStubRows, 6] out, and it does detect, from the pixels:
//
// - class 0: the pixels whose red exceeds their green by more than half the
//   range, that is red - green > 0.5 in the normalized input;
// - class 1: likewise blue over green.
//
// Each class gives one row: the bounding box of its pixels in input pixels
// (x2, y2 exclusive) and the score kStubScores[class], or a score of 0 when
// no pixel qualifies. Gray padding never does. The remaining rows are fixed,
// with scores below kStubMaxFillerScore, as a model's low-confidence tail.
// A synthetic image with one red and one blue rectangle on a gray background
// therefore yields their two boxes, through the letterbox, TensorRT and the
// decode, and swapped red and blue channels swap their classes. No Holoscan.

#pragma once

#include <filesystem>
#include <string>

namespace yolo26_detection {

inline constexpr int kStubRows = 300;
inline constexpr int kStubClasses = 2;
inline constexpr float kStubScores[kStubClasses] = {0.9F, 0.8F};
inline constexpr float kStubMaxFillerScore = 0.2F;

// The serialized ONNX model (opset 13) for a size x size input.
// @throws std::invalid_argument for a size outside the letterbox's range.
[[nodiscard]] std::string stub_model_onnx(int size);

// Writes the model for `size` to `directory`/yolo26_stub_<size>.onnx unless
// that file already holds it, and returns its path. The file is replaced
// atomically, so concurrent runs can share the directory, and a TensorRT
// engine cached beside it stays valid as long as the file does not change.
// @throws std::runtime_error if the file cannot be written.
std::filesystem::path write_stub_model(const std::filesystem::path& directory, int size);

}  // namespace yolo26_detection
