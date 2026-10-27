// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DetectionOverlayOp: turns one image's detections from Yolo26DetectionOp
// into two Holoviz layers: the boxes, 2 px wide in a color per class, on a
// transparent RGBA8 image, and a "<class> 0.87" label above each box. Both
// carry the image's capture time, so Holoviz draws them over their own frame.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace yolo26_detection {

// Bytes of label text per detection; longer labels are cut.
inline constexpr std::uint32_t kOverlayLabelBytes = 32;

struct DetectionOverlayConfig {
  // The image's size, in pixels.
  int width{};
  int height{};
  // The detector's capacity: the length of its outputs.
  int max_detections{300};
  // Class names by id; ids without one show as "class N".
  std::vector<std::string> labels;
};

class DetectionOverlayOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument for a non-positive size or capacity.
  explicit DetectionOverlayOp(DetectionOverlayConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Yolo26DetectionOp's outputs, in GPU or host memory.
  holoscan::Input<holoscan::Tensor> boxes;
  holoscan::Input<holoscan::Tensor> scores;
  holoscan::Input<holoscan::Tensor> class_ids;
  holoscan::Input<holoscan::Tensor> count;
  // For a Holoviz text layer: per detection, a dark shadow row and then the
  // label row; unused rows are empty. Emitted first.
  holoscan::Output<holoscan::Tensor> labels;
  // For a Holoviz RGBA8 image layer: the boxes over a transparent background,
  // fully transparent without detections. Emitted last, so it can trigger
  // the frame.
  holoscan::Output<holoscan::schema::ImageT> overlay;

 private:
  DetectionOverlayConfig config_;
  std::vector<float> boxes_;
  std::vector<float> scores_;
  std::vector<std::int32_t> class_ids_;
};

}  // namespace yolo26_detection
