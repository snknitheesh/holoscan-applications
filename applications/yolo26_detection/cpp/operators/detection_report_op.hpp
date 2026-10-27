// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DetectionReportOp: the application's sink. It takes each image's
// detections from Yolo26DetectionOp, checks that frame ids run 1, 2, 3, ...
// without a gap, measures the time from capture to detections on the host,
// and then, depending on the input:
//
// - synthetic frames with the stub model: checks the detections against the
//   frame's known ones;
// - image files: prints each file's detections the first time it is seen,
//   and checks the expected detections listed for it, if any.
//
// It optionally takes the image too, to write it with its boxes drawn as a
// PPM file, and asks the run to complete after its expected frames.

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
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

#include "detection/matching.hpp"
#include "detection/synthetic_scene.hpp"

namespace yolo26_detection {

// What one run saw. Written by the reporter during the run and read by main()
// after it has stopped; only `frames` may be read while it runs.
struct DetectionReport {
  std::atomic<std::uint32_t> frames{};
  std::uint64_t detections{};
  // Frames whose id is not the one after the previous frame's.
  std::uint32_t frame_gaps{};
  // Synthetic truth: detections missed and extra, and the largest edge error
  // of the nearest candidates, in image pixels.
  bool verified{};
  std::uint64_t missed{};
  std::uint64_t extra{};
  float max_edge_error{};
  // Expected detections of image files, and how many were found.
  std::uint64_t expected{};
  std::uint64_t matched{};
  std::uint32_t files_written{};
  // Capture to detections on the host, per frame, in milliseconds.
  std::vector<double> latencies_ms;

  [[nodiscard]] bool passed() const noexcept {
    return frame_gaps == 0 && (!verified || (missed == 0 && extra == 0)) && matched == expected;
  }
};

// The synthetic frames the detections should match, and the letterbox side
// that sets how closely.
struct SyntheticTruth {
  int width{};
  int height{};
  int input_size{};
};

struct DetectionReportConfig {
  // Where the reporter adds what it sees; one report per run.
  std::shared_ptr<DetectionReport> report;
  // Frames after which the run completes; 0 never completes.
  std::uint32_t expected_frames{};
  // The detector's capacity: the length of its outputs.
  int max_detections{300};
  // Check synthetic frames against their known detections.
  std::optional<SyntheticTruth> truth;
  // The images of a file source, frame f being files[(f - 1) % size]: named
  // when printing, and looked up in `expectations`.
  std::vector<std::filesystem::path> files;
  std::vector<Expectation> expectations;
  float expect_iou{0.8F};
  // Class names by id, for printing; ids without one print as numbers.
  std::vector<std::string> labels;
  // Print each file's detections when it is first seen.
  bool print{false};
  // Declare the `image` input and write each frame with its boxes drawn as
  // `image_dir`/detections_NNNNNN.ppm, numbered by frame id.
  std::filesystem::path image_dir;
};

class DetectionReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if the report is missing or the settings
  // unusable.
  explicit DetectionReportOp(DetectionReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> boxes;
  holoscan::Input<holoscan::Tensor> scores;
  holoscan::Input<holoscan::Tensor> class_ids;
  holoscan::Input<holoscan::Tensor> count;
  holoscan::Input<holoscan::schema::ImageT> image;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> process(
      holoscan::ExecutionContext& context);
  [[nodiscard]] holoscan::expected<void, holoscan::Error> write_image(
      holoscan::ExecutionContext& context, const holoscan::schema::ImageT& frame,
      std::uint64_t frame_id, const std::vector<BoxDetection>& detections);
  [[nodiscard]] std::string label(std::int32_t class_id) const;

  DetectionReportConfig config_;
  std::uint64_t last_frame_id_{};
  std::uint32_t seen_{};
  std::vector<float> boxes_;
  std::vector<float> scores_;
  std::vector<std::int32_t> class_ids_;
  std::vector<std::uint8_t> pixels_;
};

}  // namespace yolo26_detection
