// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DepthImageReportOp: the application's sink. Counts filled pixels and their
// depth range, checks that frame ids run 1, 2, 3, ... without a gap, checks
// every pixel (and color) against the synthetic camera's ground truth when the
// input is synthetic, optionally writes the images as PGM and PPM files, and
// asks the run to complete after its expected frames.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "projection/depth_truth.hpp"
#include "projection/project.hpp"

namespace point_cloud_to_depth {

// What one run saw. Written by the reporter during the run and read by main()
// after the run has stopped; only `frames` may be read while it runs, to see
// that it makes progress.
struct DepthReport {
  std::atomic<std::uint32_t> frames{};
  std::uint64_t pixels{};
  std::uint64_t filled_pixels{};
  float min_depth_m{};
  float max_depth_m{};
  std::uint32_t files_written{};
  // Images whose frame id is not the one after the previous image's.
  std::uint32_t frame_gaps{};
  // Ground-truth checks; zero when the input is not synthetic.
  bool verified{};
  std::uint64_t wrong_depths{};
  std::uint64_t wrong_colors{};
  double max_error_m{};

  [[nodiscard]] bool passed() const noexcept {
    return frame_gaps == 0 && (!verified || (wrong_depths == 0 && wrong_colors == 0));
  }
};

// The synthetic camera the images should match.
struct DepthImageTruth {
  int width{};
  int height{};
  // The operator settings, as check_against_scene() takes them.
  SceneTruth scene;
};

struct DepthImageReportConfig {
  // Where the reporter adds what it sees; one report per run.
  std::shared_ptr<DepthReport> report;
  // Frames after which the run completes, counted by each reporter on its
  // own; 0 never completes.
  std::uint32_t expected_frames{};
  // The depth images' format and, for uint16, meters per count.
  DepthFormat depth_format{DepthFormat::kFloat32};
  float depth_scale{0.001F};
  // Declare the `color` input.
  bool color{false};
  // Its format must be `depth_format`, and uint16 depth must be in whole
  // millimeters of a millimeter-depth scene (see SceneTruth).
  std::optional<DepthImageTruth> truth;
  // Where to write depth_NNNNNN.pgm (millimeters) and color_NNNNNN.ppm; none
  // when empty.
  std::filesystem::path image_dir;
};

class DepthImageReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` is missing, the scale unusable,
  // or `truth` does not fit the depth format.
  explicit DepthImageReportOp(DepthImageReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> depth;
  holoscan::Input<holoscan::schema::ImageT> color;

 private:
  // The depth of pixel `pixel` of the current image in meters, 0 if empty.
  [[nodiscard]] float depth_at(std::size_t pixel) const;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> process(
      holoscan::ExecutionContext& context);
  void measure(int width, int height);
  [[nodiscard]] holoscan::expected<void, holoscan::Error> write_images(int width, int height,
                                                                       bool with_color);

  DepthImageReportConfig config_;
  // The current images, on the host: float32 meters or uint16 counts, and RGB8;
  // and the depth image in millimeters for its file. Sized at the first image.
  std::vector<float> depth_m_;
  std::vector<std::uint16_t> depth_counts_;
  std::vector<std::uint8_t> rgb_;
  std::vector<std::uint16_t> millimeters_;
  std::uint64_t last_frame_id_{};
  // Images this reporter has seen; it requests completion at expected_frames.
  std::uint32_t seen_{};
};

}  // namespace point_cloud_to_depth
