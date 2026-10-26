// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PointCloudReportOp: the application's sink. Counts valid points and their
// depth range, checks that frame ids run 1, 2, 3, ... without a gap, checks
// every point (and color) against the synthetic camera's ground truth when the
// input is synthetic, optionally writes PLY files, and asks the run to
// complete after its expected frames.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "deproject/deproject.hpp"

namespace depth_to_point_cloud {

// What one run saw. Written by the reporter during the run and read by main()
// after the run has stopped.
struct CloudReport {
  std::uint32_t frames{};
  std::uint64_t points{};
  std::uint64_t valid_points{};
  float min_depth{};
  float max_depth{};
  std::uint32_t ply_files{};
  // Clouds whose frame id is not the one after the previous cloud's.
  std::uint32_t frame_gaps{};
  // Ground-truth checks; zero when the input is not synthetic.
  bool verified{};
  std::uint64_t wrong_points{};  // Valid where invalid, invalid where valid, or misplaced.
  std::uint64_t wrong_colors{};
  double max_error_m{};  // Largest distance of a valid point from the truth.

  [[nodiscard]] bool passed() const noexcept {
    return frame_gaps == 0 && (!verified || (wrong_points == 0 && wrong_colors == 0));
  }
};

// The synthetic camera and operator settings the clouds should match.
struct CloudTruth {
  int width{};
  int height{};
  bool millimeters{};
  bool vary_intrinsics{};
  float depth_scale{};
  float depth_min{};
  float depth_max{};
};

struct PointCloudReportConfig {
  std::shared_ptr<CloudReport> report;
  // Frames after which the run completes; 0 never completes.
  std::uint32_t expected_frames{};
  // Declare the `colors` input.
  bool colors{false};
  std::optional<CloudTruth> truth;
  // Where to write frame_NNNNNN.ply; none when empty.
  std::filesystem::path ply_dir;
};

// Largest distance, in meters, a verified point may be from the truth.
inline constexpr double kMaxPointError = 1e-4;

class PointCloudReportOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `report` is missing.
  explicit PointCloudReportOp(PointCloudReportConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> point_cloud;
  holoscan::Input<holoscan::Tensor> colors;

 private:
  void verify(std::uint32_t frame, const std::vector<float>& xyz,
              const std::vector<std::uint8_t>* rgb, int width, int height);

  PointCloudReportConfig config_;
  std::vector<float> xyz_;
  std::vector<std::uint8_t> rgb_;
  std::uint64_t last_frame_id_{};
  bool finished_{};
};

}  // namespace depth_to_point_cloud
