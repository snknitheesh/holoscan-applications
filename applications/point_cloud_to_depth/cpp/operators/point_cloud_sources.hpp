// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Point cloud sources for the application: the synthetic cloud, and a folder
// of PLY files. Both emit a fixed number of clouds on a clock, then idle.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace point_cloud_to_depth {

struct SyntheticCloudSourceConfig {
  // The synthetic camera's image size, which sets the cloud's size.
  int width{640};
  int height{480};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Scene depths in whole millimeters, which a uint16 depth image in
  // millimeters reproduces exactly.
  bool millimeter_depth{false};
  // Also emit each point's RGB8 color on `colors`.
  bool color{false};
  // Also emit each frame's intrinsics on `intrinsics`, varying by frame.
  bool intrinsics{false};
  // Emit points and colors in CUDA device memory instead of host memory.
  bool device_output{true};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SyntheticCloudSourceConfig& config);

// Clouds of the synthetic camera in synthetic_scene.hpp, frame = 0, 1, ...,
// in the body frame (synthetic_body_from_camera()), as float32 [N, 3] with N the same for
// every frame.
class SyntheticCloudSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SyntheticCloudSourceOp(SyntheticCloudSourceConfig config);
  ~SyntheticCloudSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::Tensor> points;
  holoscan::Output<holoscan::Tensor> colors;
  holoscan::Output<holoscan::Tensor> intrinsics;

 private:
  void release() noexcept;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> ensure_scratch();

  SyntheticCloudSourceConfig config_;
  std::uint32_t point_count_{};
  std::uint32_t emitted_{};
  // The frame being sent: its capture time and which outputs are out.
  std::optional<holoscan::TimePoint> pending_capture_;
  std::uint32_t sent_{};
  // Generation scratch for host output, and the stream it uses when the
  // runtime gives none.
  void* scratch_{};
  cudaStream_t own_stream_{};
};

struct CloudFileSourceConfig {
  std::vector<std::filesystem::path> files;
  // The most points in any file, which sizes the outputs: read the headers
  // with read_ply_header() before building the graph.
  std::uint64_t max_points{};
  // Every file has colors, and they are emitted on `colors`.
  bool color{false};
  // Clouds to emit; 0 emits every file once.
  std::uint32_t count{0};
  // Start over after the last file until `count` clouds are out.
  bool loop{false};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const CloudFileSourceConfig& config);

// float32 [N, 3] clouds, and with `color` uint8 [N, 3] colors, in host
// memory, from PLY files, in order. A file that cannot be read, or that has
// more points than configured or not the configured colors, fails the run.
class CloudFileSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit CloudFileSourceOp(CloudFileSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::Tensor> points;
  holoscan::Output<holoscan::Tensor> colors;

 private:
  CloudFileSourceConfig config_;
  std::uint32_t total_{};
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> pending_capture_;
  std::uint32_t sent_{};
};

}  // namespace point_cloud_to_depth
