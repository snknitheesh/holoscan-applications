// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Depth sources for the application: the synthetic depth camera, and a folder
// of PGM depth images. Both emit a fixed number of frames on a clock, then idle.

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
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "deproject/image_io.hpp"

namespace depth_to_point_cloud {

struct SyntheticSourceConfig {
  int width{640};
  int height{480};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // uint16 millimeters (MONO16) instead of float32 meters (MONO32F).
  bool millimeters{false};
  // Also emit an aligned RGB8 image on `color`.
  bool color{false};
  // Also emit each frame's intrinsics on `intrinsics`, varying by frame.
  bool intrinsics{false};
  // Emit images in CUDA device memory instead of host memory.
  bool device_output{true};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SyntheticSourceConfig& config);

// Frames of the synthetic camera in synthetic_depth.hpp, frame = 0, 1, ...
class SyntheticDepthSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SyntheticDepthSourceOp(SyntheticSourceConfig config);
  ~SyntheticDepthSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> depth;
  holoscan::Output<holoscan::schema::ImageT> color;
  holoscan::Output<holoscan::Tensor> intrinsics;

 private:
  static constexpr std::uint32_t kSentIntrinsics = 1U;
  static constexpr std::uint32_t kSentColor = 2U;
  static constexpr std::uint32_t kSentDepth = 4U;

  void release() noexcept;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> ensure_scratch(std::size_t bytes);

  SyntheticSourceConfig config_;
  std::uint32_t emitted_{};
  // The frame being sent: its capture time and which outputs are out.
  std::optional<holoscan::TimePoint> pending_capture_;
  std::uint32_t sent_{};
  // Rendering scratch for host output, and the stream used without a runtime one.
  void* scratch_{};
  cudaStream_t own_stream_{};
};

struct DepthFileSourceConfig {
  std::vector<std::filesystem::path> files;
  // The size of every image, which sizes the output: read it from the first
  // file with read_pgm_depth() before building the graph.
  int width{};
  int height{};
  // Frames to emit; 0 emits every file once.
  std::uint32_t count{0};
  // Start over after the last file until `count` frames are out.
  bool loop{false};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const DepthFileSourceConfig& config);

// MONO16 frames, in host memory, from 8- or 16-bit PGM files, in order. A
// file that cannot be read, or whose size is not the configured one, fails
// the run.
class DepthFileSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit DepthFileSourceOp(DepthFileSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> depth;

 private:
  DepthFileSourceConfig config_;
  std::uint32_t total_{};
  std::uint32_t emitted_{};
};

}  // namespace depth_to_point_cloud
