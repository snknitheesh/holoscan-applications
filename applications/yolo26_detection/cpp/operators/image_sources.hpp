// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Image sources for the application: synthetic frames with known detections
// for the stub model, and a list of JPEG or PPM files. Both emit a fixed
// number of frames on a clock, numbered from 1, then idle. A frame that a
// busy consumer pushes back is sent again on the next tick with its original
// capture time.

#pragma once

#include <chrono>
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

#include "detection/image_files.hpp"
#include "detection/letterbox.hpp"

namespace yolo26_detection {

struct SyntheticImageSourceConfig {
  int width{1280};
  int height{720};
  std::uint32_t count{100};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  PixelFormat format{PixelFormat::kRgb8};
  // Emit frames in GPU memory rather than host memory.
  bool device_output{true};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const SyntheticImageSourceConfig& config);

// Frame f (1, 2, ...) is make_scene(width, height, f - 1), drawn in `format`.
class SyntheticImageSourceOp final : public holoscan::Operator<> {
 public:
  // @throws std::invalid_argument if `config` is not valid.
  explicit SyntheticImageSourceOp(SyntheticImageSourceConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  SyntheticImageSourceConfig config_;
  std::vector<std::uint8_t> pixels_;
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> pending_capture_;
};

struct ImageFileSourceConfig {
  std::vector<std::filesystem::path> files;
  // Frames to emit; 0 emits every file once.
  std::uint32_t count{0};
  // Start over after the last file until `count` frames are out.
  bool loop{false};
  std::chrono::nanoseconds period{std::chrono::milliseconds{33}};
  // Emit BGR8 rather than RGB8.
  bool bgr{false};
};

// @throws std::invalid_argument naming the first unusable field.
void validate_config(const ImageFileSourceConfig& config);

// Frame f (1, 2, ...) is files[(f - 1) % files.size()], decoded into GPU
// memory as RGB8 (or BGR8): JPEG files with nvJPEG, PPM files on the host.
class ImageFileSourceOp final : public holoscan::Operator<> {
 public:
  // Reads every file's header to size the output.
  // @throws std::invalid_argument if `config` is not valid, or
  // std::runtime_error for a file it cannot use.
  explicit ImageFileSourceOp(ImageFileSourceConfig config);
  ~ImageFileSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Frames this source emits in total.
  [[nodiscard]] std::uint32_t frame_count() const noexcept { return total_; }

  holoscan::Output<holoscan::schema::ImageT> image;

 private:
  ImageFileSourceConfig config_;
  std::vector<ImageFileInfo> infos_;
  std::uint64_t max_bytes_{};
  std::uint32_t total_{};
  std::uint32_t emitted_{};
  std::optional<holoscan::TimePoint> pending_capture_;
  std::unique_ptr<JpegDecoder> decoder_;
};

}  // namespace yolo26_detection
