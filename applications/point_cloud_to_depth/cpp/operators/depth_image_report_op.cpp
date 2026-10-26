// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "depth_image_report_op.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "projection/cloud_io.hpp"
#include "projection/synthetic_scene.hpp"

namespace point_cloud_to_depth {
namespace {

using std::chrono_literals::operator""ns;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "DepthImageReportOp: " + message};
}

// Copies `image` (of `encoding`, `channels` elements of type T per pixel,
// packed) to `out`, waiting for its producer.
template <typename T>
holoscan::expected<void, holoscan::Error> fetch(const holoscan::schema::ImageT& image,
                                                holoscan::schema::ImageEncoding encoding,
                                                std::int64_t channels, ::cuda::stream_ref stream,
                                                std::vector<T>& out) {
  if (!image.data) {
    return holoscan::make_unexpected(invalid("image has no tensor"));
  }
  if (image.encoding != encoding) {
    return holoscan::make_unexpected(
        invalid(std::string("expected a ") + holoscan::schema::EnumNameImageEncoding(encoding) +
                " image, got " + holoscan::schema::EnumNameImageEncoding(image.encoding)));
  }
  const holoscan::Tensor& tensor = *image.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const bool shape_ok =
      channels == 1 ? shape.size() == 2 : shape.size() == 3 && shape[2] == channels;
  if (!shape_ok || shape[0] != image.height || shape[1] != image.width ||
      tensor.dtype().code != (std::is_floating_point_v<T> ? kDLFloat : kDLUInt) ||
      tensor.dtype().bits != sizeof(T) * 8U ||
      tensor.nbytes() != static_cast<std::int64_t>(shape[0] * shape[1] * channels *
                                                   static_cast<std::int64_t>(sizeof(T)))) {
    return holoscan::make_unexpected(invalid("image tensor disagrees with its descriptor"));
  }
  out.resize(static_cast<std::size_t>(shape[0] * shape[1] * channels));
  return tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream);
}

}  // namespace

DepthImageReportOp::DepthImageReportOp(DepthImageReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("DepthImageReportOp needs a report");
  }
  if (!(std::isfinite(config_.depth_scale) && config_.depth_scale > 0)) {
    throw std::invalid_argument("DepthImageReportOp needs a finite, positive depth_scale");
  }
  if (config_.truth) {
    const SceneTruth& scene = config_.truth->scene;
    if (scene.format != config_.depth_format) {
      throw std::invalid_argument(
          "DepthImageReportOp: the truth's depth format is not the images'");
    }
    // check_against_scene() compares uint16 counts exactly, which holds only
    // for a scene in whole millimeters counted in millimeters.
    if (scene.format == DepthFormat::kUint16 &&
        (!scene.millimeter_depth || scene.depth_scale != 0.001F ||
         config_.depth_scale != scene.depth_scale)) {
      throw std::invalid_argument(
          "DepthImageReportOp: a uint16 truth needs a millimeter scene and a 0.001 scale");
    }
  }
}

void DepthImageReportOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one image in compute() while the next one arrives.
  spec.input(depth, "depth").queue_depth(2U);
  if (config_.color) {
    spec.input(color, "color").queue_depth(2U);
  }
}

// Each run counts its own images and frame ids from the start.
void DepthImageReportOp::start() {
  seen_ = 0;
  last_frame_id_ = 0;
}

holoscan::Contract DepthImageReportOp::contract() const {
  holoscan::Contract result;
  if (!config_.color) {
    result.trigger(holoscan::OnEach{depth});
    return result;
  }
  result.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::vector<holoscan::PortRef>{depth, color}}, .window = 0ns});
  result.input(depth, holoscan::Exact{});
  result.input(color, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> DepthImageReportOp::compute(
    holoscan::ExecutionContext& context) {
  // The host buffers grow at the first image, or at a larger one.
  try {
    return process(context);
  } catch (const std::bad_alloc&) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted,
                                                     "DepthImageReportOp: out of host memory"});
  }
}

holoscan::expected<void, holoscan::Error> DepthImageReportOp::process(
    holoscan::ExecutionContext& context) {
  auto image = depth.receive();
  if (!image) {
    return holoscan::make_unexpected(std::move(image).error());
  }
  const bool uint16 = config_.depth_format == DepthFormat::kUint16;
  auto fetched = uint16 ? fetch(image->data, holoscan::schema::ImageEncoding_MONO16, 1,
                                context.cuda_stream(), depth_counts_)
                        : fetch(image->data, holoscan::schema::ImageEncoding_MONO32F, 1,
                                context.cuda_stream(), depth_m_);
  if (!fetched) {
    return fetched;
  }
  const int width = image->data.width;
  const int height = image->data.height;
  const std::uint8_t* rgb = nullptr;
  if (config_.color) {
    auto sample = color.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if (auto color_fetched = fetch(sample->data, holoscan::schema::ImageEncoding_RGB8, 3,
                                   context.cuda_stream(), rgb_);
        !color_fetched) {
      return color_fetched;
    }
    if (sample->data.width != width || sample->data.height != height) {
      return holoscan::make_unexpected(invalid("color and depth images differ in size"));
    }
    rgb = rgb_.data();
  }

  DepthReport& report = *config_.report;
  // Sources number their frames from 1; the images keep their cloud's id.
  const std::uint64_t frame_id = image->metadata.frame_id;
  if (frame_id != last_frame_id_ + 1U) {
    ++report.frame_gaps;
  }
  last_frame_id_ = frame_id;
  measure(width, height);
  if (config_.truth) {
    report.verified = true;
    const DepthImageTruth& truth = *config_.truth;
    if (width != truth.width || height != truth.height) {
      report.wrong_depths += static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    } else {
      const TruthCheck check = check_against_scene(
          synthetic_frame(width, height, static_cast<std::uint32_t>(frame_id - 1U), false),
          truth.scene, uint16 ? static_cast<const void*>(depth_counts_.data()) : depth_m_.data(),
          rgb);
      report.wrong_depths += check.wrong_depths;
      report.wrong_colors += check.wrong_colors;
      report.max_error_m = std::max(report.max_error_m, check.max_error_m);
    }
  }
  if (!config_.image_dir.empty()) {
    if (auto written = write_images(width, height, rgb != nullptr); !written) {
      return written;
    }
  }
  ++seen_;
  report.frames.fetch_add(1U);
  if (config_.expected_frames != 0 && seen_ == config_.expected_frames) {
    context.request_completion();
  }
  return {};
}

float DepthImageReportOp::depth_at(std::size_t pixel) const {
  return config_.depth_format == DepthFormat::kUint16
             ? static_cast<float>(depth_counts_[pixel]) * config_.depth_scale
             : depth_m_[pixel];
}

void DepthImageReportOp::measure(int width, int height) {
  DepthReport& report = *config_.report;
  const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  std::uint64_t filled = 0;
  for (std::size_t i = 0; i < pixels; ++i) {
    const float z = depth_at(i);
    if (z != 0.0F) {
      ++filled;
      lo = std::min(lo, z);
      hi = std::max(hi, z);
    }
  }
  if (filled != 0) {
    report.min_depth_m = report.filled_pixels == 0 ? lo : std::min(report.min_depth_m, lo);
    report.max_depth_m = report.filled_pixels == 0 ? hi : std::max(report.max_depth_m, hi);
  }
  report.pixels += pixels;
  report.filled_pixels += filled;
}

holoscan::expected<void, holoscan::Error> DepthImageReportOp::write_images(int width, int height,
                                                                           bool with_color) {
  DepthReport& report = *config_.report;
  const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  // Depth in whole millimeters, which image viewers and tools read as 16-bit.
  millimeters_.resize(pixels);
  for (std::size_t i = 0; i < pixels; ++i) {
    const float z = depth_at(i);
    millimeters_[i] = static_cast<std::uint16_t>(
        std::clamp(std::round(static_cast<double>(z) * 1000.0), 0.0, 65535.0));
  }
  // Numbered from 0 by this reporter's images.
  char depth_name[32];
  char color_name[32];
  std::snprintf(depth_name, sizeof(depth_name), "depth_%06u.pgm", seen_);
  std::snprintf(color_name, sizeof(color_name), "color_%06u.ppm", seen_);
  try {
    write_pgm(config_.image_dir / depth_name, width, height, millimeters_);
    ++report.files_written;
    if (with_color) {
      write_ppm(config_.image_dir / color_name, width, height, rgb_);
      ++report.files_written;
    }
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
  }
  return {};
}

}  // namespace point_cloud_to_depth
