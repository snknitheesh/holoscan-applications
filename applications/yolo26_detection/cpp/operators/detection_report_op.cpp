// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection_report_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <new>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "detection/image_files.hpp"
#include "detection/letterbox.hpp"
#include "detection/stub_model.hpp"

namespace yolo26_detection {
namespace {

using std::chrono_literals::operator""ns;

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "DetectionReportOp: " + message};
}

std::optional<PixelFormat> pixel_format(holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_RGB8:
      return PixelFormat::kRgb8;
    case holoscan::schema::ImageEncoding_BGR8:
      return PixelFormat::kBgr8;
    case holoscan::schema::ImageEncoding_RGBA8:
      return PixelFormat::kRgba8;
    case holoscan::schema::ImageEncoding_BGRA8:
      return PixelFormat::kBgra8;
    default:
      return std::nullopt;
  }
}

// Copies a packed tensor of `shape` and element type T to `out`, waiting for
// its producer.
template <typename T>
holoscan::expected<void, holoscan::Error> fetch(const holoscan::Tensor& tensor,
                                                std::span<const std::int64_t> shape,
                                                ::cuda::stream_ref stream, std::vector<T>& out) {
  const DLDataType dtype = tensor.dtype();
  const DLDataTypeCode code =
      std::is_floating_point_v<T> ? kDLFloat : (std::is_signed_v<T> ? kDLInt : kDLUInt);
  const std::span<const std::int64_t> actual = tensor.shape_span();
  if (dtype.code != code || dtype.bits != sizeof(T) * 8U || dtype.lanes != 1U ||
      !std::equal(actual.begin(), actual.end(), shape.begin(), shape.end())) {
    return holoscan::make_unexpected(invalid("unexpected detection tensor type or shape"));
  }
  std::size_t elements = 1;
  for (const std::int64_t extent : shape) {
    elements *= static_cast<std::size_t>(extent);
  }
  out.resize(elements);
  return tensor.copy_to_host(out.data(), out.size() * sizeof(T), stream);
}

}  // namespace

DetectionReportOp::DetectionReportOp(DetectionReportConfig config) : config_(std::move(config)) {
  if (!config_.report) {
    throw std::invalid_argument("DetectionReportOp needs a report");
  }
  if (config_.max_detections < 1) {
    throw std::invalid_argument("DetectionReportOp needs a positive max_detections");
  }
  if (!(config_.expect_iou > 0.0F && config_.expect_iou <= 1.0F)) {
    throw std::invalid_argument("DetectionReportOp: expect_iou must be in (0, 1]");
  }
  if (!config_.expectations.empty() && config_.files.empty()) {
    throw std::invalid_argument("DetectionReportOp: expectations apply to image files");
  }
}

void DetectionReportOp::setup(holoscan::OperatorSpec& spec) {
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(boxes, "boxes").queue_depth(2U);
  spec.input(scores, "scores").queue_depth(2U);
  spec.input(class_ids, "class_ids").queue_depth(2U);
  spec.input(count, "count").queue_depth(2U);
  if (!config_.image_dir.empty()) {
    spec.input(image, "image").queue_depth(2U);
  }
}

holoscan::Contract DetectionReportOp::contract() const {
  holoscan::Contract result;
  std::vector<holoscan::PortRef> ports{boxes, scores, class_ids, count};
  if (!config_.image_dir.empty()) {
    ports.emplace_back(image);
  }
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(boxes, holoscan::Exact{});
  result.input(scores, holoscan::Exact{});
  result.input(class_ids, holoscan::Exact{});
  result.input(count, holoscan::Exact{});
  if (!config_.image_dir.empty()) {
    result.input(image, holoscan::Exact{});
  }
  return result;
}

// Each run counts its own frames from the start.
void DetectionReportOp::start() {
  seen_ = 0;
  last_frame_id_ = 0;
}

std::string DetectionReportOp::label(std::int32_t class_id) const {
  if (class_id >= 0 && static_cast<std::size_t>(class_id) < config_.labels.size()) {
    return config_.labels[static_cast<std::size_t>(class_id)];
  }
  return "class " + std::to_string(class_id);
}

holoscan::expected<void, holoscan::Error> DetectionReportOp::compute(
    holoscan::ExecutionContext& context) {
  // The host buffers grow at the first frame, or at a larger image.
  try {
    return process(context);
  } catch (const std::bad_alloc&) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted,
                                                     "DetectionReportOp: out of host memory"});
  }
}

holoscan::expected<void, holoscan::Error> DetectionReportOp::process(
    holoscan::ExecutionContext& context) {
  auto count_sample = count.receive();
  auto boxes_sample = boxes.receive();
  auto scores_sample = scores.receive();
  auto class_ids_sample = class_ids.receive();
  if (!count_sample) return holoscan::make_unexpected(std::move(count_sample).error());
  if (!boxes_sample) return holoscan::make_unexpected(std::move(boxes_sample).error());
  if (!scores_sample) return holoscan::make_unexpected(std::move(scores_sample).error());
  if (!class_ids_sample) return holoscan::make_unexpected(std::move(class_ids_sample).error());

  const std::int64_t capacity = config_.max_detections;
  const std::array<std::int64_t, 1> one{1};
  const std::array<std::int64_t, 2> box_shape{capacity, 4};
  const std::array<std::int64_t, 1> row_shape{capacity};
  std::vector<std::int32_t> counted;
  for (auto fetched :
       {fetch(count_sample->data, one, context.cuda_stream(), counted),
        fetch(boxes_sample->data, box_shape, context.cuda_stream(), boxes_),
        fetch(scores_sample->data, row_shape, context.cuda_stream(), scores_),
        fetch(class_ids_sample->data, row_shape, context.cuda_stream(), class_ids_)}) {
    if (!fetched) {
      return fetched;
    }
  }
  const std::int32_t n = counted.front();
  if (n < 0 || n > capacity) {
    return holoscan::make_unexpected(invalid("detection count out of range"));
  }
  std::vector<BoxDetection> detections;
  detections.reserve(static_cast<std::size_t>(n));
  for (std::int32_t i = 0; i < n; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    detections.push_back(BoxDetection{.x1 = boxes_[4 * k],
                                      .y1 = boxes_[4 * k + 1],
                                      .x2 = boxes_[4 * k + 2],
                                      .y2 = boxes_[4 * k + 3],
                                      .score = scores_[k],
                                      .class_id = class_ids_[k]});
  }

  DetectionReport& report = *config_.report;
  const holoscan::SampleMetadata& metadata = count_sample->metadata;
  // The detections are on the host now: the time since capture is the
  // latency a host consumer sees.
  if (auto now = context.clock().now();
      now && metadata.source_clock_id.valid() && now->clock_id == metadata.source_clock_id) {
    report.latencies_ms.push_back(
        static_cast<double>(now->timestamp_ns - metadata.capture_timestamp_ns) * 1e-6);
  }
  // Sources number their frames from 1; the detections keep their image's.
  const std::uint64_t frame_id = metadata.frame_id;
  if (frame_id != last_frame_id_ + 1U) {
    ++report.frame_gaps;
  }
  last_frame_id_ = frame_id;
  report.detections += detections.size();

  if (config_.truth && frame_id != 0) {
    const SyntheticTruth& truth = *config_.truth;
    const SyntheticScene scene =
        make_scene(truth.width, truth.height, static_cast<std::uint32_t>(frame_id - 1U));
    const float scale = make_letterbox(truth.width, truth.height, truth.input_size).scale;
    // A box edge lands within a model input pixel or so of the true one.
    const TruthMatch match =
        match_truth(detections, scene_truth(scene), 2.0F / scale + 1.0F, 1e-3F);
    report.verified = true;
    report.missed += match.missed;
    report.extra += match.extra;
    report.max_edge_error = std::max(report.max_edge_error, match.max_edge_error);
  }

  if (!config_.files.empty() && frame_id != 0) {
    const std::size_t index = (frame_id - 1U) % config_.files.size();
    const std::string name = config_.files[index].filename().string();
    std::vector<Expectation> expected;
    for (const Expectation& e : config_.expectations) {
      if (e.image == name) {
        expected.push_back(e);
      }
    }
    report.expected += expected.size();
    report.matched += count_matched(detections, expected, config_.expect_iou);
    if (config_.print && frame_id <= config_.files.size()) {
      std::ostringstream out;
      out << std::fixed << "yolo26_detection: " << name << ": " << detections.size()
          << (detections.size() == 1 ? " detection\n" : " detections\n");
      for (const BoxDetection& d : detections) {
        out << "  " << label(d.class_id) << " (" << d.class_id << ") " << std::setprecision(3)
            << d.score << std::setprecision(1) << " [" << d.x1 << ", " << d.y1 << ", " << d.x2
            << ", " << d.y2 << "]\n";
      }
      std::cout << out.str() << std::flush;
    }
  }

  if (!config_.image_dir.empty()) {
    auto frame = image.receive();
    if (!frame) {
      return holoscan::make_unexpected(std::move(frame).error());
    }
    if (auto written = write_image(context, frame->data, frame_id, detections); !written) {
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

holoscan::expected<void, holoscan::Error> DetectionReportOp::write_image(
    holoscan::ExecutionContext& context, const holoscan::schema::ImageT& frame,
    std::uint64_t frame_id, const std::vector<BoxDetection>& detections) {
  const auto format = pixel_format(frame.encoding);
  if (!format || !frame.data) {
    return holoscan::make_unexpected(invalid("can draw only RGB8, BGR8, RGBA8 and BGRA8 images"));
  }
  const holoscan::Tensor& tensor = *frame.data;
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const int n = channels(*format);
  if (shape.size() != 3 || shape[2] != n || !tensor.is_contiguous()) {
    return holoscan::make_unexpected(invalid("the image must be packed [height, width, channels]"));
  }
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);
  pixels_.resize(static_cast<std::size_t>(width) * height * n);
  if (auto copied = tensor.copy_to_host(pixels_.data(), pixels_.size(), context.cuda_stream());
      !copied) {
    return copied;
  }
  std::vector<std::uint8_t> rgb =
      to_rgb(pixels_.data(), static_cast<std::size_t>(width) * n, width, height, *format);
  draw_boxes(rgb, width, height, detections);
  // Numbered by frame id, six digits at least.
  std::string number = std::to_string(frame_id);
  number.insert(0, number.size() < 6 ? 6 - number.size() : 0, '0');
  const std::string name = "detections_" + number + ".ppm";
  try {
    write_ppm(config_.image_dir / name, width, height, rgb);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
  }
  ++config_.report->files_written;
  return {};
}

}  // namespace yolo26_detection
