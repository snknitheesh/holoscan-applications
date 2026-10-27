// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_sources.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "detection/synthetic_scene.hpp"

namespace yolo26_detection {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

using Result = holoscan::expected<void, holoscan::Error>;

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

holoscan::TensorOutputSpec image_spec(holoscan::MemoryKind memory, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory,
              .dtype = kUInt8,
              .rank = 3U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{
      .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
  };
}

holoscan::schema::ImageEncoding encoding_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::kBgr8:
      return holoscan::schema::ImageEncoding_BGR8;
    case PixelFormat::kRgba8:
      return holoscan::schema::ImageEncoding_RGBA8;
    case PixelFormat::kBgra8:
      return holoscan::schema::ImageEncoding_BGRA8;
    case PixelFormat::kRgb8:
    default:
      return holoscan::schema::ImageEncoding_RGB8;
  }
}

// What compute() returns after an emit: on success nothing more to do, the
// frame is out; a consumer still busy with earlier frames makes the emit fail
// with kBackpressured before anything is delivered, and the frame goes again
// on the next tick; any other failure fails the run.
Result settle(Result sent, bool& delivered) {
  delivered = static_cast<bool>(sent);
  if (sent || sent.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};
  }
  return sent;
}

holoscan::schema::ImageT descriptor(int width, int height, PixelFormat format) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding_of(format);
  return image;
}

}  // namespace

void validate_config(const SyntheticImageSourceConfig& config) {
  // make_scene() checks the size.
  (void)make_scene(config.width, config.height, 0);
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the synthetic source needs a positive count and period");
  }
}

SyntheticImageSourceOp::SyntheticImageSourceOp(SyntheticImageSourceConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
}

void SyntheticImageSourceOp::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t bytes = static_cast<std::uint64_t>(config_.width) * config_.height *
                              static_cast<std::uint64_t>(channels(config_.format));
  spec.output(image, "image")
      .produces_tensor(image_spec(
          config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost,
          bytes))
      .payload_options(image_payload());
}

holoscan::Contract SyntheticImageSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticImageSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
}

holoscan::expected<void, holoscan::Error> SyntheticImageSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the run completes.
  }
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
  }
  const int n = channels(config_.format);
  const std::size_t pitch = static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(n);
  pixels_.resize(pitch * static_cast<std::size_t>(config_.height));
  render_scene(make_scene(config_.width, config_.height, emitted_), config_.format, pixels_.data(),
               pitch);
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{config_.height, config_.width, n}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (config_.device_output) {
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(
          failure("SyntheticImageSourceOp: the runtime gave no CUDA stream for the device output"));
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    // From pageable memory the copy is staged before it returns, so the
    // pixels may change afterwards.
    if (const cudaError_t status = cudaMemcpyAsync(writer->data_as<std::uint8_t>(), pixels_.data(),
                                                   pixels_.size(), cudaMemcpyHostToDevice, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(
          failure(std::string("SyntheticImageSourceOp: ") + cudaGetErrorString(status)));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  } else {
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      if (writer.byte_size() != pixels_.size()) {
        return holoscan::make_unexpected(failure("image loan has an unexpected size"));
      }
      std::memcpy(writer.data_as<std::uint8_t>(), pixels_.data(), pixels_.size());
      return {};
    });
    if (!committed) {
      return committed;
    }
  }
  bool delivered = false;
  const Result result = settle(
      image.emit_tensor(
          std::move(*loan), descriptor(config_.width, config_.height, config_.format),
          holoscan::EmitOptions{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U}),
      delivered);
  if (delivered) {
    pending_capture_.reset();
    ++emitted_;
  }
  return result;
}

void validate_config(const ImageFileSourceConfig& config) {
  if (config.files.empty()) {
    throw std::invalid_argument("the file source needs at least one image");
  }
  if (config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("the file source needs a positive period");
  }
  if (!config.loop && config.count > config.files.size()) {
    throw std::invalid_argument("--count exceeds the number of images; add --loop true");
  }
}

ImageFileSourceOp::ImageFileSourceOp(ImageFileSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
  for (const auto& file : config_.files) {
    infos_.push_back(read_image_info(file));
    max_bytes_ = std::max<std::uint64_t>(
        max_bytes_, static_cast<std::uint64_t>(infos_.back().width) * infos_.back().height * 3U);
  }
  total_ = config_.count == 0 ? static_cast<std::uint32_t>(config_.files.size()) : config_.count;
}

ImageFileSourceOp::~ImageFileSourceOp() = default;

void ImageFileSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(image, "image")
      .produces_tensor(image_spec(holoscan::MemoryKind::kCudaDevice, max_bytes_))
      .payload_options(image_payload());
}

holoscan::Contract ImageFileSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void ImageFileSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
}

void ImageFileSourceOp::stop() { decoder_.reset(); }

holoscan::expected<void, holoscan::Error> ImageFileSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= total_) {
    return {};
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        failure("ImageFileSourceOp: the runtime gave no CUDA stream for the device output"));
  }
  const std::size_t index = emitted_ % config_.files.size();
  const std::filesystem::path& path = config_.files[index];
  const ImageFileInfo& info = infos_[index];
  const PixelFormat format = config_.bgr ? PixelFormat::kBgr8 : PixelFormat::kRgb8;
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
  }
  auto loan = image.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 3>{info.height, info.width, 3}, .dtype = kUInt8});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const std::size_t pitch = static_cast<std::size_t>(info.width) * 3U;
  try {
    if (info.kind == ImageFileKind::kJpeg) {
      if (!decoder_) {
        decoder_ = std::make_unique<JpegDecoder>();
      }
      const std::vector<std::uint8_t> bytes = read_file(path);
      const std::string error = decoder_->decode(bytes, info.width, info.height, config_.bgr,
                                                 writer->data_as<std::uint8_t>(), pitch, stream);
      if (!error.empty()) {
        return holoscan::make_unexpected(failure(path.string() + ": " + error));
      }
      // The decoder may still read `bytes`; wait before they go.
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(path.string() + ": " + cudaGetErrorString(status)));
      }
    } else {
      int width = 0;
      int height = 0;
      std::vector<std::uint8_t> rgb = read_ppm(path, width, height);
      if (width != info.width || height != info.height) {
        return holoscan::make_unexpected(failure(path.string() + ": changed size during the run"));
      }
      if (config_.bgr) {
        for (std::size_t i = 0; i + 2 < rgb.size(); i += 3) {
          std::swap(rgb[i], rgb[i + 2]);
        }
      }
      // Copied from pageable memory: staged before the call returns.
      if (const cudaError_t status = cudaMemcpyAsync(writer->data_as<std::uint8_t>(), rgb.data(),
                                                     rgb.size(), cudaMemcpyHostToDevice, stream);
          status != cudaSuccess) {
        return holoscan::make_unexpected(
            failure(path.string() + ": " + cudaGetErrorString(status)));
      }
    }
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  bool delivered = false;
  const Result result = settle(
      image.emit_tensor(
          std::move(*loan), descriptor(info.width, info.height, format),
          holoscan::EmitOptions{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U}),
      delivered);
  if (delivered) {
    pending_capture_.reset();
    ++emitted_;
  }
  return result;
}

}  // namespace yolo26_detection
