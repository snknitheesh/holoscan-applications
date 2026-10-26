// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "depth_sources.hpp"

#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "deproject/synthetic_depth.hpp"

namespace depth_to_point_cloud {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

holoscan::TensorOutputSpec tensor_spec(holoscan::MemoryKind memory, DLDataType dtype,
                                       std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = memory,
              .dtype = dtype,
              .rank = rank,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

holoscan::Error failure(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

holoscan::PayloadOptions image_payload() {
  return holoscan::PayloadOptions{
      .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
  };
}

holoscan::schema::ImageT image_descriptor(int width, int height,
                                          holoscan::schema::ImageEncoding encoding) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  return image;
}

// A consumer still busy with earlier frames makes an emit fail with
// kBackpressured before anything is delivered; the source then sends the same
// frame on its next tick instead of failing the run or skipping the frame.
holoscan::expected<void, holoscan::Error> retry_if_backpressured(
    holoscan::expected<void, holoscan::Error> emitted, std::uint32_t& sent) {
  if (emitted) {
    ++sent;
    return {};
  }
  if (emitted.error().code == holoscan::ErrorCode::kBackpressured) {
    return {};
  }
  return emitted;
}

}  // namespace

void validate_config(const SyntheticSourceConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("synthetic depth images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("synthetic source needs a positive count and period");
  }
}

SyntheticDepthSourceOp::SyntheticDepthSourceOp(SyntheticSourceConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
}

SyntheticDepthSourceOp::~SyntheticDepthSourceOp() { release(); }

void SyntheticDepthSourceOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height);
  const holoscan::MemoryKind memory =
      config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost;
  spec.output(depth, "depth")
      .produces_tensor(tensor_spec(memory, config_.millimeters ? kUInt16 : kFloat32, 2U,
                                   pixels * (config_.millimeters ? 2U : 4U)))
      .payload_options(image_payload());
  if (config_.color) {
    spec.output(color, "color")
        .produces_tensor(tensor_spec(memory, kUInt8, 3U, pixels * 3U))
        .payload_options(image_payload());
  }
  if (config_.intrinsics) {
    spec.output(intrinsics, "intrinsics")
        .produces_tensor(
            tensor_spec(holoscan::MemoryKind::kHost, kFloat32, 1U, 4U * sizeof(float)));
  }
}

holoscan::Contract SyntheticDepthSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticDepthSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
  sent_ = 0U;
}

void SyntheticDepthSourceOp::stop() { release(); }

void SyntheticDepthSourceOp::release() noexcept {
  if (own_stream_ != nullptr) {
    cudaStreamSynchronize(own_stream_);
    cudaStreamDestroy(own_stream_);
    own_stream_ = nullptr;
  }
  if (scratch_ != nullptr) {
    cudaFree(scratch_);
    scratch_ = nullptr;
  }
}

holoscan::expected<void, holoscan::Error> SyntheticDepthSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the consumers request completion.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    if (own_stream_ == nullptr) {
      if (const cudaError_t status = cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking);
          status != cudaSuccess) {
        own_stream_ = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
    }
    stream = own_stream_;
  }
  // A frame keeps its capture time until all its outputs are out, so outputs
  // sent on a retry still pair with those sent before it.
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
    sent_ = 0U;
  }
  // Frame ids count from 1; 0 means none.
  const holoscan::EmitOptions emit{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U};
  const SyntheticFrame frame =
      synthetic_frame(config_.width, config_.height, emitted_, config_.intrinsics);
  const auto pixels = static_cast<std::size_t>(config_.width) * config_.height;
  const std::size_t depth_bytes = pixels * (config_.millimeters ? 2U : 4U);
  const std::size_t color_bytes = config_.color ? pixels * 3U : 0U;

  // Sends one output; on backpressure the frame is retried on the next tick.
  bool blocked = false;
  const auto settle = [&](holoscan::expected<void, holoscan::Error> sent,
                          std::uint32_t bit) -> holoscan::expected<void, holoscan::Error> {
    if (sent) {
      sent_ |= bit;
      return {};
    }
    if (sent.error().code == holoscan::ErrorCode::kBackpressured) {
      blocked = true;
      return {};
    }
    return sent;
  };

  if (config_.intrinsics && (sent_ & kSentIntrinsics) == 0U) {
    auto loan = intrinsics.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 1>{4}, .dtype = kFloat32});
    if (!loan) {
      return settle(holoscan::make_unexpected(std::move(loan).error()), kSentIntrinsics);
    }
    const CameraIntrinsics& k = frame.intrinsics;
    if (auto written =
            loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
              const std::array<float, 4> values{k.fx, k.fy, k.cx, k.cy};
              std::memcpy(writer.data_as<float>(), values.data(), sizeof(values));
              return {};
            });
        !written) {
      return written;
    }
    if (auto settled = settle(intrinsics.emit(std::move(*loan), emit), kSentIntrinsics);
        !settled || blocked) {
      return settled;
    }
  }

  const bool need_color = config_.color && (sent_ & kSentColor) == 0U;
  const bool need_depth = (sent_ & kSentDepth) == 0U;
  std::optional<holoscan::TensorOutputLoan> depth_loan;
  std::optional<holoscan::TensorOutputLoan> color_loan;
  if (need_depth) {
    auto loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 2>{config_.height, config_.width},
        .dtype = config_.millimeters ? kUInt16 : kFloat32});
    if (!loan) {
      return settle(holoscan::make_unexpected(std::move(loan).error()), kSentDepth);
    }
    depth_loan.emplace(std::move(*loan));
  }
  if (need_color) {
    auto loan = color.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 3>{config_.height, config_.width, 3}, .dtype = kUInt8});
    if (!loan) {
      return settle(holoscan::make_unexpected(std::move(loan).error()), kSentColor);
    }
    color_loan.emplace(std::move(*loan));
  }

  if (config_.device_output) {
    std::optional<holoscan::TensorOutputWriteGuard> depth_writer;
    std::optional<holoscan::TensorOutputWriteGuard> color_writer;
    void* depth_out = nullptr;
    std::uint8_t* color_out = nullptr;
    if (depth_loan) {
      auto writer = depth_loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      depth_writer.emplace(std::move(*writer));
      depth_out = depth_writer->data_as<void>();
    }
    if (color_loan) {
      auto writer = color_loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      color_writer.emplace(std::move(*writer));
      color_out = color_writer->data_as<std::uint8_t>();
    }
    // Depth is always rendered; a scratch target stands in when only the color
    // is still owed.
    if (depth_out == nullptr) {
      if (auto scratch = ensure_scratch(depth_bytes + color_bytes); !scratch) {
        return scratch;
      }
      depth_out = scratch_;
    }
    if (const cudaError_t status =
            render_synthetic(frame, config_.millimeters, depth_out, color_out, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("render_synthetic", status));
    }
    if (stream == own_stream_) {
      if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
        return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
      }
    }
    if (depth_writer) {
      if (auto committed = std::move(*depth_writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    }
    if (color_writer) {
      if (auto committed = std::move(*color_writer).commit(); !committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    }
  } else {
    // Render on the GPU, then fill the host loans from the result.
    if (auto scratch = ensure_scratch(depth_bytes + color_bytes); !scratch) {
      return scratch;
    }
    auto* bytes = static_cast<std::uint8_t*>(scratch_);
    if (const cudaError_t status =
            render_synthetic(frame, config_.millimeters, bytes,
                             config_.color ? bytes + depth_bytes : nullptr, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("render_synthetic", status));
    }
    const auto copy_out = [&](holoscan::TensorOutputLoan& loan, std::size_t offset,
                              std::size_t size) {
      return loan.write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
        if (writer.byte_size() != size) {
          return holoscan::make_unexpected(failure("image loan has an unexpected size"));
        }
        const cudaError_t status = cudaMemcpyAsync(writer.data_as<std::uint8_t>(), bytes + offset,
                                                   size, cudaMemcpyDeviceToHost, stream);
        const cudaError_t synced = status == cudaSuccess ? cudaStreamSynchronize(stream) : status;
        if (synced != cudaSuccess) {
          return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", synced));
        }
        return {};
      });
    };
    if (depth_loan) {
      if (auto copied = copy_out(*depth_loan, 0, depth_bytes); !copied) {
        return copied;
      }
    }
    if (color_loan) {
      if (auto copied = copy_out(*color_loan, depth_bytes, color_bytes); !copied) {
        return copied;
      }
    }
  }

  if (color_loan) {
    if (auto settled =
            settle(color.emit_tensor(std::move(*color_loan),
                                     image_descriptor(config_.width, config_.height,
                                                      holoscan::schema::ImageEncoding_RGB8),
                                     emit),
                   kSentColor);
        !settled || blocked) {
      return settled;
    }
  }
  if (depth_loan) {
    if (auto settled = settle(
            depth.emit_tensor(
                std::move(*depth_loan),
                image_descriptor(config_.width, config_.height,
                                 config_.millimeters ? holoscan::schema::ImageEncoding_MONO16
                                                     : holoscan::schema::ImageEncoding_MONO32F),
                emit),
            kSentDepth);
        !settled || blocked) {
      return settled;
    }
  }
  // Every output of this frame is out.
  pending_capture_.reset();
  ++emitted_;
  return {};
}

holoscan::expected<void, holoscan::Error> SyntheticDepthSourceOp::ensure_scratch(
    std::size_t bytes) {
  if (scratch_ != nullptr) {
    return {};
  }
  if (const cudaError_t status = cudaMalloc(&scratch_, bytes); status != cudaSuccess) {
    scratch_ = nullptr;
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  return {};
}

void validate_config(const DepthFileSourceConfig& config) {
  if (config.files.empty()) {
    throw std::invalid_argument("depth file source needs at least one file");
  }
  if (config.width < 1 || config.height < 1 || config.width > 16384 || config.height > 16384) {
    throw std::invalid_argument("depth file source needs an image size of 1..16384");
  }
  if (config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("depth file source needs a positive period");
  }
  if (!config.loop && config.count > config.files.size()) {
    throw std::invalid_argument("--count exceeds the number of images; add --loop true");
  }
}

DepthFileSourceOp::DepthFileSourceOp(DepthFileSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
  total_ = config_.count == 0 ? static_cast<std::uint32_t>(config_.files.size()) : config_.count;
}

void DepthFileSourceOp::setup(holoscan::OperatorSpec& spec) {
  const auto bytes =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height) * 2U;
  spec.output(depth, "depth")
      .produces_tensor(tensor_spec(holoscan::MemoryKind::kHost, kUInt16, 2U, bytes))
      .payload_options(image_payload());
}

holoscan::Contract DepthFileSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void DepthFileSourceOp::start() { emitted_ = 0; }

holoscan::expected<void, holoscan::Error> DepthFileSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= total_) {
    return {};
  }
  const auto& path = config_.files[emitted_ % config_.files.size()];
  DepthImage frame;
  try {
    frame = read_pgm_depth(path);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (frame.width != config_.width || frame.height != config_.height) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     path.string() + ": every image must be " +
                                                         std::to_string(config_.width) + "x" +
                                                         std::to_string(config_.height)});
  }
  auto loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 2>{config_.height, config_.width}, .dtype = kUInt16});
  if (!loan) {
    return retry_if_backpressured(holoscan::make_unexpected(std::move(loan).error()), emitted_);
  }
  auto written = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                 -> holoscan::expected<void, holoscan::Error> {
    const std::size_t size = frame.pixels.size() * sizeof(std::uint16_t);
    if (writer.byte_size() != size) {
      return holoscan::make_unexpected(failure("image loan has an unexpected size"));
    }
    std::memcpy(writer.data_as<std::uint8_t>(), frame.pixels.data(), size);
    return {};
  });
  if (!written) {
    return written;
  }
  return retry_if_backpressured(
      depth.emit_tensor(
          std::move(*loan),
          image_descriptor(config_.width, config_.height, holoscan::schema::ImageEncoding_MONO16),
          holoscan::EmitOptions{.capture_time = context.activation_time(),
                                .frame_id = emitted_ + 1U}),
      emitted_);
}

}  // namespace depth_to_point_cloud
