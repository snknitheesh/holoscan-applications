// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_sources.hpp"

#include <array>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "projection/cloud_io.hpp"
#include "projection/synthetic_scene.hpp"

namespace point_cloud_to_depth {
namespace {

constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};
constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

// Which outputs of the frame being sent are out.
constexpr std::uint32_t kSentIntrinsics = 1U;
constexpr std::uint32_t kSentColors = 2U;
constexpr std::uint32_t kSentPoints = 4U;

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

using Result = holoscan::expected<void, holoscan::Error>;

// Records the outcome of sending one output of a frame. A consumer still busy
// with earlier frames makes an emit fail with kBackpressured before anything
// is delivered; the source then sends that output again on its next tick,
// with the frame's capture time, rather than failing the run or skipping the
// frame. Returns nothing to go on with the frame's other outputs, or what
// compute() returns now: success when the consumer is busy, else the error.
std::optional<Result> settle(Result sent, std::uint32_t output, std::uint32_t& sent_outputs) {
  if (sent) {
    sent_outputs |= output;
    return std::nullopt;
  }
  if (sent.error().code == holoscan::ErrorCode::kBackpressured) {
    return Result{};
  }
  return Result{holoscan::make_unexpected(std::move(sent).error())};
}

// Fills a host loan with `size` bytes from `bytes` and commits it.
holoscan::expected<void, holoscan::Error> fill_host(holoscan::TensorOutputLoan& loan,
                                                    const void* bytes, std::size_t size) {
  return loan.write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                        -> holoscan::expected<void, holoscan::Error> {
    if (writer.byte_size() != size) {
      return holoscan::make_unexpected(failure("tensor loan has an unexpected size"));
    }
    if (size != 0U) {
      std::memcpy(writer.data_as<std::uint8_t>(), bytes, size);
    }
    return {};
  });
}

std::array<std::int64_t, 2> cloud_shape(std::size_t points) {
  return {static_cast<std::int64_t>(points), 3};
}

}  // namespace

void validate_config(const SyntheticCloudSourceConfig& config) {
  if (config.width < 64 || config.height < 48 || config.width > 4096 || config.height > 4096) {
    throw std::invalid_argument("synthetic images must be 64x48 to 4096x4096");
  }
  if (config.count == 0 || config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("synthetic source needs a positive count and period");
  }
}

SyntheticCloudSourceOp::SyntheticCloudSourceOp(SyntheticCloudSourceConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
  point_count_ = synthetic_cloud_layout(config_.width, config_.height).total();
}

SyntheticCloudSourceOp::~SyntheticCloudSourceOp() { release(); }

void SyntheticCloudSourceOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::MemoryKind memory =
      config_.device_output ? holoscan::MemoryKind::kCudaDevice : holoscan::MemoryKind::kHost;
  spec.output(points, "points")
      .produces_tensor(tensor_spec(memory, kFloat32, 2U, point_count_ * 3ULL * sizeof(float)));
  if (config_.color) {
    spec.output(colors, "colors")
        .produces_tensor(tensor_spec(memory, kUInt8, 2U, point_count_ * 3ULL));
  }
  if (config_.intrinsics) {
    spec.output(intrinsics, "intrinsics")
        .produces_tensor(
            tensor_spec(holoscan::MemoryKind::kHost, kFloat32, 1U, 4U * sizeof(float)));
  }
}

holoscan::Contract SyntheticCloudSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void SyntheticCloudSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
  sent_ = 0U;
}

void SyntheticCloudSourceOp::stop() { release(); }

void SyntheticCloudSourceOp::release() noexcept {
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

holoscan::expected<void, holoscan::Error> SyntheticCloudSourceOp::ensure_scratch() {
  if (scratch_ != nullptr) {
    return {};
  }
  const std::size_t bytes = point_count_ * (3U * sizeof(float) + 3U);
  if (const cudaError_t status = cudaMalloc(&scratch_, bytes); status != cudaSuccess) {
    scratch_ = nullptr;
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  return {};
}

holoscan::expected<void, holoscan::Error> SyntheticCloudSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= config_.count) {
    return {};  // Done: idle until the consumers request completion.
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // The runtime writes device loans only on the stream it assigns to an
    // operator with device outputs. One with host outputs gets none, and
    // makes its clouds on its own stream, waiting for each.
    if (config_.device_output) {
      return holoscan::make_unexpected(failure(
          "SyntheticCloudSourceOp: the runtime gave no CUDA stream for the device outputs"));
    }
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
  const std::size_t point_bytes = point_count_ * 3U * sizeof(float);
  const std::size_t color_bytes = point_count_ * 3U;

  if (config_.intrinsics && (sent_ & kSentIntrinsics) == 0U) {
    auto loan = intrinsics.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = std::array<std::int64_t, 1>{4}, .dtype = kFloat32});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    const CameraIntrinsics& k = frame.intrinsics;
    const std::array<float, 4> values{k.fx, k.fy, k.cx, k.cy};
    if (auto written = fill_host(*loan, values.data(), sizeof(values)); !written) {
      return written;
    }
    if (auto stop = settle(intrinsics.emit(std::move(*loan), emit), kSentIntrinsics, sent_)) {
      return *stop;
    }
  }

  const bool need_colors = config_.color && (sent_ & kSentColors) == 0U;
  const bool need_points = (sent_ & kSentPoints) == 0U;
  std::optional<holoscan::TensorOutputLoan> points_loan;
  std::optional<holoscan::TensorOutputLoan> colors_loan;
  if (need_points) {
    auto loan = points.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = cloud_shape(point_count_), .dtype = kFloat32});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    points_loan.emplace(std::move(*loan));
  }
  if (need_colors) {
    auto loan = colors.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = cloud_shape(point_count_), .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    colors_loan.emplace(std::move(*loan));
  }

  if (config_.device_output) {
    std::optional<holoscan::TensorOutputWriteGuard> points_writer;
    std::optional<holoscan::TensorOutputWriteGuard> colors_writer;
    if (points_loan) {
      auto writer = points_loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      points_writer.emplace(std::move(*writer));
    }
    if (colors_loan) {
      auto writer = colors_loan->write(stream);
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      colors_writer.emplace(std::move(*writer));
    }
    // Only the outputs still owed are made.
    if (const cudaError_t status = generate_synthetic_cloud(
            frame, config_.millimeter_depth,
            points_writer ? points_writer->data_as<float>() : nullptr,
            colors_writer ? colors_writer->data_as<std::uint8_t>() : nullptr, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("generate_synthetic_cloud", status));
    }
    for (auto* writer : {&points_writer, &colors_writer}) {
      if (*writer) {
        if (auto committed = std::move(**writer).commit(); !committed) {
          return holoscan::make_unexpected(std::move(committed).error());
        }
      }
    }
  } else {
    // Make the cloud on the GPU, then fill the host loans from it.
    if (auto scratch = ensure_scratch(); !scratch) {
      return scratch;
    }
    auto* bytes = static_cast<std::uint8_t*>(scratch_);
    if (const cudaError_t status =
            generate_synthetic_cloud(frame, config_.millimeter_depth,
                                     reinterpret_cast<float*>(bytes), bytes + point_bytes, stream);
        status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("generate_synthetic_cloud", status));
    }
    const auto copy_out = [&](holoscan::TensorOutputLoan& loan, std::size_t offset,
                              std::size_t size) {
      return loan.write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                            -> holoscan::expected<void, holoscan::Error> {
        if (writer.byte_size() != size) {
          return holoscan::make_unexpected(failure("tensor loan has an unexpected size"));
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
    if (points_loan) {
      if (auto copied = copy_out(*points_loan, 0, point_bytes); !copied) {
        return copied;
      }
    }
    if (colors_loan) {
      if (auto copied = copy_out(*colors_loan, point_bytes, color_bytes); !copied) {
        return copied;
      }
    }
  }

  if (colors_loan) {
    if (auto stop = settle(colors.emit(std::move(*colors_loan), emit), kSentColors, sent_)) {
      return *stop;
    }
  }
  if (points_loan) {
    if (auto stop = settle(points.emit(std::move(*points_loan), emit), kSentPoints, sent_)) {
      return *stop;
    }
  }
  // Every output of this frame is out.
  pending_capture_.reset();
  ++emitted_;
  return {};
}

void validate_config(const CloudFileSourceConfig& config) {
  if (config.files.empty()) {
    throw std::invalid_argument("point cloud file source needs at least one file");
  }
  if (config.max_points < 1 || config.max_points > kMaxPlyPoints) {
    throw std::invalid_argument("point cloud file source needs max_points of 1.." +
                                std::to_string(kMaxPlyPoints));
  }
  if (config.period <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("point cloud file source needs a positive period");
  }
  if (!config.loop && config.count > config.files.size()) {
    throw std::invalid_argument("--count exceeds the number of files; add --loop true");
  }
}

CloudFileSourceOp::CloudFileSourceOp(CloudFileSourceConfig config) : config_(std::move(config)) {
  validate_config(config_);
  total_ = config_.count == 0 ? static_cast<std::uint32_t>(config_.files.size()) : config_.count;
}

void CloudFileSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(points, "points")
      .produces_tensor(tensor_spec(holoscan::MemoryKind::kHost, kFloat32, 2U,
                                   config_.max_points * 3U * sizeof(float)));
  if (config_.color) {
    spec.output(colors, "colors")
        .produces_tensor(
            tensor_spec(holoscan::MemoryKind::kHost, kUInt8, 2U, config_.max_points * 3U));
  }
}

holoscan::Contract CloudFileSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = config_.period});
  return result;
}

void CloudFileSourceOp::start() {
  emitted_ = 0;
  pending_capture_.reset();
  sent_ = 0U;
}

holoscan::expected<void, holoscan::Error> CloudFileSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (emitted_ >= total_) {
    return {};
  }
  const auto& path = config_.files[emitted_ % config_.files.size()];
  PointCloud cloud;
  try {
    cloud = read_ply(path);
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(failure(error.what()));
  }
  if (cloud.size() > config_.max_points) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        path.string() + ": more than " + std::to_string(config_.max_points) +
                            " points, the most the first look at the files found"});
  }
  if (config_.color && !cloud.color) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, path.string() + ": has no colors"});
  }
  if (!pending_capture_) {
    pending_capture_ = context.activation_time();
    sent_ = 0U;
  }
  const holoscan::EmitOptions emit{.capture_time = *pending_capture_, .frame_id = emitted_ + 1U};
  if (config_.color && (sent_ & kSentColors) == 0U) {
    auto loan = colors.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = cloud_shape(cloud.size()), .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    if (auto written = fill_host(*loan, cloud.rgb.data(), cloud.rgb.size()); !written) {
      return written;
    }
    if (auto stop = settle(colors.emit(std::move(*loan), emit), kSentColors, sent_)) {
      return *stop;
    }
  }
  auto loan = points.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = cloud_shape(cloud.size()), .dtype = kFloat32});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  if (auto written = fill_host(*loan, cloud.xyz.data(), cloud.xyz.size() * sizeof(float));
      !written) {
    return written;
  }
  if (auto stop = settle(points.emit(std::move(*loan), emit), kSentPoints, sent_)) {
    return *stop;
  }
  pending_capture_.reset();
  ++emitted_;
  return {};
}

}  // namespace point_cloud_to_depth
