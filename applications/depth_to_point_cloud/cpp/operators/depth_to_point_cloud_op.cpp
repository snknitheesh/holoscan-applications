// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "depth_to_point_cloud_op.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace depth_to_point_cloud {
namespace {

using std::chrono_literals::operator""ns;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "DepthToPointCloudOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string("DepthToPointCloudOp: ") + what + ": " + cudaGetErrorString(status)};
}

holoscan::TensorOutputSpec device_tensor(DLDataType dtype, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .rank = 3U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds = holoscan::tensor_bounds(bytes),
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// Whether the tensor's rows are packed: each element directly after the last.
bool packed(const holoscan::Tensor& tensor, std::size_t element_bytes) {
  const std::vector<std::int64_t> strides = tensor.strides();
  const std::span<const std::int64_t> shape = tensor.shape_span();
  std::int64_t expected = static_cast<std::int64_t>(element_bytes);
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    if (shape[axis] > 1 && strides[axis] != expected) {
      return false;
    }
    expected *= shape[axis];
  }
  return true;
}

// Waits for the producer of an input that only this operator's stream reads.
// The runtime has already ordered device, managed and stream-ready pinned
// memory on that stream before compute(), which is all the GPU needs; plain
// host memory it does not order, so wait for that here. (An input the host
// reads itself needs synchronize_for_host() in every case.)
holoscan::expected<void, holoscan::Error> wait_for_host(const holoscan::Tensor& tensor) {
  if (tensor.device().device_type != kDLCPU) {
    return {};
  }
  return tensor.synchronize_for_host();
}

// Whether a GPU-resident tensor's data can be read in place as elements of
// `alignment` bytes.
bool aligned(const holoscan::Tensor& tensor, std::size_t alignment) {
  return !on_gpu(tensor) || reinterpret_cast<std::uintptr_t>(tensor.data()) % alignment == 0U;
}

}  // namespace

namespace detail {

holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  // The memory cannot wait for the stream, so wait here.
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  return {};
}

}  // namespace detail

void validate_config(const DepthToPointCloudConfig& config) {
  if (config.intrinsics) {
    const CameraIntrinsics& k = *config.intrinsics;
    if (!(std::isfinite(k.fx) && k.fx > 0 && std::isfinite(k.fy) && k.fy > 0 &&
          std::isfinite(k.cx) && std::isfinite(k.cy))) {
      throw std::invalid_argument("intrinsics need finite, positive fx and fy and finite cx, cy");
    }
    if (config.intrinsics_input) {
      // Per-frame intrinsics are paired with each depth image; nothing would
      // fall back to configured ones, so refuse a setting that would be ignored.
      throw std::invalid_argument("give intrinsics or enable the intrinsics input, not both");
    }
  } else if (!config.intrinsics_input) {
    throw std::invalid_argument("give intrinsics or enable the intrinsics input");
  }
  if (!(std::isfinite(config.depth_scale) && config.depth_scale > 0)) {
    throw std::invalid_argument("depth_scale must be finite and positive");
  }
  if (!(config.depth_min >= 0 && config.depth_max > config.depth_min)) {
    throw std::invalid_argument("depth range needs 0 <= depth_min < depth_max");
  }
  if (config.max_width < 1 || config.max_height < 1 || config.max_width > 16384 ||
      config.max_height > 16384) {
    throw std::invalid_argument("max_width and max_height must be 1..16384");
  }
}

DepthToPointCloudOp::DepthToPointCloudOp(DepthToPointCloudConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
}

DepthToPointCloudOp::~DepthToPointCloudOp() { release(); }

void DepthToPointCloudOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels = static_cast<std::uint64_t>(config_.max_width) *
                      static_cast<std::uint64_t>(config_.max_height);
  // Two slots per input: one frame in compute() while the next one arrives.
  spec.input(depth, "depth").queue_depth(2U);
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (config_.color) {
    spec.input(color, "color").queue_depth(2U);
  }
  if (config_.intrinsics_input) {
    spec.input(intrinsics, "intrinsics").queue_depth(2U);
  }
  spec.output(point_cloud, "point_cloud")
      .produces_tensor(device_tensor(kFloat32, pixels * 3U * sizeof(float)));
  if (config_.color) {
    spec.output(colors, "colors").produces_tensor(device_tensor(kUInt8, pixels * 3U));
  }
}

holoscan::Contract DepthToPointCloudOp::contract() const {
  holoscan::Contract result;
  if (!config_.color && !config_.intrinsics_input) {
    result.trigger(holoscan::OnEach{depth});
    return result;
  }
  // Every input of one frame carries the same capture time.
  std::vector<holoscan::PortRef> ports{depth};
  if (config_.color) ports.emplace_back(color);
  if (config_.intrinsics_input) ports.emplace_back(intrinsics);
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(depth, holoscan::Exact{});
  if (config_.color) result.input(color, holoscan::Exact{});
  if (config_.intrinsics_input) result.input(intrinsics, holoscan::Exact{});
  return result;
}

void DepthToPointCloudOp::stop() { release(); }

void DepthToPointCloudOp::release() noexcept {
  // Finish the work that may still use the staging buffers before freeing them.
  if (own_stream_ != nullptr) {
    cudaStreamSynchronize(own_stream_);
    cudaStreamDestroy(own_stream_);
    own_stream_ = nullptr;
  }
  for (Staging* staging : {&depth_staging_, &color_staging_}) {
    if (staging->data != nullptr) {
      cudaFree(staging->data);
    }
    *staging = {};
  }
}

holoscan::expected<const void*, holoscan::Error> DepthToPointCloudOp::on_device(
    const holoscan::Tensor& tensor, std::size_t bytes, Staging& staging, cudaStream_t stream) {
  if (on_gpu(tensor)) {
    return tensor.data();
  }
  if (staging.bytes < bytes) {
    // Free only after the stream is done with the old buffer.
    cudaStreamSynchronize(stream);
    cudaFree(staging.data);
    staging = {};
    if (const cudaError_t status = cudaMalloc(&staging.data, bytes); status != cudaSuccess) {
      staging.data = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    staging.bytes = bytes;
  }
  if (const cudaError_t status =
          cudaMemcpyAsync(staging.data, tensor.data(), bytes, cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  // The runtime orders the release of an input the GPU reads after this
  // operator's stream, but not of one in host memory, so a producer could
  // reuse a pinned buffer while the copy still reads it. Pageable memory needs
  // nothing: cudaMemcpyAsync has staged it by the time it returns.
  if (tensor.device().device_type != kDLCUDAHost) {
    return static_cast<const void*>(staging.data);
  }
  if (stream == own_stream_) {
    // Destroyed at stop(), so no allocator may wait for this stream later.
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
  } else if (auto held = detail::hold_until_done(tensor, stream); !held) {
    return holoscan::make_unexpected(std::move(held).error());
  }
  return static_cast<const void*>(staging.data);
}

holoscan::expected<void, holoscan::Error> DepthToPointCloudOp::compute(
    holoscan::ExecutionContext& context) {
  auto depth_sample = depth.receive();
  if (!depth_sample) {
    return holoscan::make_unexpected(std::move(depth_sample).error());
  }
  if (!depth_sample->data.data) {
    return holoscan::make_unexpected(invalid("depth image has no tensor"));
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    // Not reached in a compiled graph: the outputs are in device memory, so
    // the runtime always assigns a stream. Kept so the operator never falls
    // back to the legacy default stream, which serializes with every other
    // stream in the process.
    if (own_stream_ == nullptr) {
      if (const cudaError_t status = cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking);
          status != cudaSuccess) {
        own_stream_ = nullptr;
        return holoscan::make_unexpected(cuda_failure("cudaStreamCreate", status));
      }
    }
    stream = own_stream_;
  }

  // The depth image.
  const holoscan::schema::ImageT& depth_image = depth_sample->data;
  const holoscan::Tensor& depth_tensor = *depth_image.data;
  DeprojectParams params;
  std::size_t depth_bytes = 0;
  if (depth_image.encoding == holoscan::schema::ImageEncoding_MONO16) {
    params.depth_type = DepthType::kUint16;
    depth_bytes = sizeof(std::uint16_t);
  } else if (depth_image.encoding == holoscan::schema::ImageEncoding_MONO32F) {
    params.depth_type = DepthType::kFloat32;
    depth_bytes = sizeof(float);
  } else {
    return holoscan::make_unexpected(invalid(
        std::string("unsupported depth encoding ") +
        holoscan::schema::EnumNameImageEncoding(depth_image.encoding) + "; use MONO16 or MONO32F"));
  }
  const DLDataType dtype = depth_tensor.dtype();
  const bool dtype_ok = params.depth_type == DepthType::kUint16
                            ? dtype.code == kDLUInt && dtype.bits == 16 && dtype.lanes == 1
                            : dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
  const std::span<const std::int64_t> shape = depth_tensor.shape_span();
  if (!dtype_ok) {
    return holoscan::make_unexpected(
        invalid("depth tensor element type disagrees with its encoding"));
  }
  if (!(shape.size() == 2 || (shape.size() == 3 && shape[2] == 1))) {
    return holoscan::make_unexpected(invalid("depth tensor must be [H, W] or [H, W, 1]"));
  }
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);
  if (width < 1 || height < 1 || width > config_.max_width || height > config_.max_height) {
    return holoscan::make_unexpected(invalid("depth image " + std::to_string(width) + "x" +
                                             std::to_string(height) + " exceeds the configured " +
                                             std::to_string(config_.max_width) + "x" +
                                             std::to_string(config_.max_height)));
  }
  if (!packed(depth_tensor, depth_bytes)) {
    return holoscan::make_unexpected(invalid("depth rows must be packed"));
  }
  if (!aligned(depth_tensor, depth_bytes)) {
    return holoscan::make_unexpected(invalid("depth data must be aligned to its element size"));
  }
  if (auto ready = wait_for_host(depth_tensor); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }
  const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

  // The frame's own intrinsics, or the configured ones. The sample stays
  // alive until compute() returns, as the kernel may read it in place.
  std::optional<holoscan::Sample<holoscan::Tensor>> intrinsics_sample;
  if (config_.intrinsics_input) {
    auto sample = intrinsics.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    intrinsics_sample.emplace(std::move(*sample));
    const holoscan::Tensor& tensor = intrinsics_sample->data;
    const DLDataType idt = tensor.dtype();
    if (idt.code != kDLFloat || idt.bits != 32 || tensor.size() != 4 || !packed(tensor, 4)) {
      return holoscan::make_unexpected(invalid("intrinsics must be float32 [fx, fy, cx, cy]"));
    }
    if (!aligned(tensor, sizeof(float))) {
      return holoscan::make_unexpected(invalid("intrinsics data must be aligned to 4 bytes"));
    }
    if (on_gpu(tensor)) {
      // The kernel reads them where they are, so the host never waits for them.
      params.device_intrinsics = static_cast<const float*>(tensor.data());
    } else {
      // The host reads them, so it waits for their producer, pinned or not.
      if (auto ready = tensor.synchronize_for_host(); !ready) {
        return holoscan::make_unexpected(std::move(ready).error());
      }
      std::array<float, 4> k{};
      std::memcpy(k.data(), tensor.data(), sizeof(k));
      params.intrinsics = {k[0], k[1], k[2], k[3]};
    }
  } else {
    params.intrinsics = *config_.intrinsics;
  }
  if (params.device_intrinsics == nullptr &&
      !(params.intrinsics.fx > 0 && params.intrinsics.fy > 0 &&
        std::isfinite(params.intrinsics.fx) && std::isfinite(params.intrinsics.fy) &&
        std::isfinite(params.intrinsics.cx) && std::isfinite(params.intrinsics.cy))) {
    return holoscan::make_unexpected(
        invalid("intrinsics need finite, positive fx and fy and finite cx, cy"));
  }
  params.depth_scale = config_.depth_scale;
  params.depth_min = config_.depth_min;
  params.depth_max = config_.depth_max;
  params.invalid_value = config_.invalid_value;

  // The color image.
  ColorInput color_input;
  std::optional<holoscan::Sample<holoscan::schema::ImageT>> color_sample;
  if (config_.color) {
    auto sample = color.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    color_sample.emplace(std::move(*sample));
    const holoscan::schema::ImageT& image = color_sample->data;
    if (!image.data) {
      return holoscan::make_unexpected(invalid("color image has no tensor"));
    }
    switch (image.encoding) {
      case holoscan::schema::ImageEncoding_RGB8:
        color_input = {nullptr, 3, false};
        break;
      case holoscan::schema::ImageEncoding_BGR8:
        color_input = {nullptr, 3, true};
        break;
      case holoscan::schema::ImageEncoding_RGBA8:
        color_input = {nullptr, 4, false};
        break;
      case holoscan::schema::ImageEncoding_BGRA8:
        color_input = {nullptr, 4, true};
        break;
      default:
        return holoscan::make_unexpected(
            invalid(std::string("unsupported color encoding ") +
                    holoscan::schema::EnumNameImageEncoding(image.encoding) +
                    "; use RGB8, BGR8, RGBA8 or BGRA8"));
    }
    const holoscan::Tensor& tensor = *image.data;
    const DLDataType cdt = tensor.dtype();
    const std::span<const std::int64_t> cshape = tensor.shape_span();
    if (cdt.code != kDLUInt || cdt.bits != 8 || cshape.size() != 3 ||
        cshape[2] != color_input.channels) {
      return holoscan::make_unexpected(
          invalid("color tensor must be uint8 [H, W, channels] matching its encoding"));
    }
    if (cshape[0] != height || cshape[1] != width) {
      return holoscan::make_unexpected(invalid("color image size must match the depth image"));
    }
    if (!packed(tensor, 1)) {
      return holoscan::make_unexpected(invalid("color rows must be packed"));
    }
    // Four channels are read as one uchar4.
    if (!aligned(tensor, color_input.channels == 4 ? 4U : 1U)) {
      return holoscan::make_unexpected(
          invalid("color data must be aligned to 4 bytes for four channels"));
    }
    if (auto ready = wait_for_host(tensor); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    auto data = on_device(tensor, pixels * static_cast<std::size_t>(color_input.channels),
                          color_staging_, stream);
    if (!data) {
      return holoscan::make_unexpected(std::move(data).error());
    }
    color_input.data = *data;
  }
  auto depth_data = on_device(depth_tensor, pixels * depth_bytes, depth_staging_, stream);
  if (!depth_data) {
    return holoscan::make_unexpected(std::move(depth_data).error());
  }

  // The outputs.
  const std::array<std::int64_t, 3> out_shape{height, width, 3};
  auto cloud_loan = point_cloud.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = out_shape, .dtype = kFloat32});
  if (!cloud_loan) {
    return holoscan::make_unexpected(std::move(cloud_loan).error());
  }
  auto cloud_writer = cloud_loan->write(stream);
  if (!cloud_writer) {
    return holoscan::make_unexpected(std::move(cloud_writer).error());
  }
  std::optional<holoscan::TensorOutputLoan> color_loan;
  std::optional<holoscan::TensorOutputWriteGuard> color_writer;
  if (config_.color) {
    auto loan =
        colors.allocate_tensor(holoscan::TensorLoanRequest{.shape = out_shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    color_loan.emplace(std::move(*loan));
    auto writer = color_loan->write(stream);
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    color_writer.emplace(std::move(*writer));
  }

  if (const cudaError_t status = launch_deproject(
          *depth_data, params, color_input, cloud_writer->data_as<float3>(),
          color_writer ? color_writer->data_as<uchar3>() : nullptr, width, height, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("deproject kernel", status));
  }
  if (stream == own_stream_) {
    // The runtime knows nothing of this stream; finish before the outputs go.
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
  }

  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*depth_sample);
  if (auto committed = std::move(*cloud_writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  if (auto sent = point_cloud.emit(std::move(*cloud_loan), emit); !sent) {
    return sent;
  }
  if (config_.color) {
    if (auto committed = std::move(*color_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return colors.emit(std::move(*color_loan), emit);
  }
  return {};
}

}  // namespace depth_to_point_cloud
