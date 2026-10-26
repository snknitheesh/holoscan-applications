// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_to_depth_op.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
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

namespace point_cloud_to_depth {
namespace {

using std::chrono_literals::operator""ns;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kUInt16{kDLUInt, 16U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "PointCloudToDepthOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string("PointCloudToDepthOp: ") + what + ": " + cudaGetErrorString(status)};
}

holoscan::TensorOutputSpec device_image(DLDataType dtype, std::uint32_t rank, std::uint64_t bytes) {
  return holoscan::TensorOutputSpec{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = dtype,
              .rank = rank,
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

// An image descriptor whose header carries the cloud's capture time, 0 when
// it has none, and the camera's coordinate frame.
holoscan::schema::ImageT image_descriptor(int width, int height,
                                          holoscan::schema::ImageEncoding encoding,
                                          const holoscan::SampleMetadata& cloud,
                                          const std::string& frame_id) {
  holoscan::schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  if (cloud.source_clock_id.valid()) {
    image.header->capture_timestamp_ns = cloud.capture_timestamp_ns;
  }
  image.header->frame_id = frame_id;
  return image;
}

bool is_dtype(const DLDataType& dtype, const DLDataType& expected) {
  return dtype.code == expected.code && dtype.bits == expected.bits &&
         dtype.lanes == expected.lanes;
}

bool on_gpu(const holoscan::Tensor& tensor) {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

// Whether the tensor is packed: each element directly after the last. Unlike
// Tensor::is_contiguous(), any stride is accepted on an axis of extent 1,
// which no element steps along (as in a [1, 3] slice of a larger cloud).
bool packed(const holoscan::Tensor& tensor) {
  const std::span<const std::int64_t> strides = tensor.dlpack_strides_span();
  if (strides.empty()) {
    return true;  // DLPack's compact row-major layout.
  }
  const std::span<const std::int64_t> shape = tensor.shape_span();
  std::int64_t expected = 1;  // In elements, as DLPack counts strides.
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

// Whether a tensor in device memory is on `device`, where the kernels run.
// Managed memory is reachable from every device.
bool on_device_of(const holoscan::Tensor& tensor, int device) {
  const DLDevice where = tensor.device();
  return where.device_type != kDLCUDA || where.device_id == device;
}

// Whether a GPU-resident tensor's data can be read in place as elements of
// `alignment` bytes.
bool aligned(const holoscan::Tensor& tensor, std::size_t alignment) {
  return !on_gpu(tensor) || reinterpret_cast<std::uintptr_t>(tensor.data()) % alignment == 0U;
}

bool usable(const CameraIntrinsics& k) {
  return std::isfinite(k.fx) && k.fx > 0 && std::isfinite(k.fy) && k.fy > 0 &&
         std::isfinite(k.cx) && std::isfinite(k.cy);
}

// The number of points in a float32 [N, 3] or [H, W, 3] cloud, or an error.
holoscan::expected<std::uint64_t, holoscan::Error> point_count(const holoscan::Tensor& tensor) {
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (!is_dtype(tensor.dtype(), kFloat32) || (shape.size() != 2 && shape.size() != 3) ||
      shape.back() != 3) {
    return holoscan::make_unexpected(invalid("points must be float32 [N, 3] or [H, W, 3]"));
  }
  std::uint64_t count = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    if (shape[axis] < 0) {
      return holoscan::make_unexpected(invalid("points have a negative extent"));
    }
    // A product that wrapped around could pass for a small cloud.
    const auto extent = static_cast<std::uint64_t>(shape[axis]);
    if (extent != 0 && count > std::numeric_limits<std::uint64_t>::max() / extent) {
      return holoscan::make_unexpected(invalid("points have more elements than can be counted"));
    }
    count *= extent;
  }
  return count;
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

void validate_config(const PointCloudToDepthConfig& config) {
  if (config.width < 1 || config.height < 1 || config.width > kMaxImageSide ||
      config.height > kMaxImageSide) {
    throw std::invalid_argument("width and height must be 1.." + std::to_string(kMaxImageSide));
  }
  if (config.intrinsics) {
    if (!usable(*config.intrinsics)) {
      throw std::invalid_argument("intrinsics need finite, positive fx and fy and finite cx, cy");
    }
    if (config.intrinsics_input) {
      // Per-frame intrinsics are paired with each cloud; nothing would fall
      // back to configured ones, so refuse a setting that would be ignored.
      throw std::invalid_argument("give intrinsics or enable the intrinsics input, not both");
    }
  } else if (!config.intrinsics_input) {
    throw std::invalid_argument("give intrinsics or enable the intrinsics input");
  }
  if (config.extrinsics &&
      !std::all_of(std::begin(config.extrinsics->matrix), std::end(config.extrinsics->matrix),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("extrinsics must be finite");
  }
  if (!(config.depth_min >= 0 && config.depth_max > config.depth_min)) {
    throw std::invalid_argument("depth range needs 0 <= depth_min < depth_max");
  }
  if (!(std::isfinite(config.depth_scale) && config.depth_scale > 0)) {
    throw std::invalid_argument("depth_scale must be finite and positive");
  }
  if (config.depth_format != DepthFormat::kFloat32 && config.depth_format != DepthFormat::kUint16) {
    throw std::invalid_argument("depth_format must be float32 or uint16");
  }
  if (!valid_point_size(config.point_size)) {
    throw std::invalid_argument("point_size must be 1, 3 or 5");
  }
  if (config.max_points < 1 || config.max_points > kMaxPointsLimit) {
    throw std::invalid_argument("max_points must be 1.." + std::to_string(kMaxPointsLimit));
  }
}

PointCloudToDepthOp::PointCloudToDepthOp(PointCloudToDepthConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
  params_.width = config_.width;
  params_.height = config_.height;
  if (config_.intrinsics) {
    params_.intrinsics = *config_.intrinsics;
  }
  if (config_.extrinsics) {
    params_.extrinsics = *config_.extrinsics;
  }
  params_.depth_min = config_.depth_min;
  params_.depth_max = config_.depth_max;
  params_.format = config_.depth_format;
  params_.depth_scale = config_.depth_scale;
  params_.point_size = config_.point_size;
}

PointCloudToDepthOp::~PointCloudToDepthOp() { release(); }

void PointCloudToDepthOp::setup(holoscan::OperatorSpec& spec) {
  const auto pixels =
      static_cast<std::uint64_t>(config_.width) * static_cast<std::uint64_t>(config_.height);
  const bool uint16 = config_.depth_format == DepthFormat::kUint16;
  // Two slots per input: one cloud in compute() while the next one arrives.
  spec.input(points, "points").queue_depth(2U);
  // Optional ports exist only when configured: an unconnected input would
  // never be satisfied, and an emit into an unconnected output fails.
  if (config_.color) {
    spec.input(colors, "colors").queue_depth(2U);
  }
  if (config_.intrinsics_input) {
    spec.input(intrinsics, "intrinsics").queue_depth(2U);
  }
  spec.output(depth, "depth")
      .produces_tensor(device_image(uint16 ? kUInt16 : kFloat32, 2U,
                                    pixels * (uint16 ? sizeof(std::uint16_t) : sizeof(float))))
      .payload_options(image_payload());
  if (config_.color) {
    spec.output(color, "color")
        .produces_tensor(device_image(kUInt8, 3U, pixels * 3U))
        .payload_options(image_payload());
  }
}

holoscan::Contract PointCloudToDepthOp::contract() const {
  holoscan::Contract result;
  if (!config_.color && !config_.intrinsics_input) {
    result.trigger(holoscan::OnEach{points});
    return result;
  }
  // Every input of one frame carries the same capture time.
  std::vector<holoscan::PortRef> ports{points};
  if (config_.color) ports.emplace_back(colors);
  if (config_.intrinsics_input) ports.emplace_back(intrinsics);
  result.trigger(holoscan::OnAligned{.ports = holoscan::PortList{ports}, .window = 0ns});
  result.input(points, holoscan::Exact{});
  if (config_.color) result.input(colors, holoscan::Exact{});
  if (config_.intrinsics_input) result.input(intrinsics, holoscan::Exact{});
  return result;
}

void PointCloudToDepthOp::stop() { release(); }

void PointCloudToDepthOp::release() noexcept {
  // cudaFree waits for the work on the runtime's streams that may still use
  // the buffers.
  for (DeviceBuffer* buffer : {&zbuffer_, &points_staging_, &colors_staging_}) {
    if (buffer->data != nullptr) {
      cudaFree(buffer->data);
    }
    *buffer = {};
  }
}

holoscan::expected<cudaStream_t, holoscan::Error> PointCloudToDepthOp::stream_for(
    holoscan::ExecutionContext& context) {
  // The runtime assigns a stream to an operator with device outputs, and
  // writes their loans on that stream only, so there is no other to use.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        "PointCloudToDepthOp: the runtime gave no CUDA stream for the device outputs"});
  }
  return stream;
}

holoscan::expected<void*, holoscan::Error> PointCloudToDepthOp::reserve(DeviceBuffer& buffer,
                                                                        std::size_t bytes,
                                                                        cudaStream_t stream) {
  if (buffer.bytes >= bytes) {
    return buffer.data;
  }
  // Grows at most a few times over a run, doubling so that slowly growing
  // clouds do not reallocate every frame. The old buffer is freed only after
  // the stream is done with it.
  const std::size_t grown = std::max(bytes, buffer.bytes * 2U);
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  cudaFree(buffer.data);
  buffer = {};
  if (const cudaError_t status = cudaMalloc(&buffer.data, grown); status != cudaSuccess) {
    buffer.data = nullptr;
    return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
  }
  buffer.bytes = grown;
  return buffer.data;
}

holoscan::expected<const void*, holoscan::Error> PointCloudToDepthOp::on_device(
    const holoscan::Tensor& tensor, std::size_t bytes, DeviceBuffer& staging, cudaStream_t stream) {
  if (on_gpu(tensor)) {
    return tensor.data();
  }
  auto buffer = reserve(staging, bytes, stream);
  if (!buffer) {
    return holoscan::make_unexpected(std::move(buffer).error());
  }
  if (const cudaError_t status =
          cudaMemcpyAsync(*buffer, tensor.data(), bytes, cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpyAsync", status));
  }
  // The runtime orders the release of device, managed and stream-ready pinned
  // inputs after this operator's stream, but not of host-written pinned
  // memory, whose producer could reuse it while the copy still reads it: hold
  // that here (stream-ready pinned memory is held twice, harmlessly).
  // Pageable memory needs nothing: cudaMemcpyAsync has staged it, after a
  // possible wait on the stream, by the time it returns.
  if (tensor.device().device_type != kDLCUDAHost) {
    return static_cast<const void*>(*buffer);
  }
  if (auto held = detail::hold_until_done(tensor, stream); !held) {
    return holoscan::make_unexpected(std::move(held).error());
  }
  return static_cast<const void*>(*buffer);
}

holoscan::expected<void, holoscan::Error> PointCloudToDepthOp::compute(
    holoscan::ExecutionContext& context) {
  auto cloud = points.receive();
  if (!cloud) {
    return holoscan::make_unexpected(std::move(cloud).error());
  }
  auto stream_or = stream_for(context);
  if (!stream_or) {
    return holoscan::make_unexpected(std::move(stream_or).error());
  }
  cudaStream_t stream = *stream_or;
  // The kernels run on the stream's device, which device inputs must be on.
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }

  // The points.
  const holoscan::Tensor& point_tensor = cloud->data;
  auto count_or = point_count(point_tensor);
  if (!count_or) {
    return holoscan::make_unexpected(std::move(count_or).error());
  }
  const std::uint64_t count = *count_or;
  if (count > config_.max_points) {
    return holoscan::make_unexpected(invalid(std::to_string(count) +
                                             " points exceed the configured maximum of " +
                                             std::to_string(config_.max_points)));
  }
  if (!packed(point_tensor)) {
    return holoscan::make_unexpected(invalid("points must be packed"));
  }
  if (!aligned(point_tensor, sizeof(float))) {
    return holoscan::make_unexpected(invalid("point data must be aligned to 4 bytes"));
  }
  if (!on_device_of(point_tensor, device)) {
    return holoscan::make_unexpected(invalid("points are on another GPU than the operator"));
  }
  if (auto ready = wait_for_host(point_tensor); !ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }

  // The frame's own intrinsics, or the configured ones. The sample stays
  // alive until compute() returns, as the kernel may read it in place.
  ProjectParams params = params_;
  std::optional<holoscan::Sample<holoscan::Tensor>> intrinsics_sample;
  if (config_.intrinsics_input) {
    auto sample = intrinsics.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    intrinsics_sample.emplace(std::move(*sample));
    const holoscan::Tensor& tensor = intrinsics_sample->data;
    if (!is_dtype(tensor.dtype(), kFloat32) || tensor.size() != 4 || !packed(tensor)) {
      return holoscan::make_unexpected(invalid("intrinsics must be float32 [fx, fy, cx, cy]"));
    }
    if (!aligned(tensor, sizeof(float))) {
      return holoscan::make_unexpected(invalid("intrinsics data must be aligned to 4 bytes"));
    }
    if (!on_device_of(tensor, device)) {
      return holoscan::make_unexpected(invalid("intrinsics are on another GPU than the operator"));
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
      if (!usable(params.intrinsics)) {
        return holoscan::make_unexpected(
            invalid("intrinsics need finite, positive fx and fy and finite cx, cy"));
      }
    }
  }

  // The colors.
  const std::uint8_t* color_data = nullptr;
  std::optional<holoscan::Sample<holoscan::Tensor>> colors_sample;
  if (config_.color) {
    auto sample = colors.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    colors_sample.emplace(std::move(*sample));
    const holoscan::Tensor& tensor = colors_sample->data;
    const std::span<const std::int64_t> shape = tensor.shape_span();
    const std::span<const std::int64_t> point_shape = point_tensor.shape_span();
    if (!is_dtype(tensor.dtype(), kUInt8) ||
        !std::equal(shape.begin(), shape.end(), point_shape.begin(), point_shape.end())) {
      return holoscan::make_unexpected(invalid("colors must be uint8 RGB of the points' shape"));
    }
    if (!packed(tensor)) {
      return holoscan::make_unexpected(invalid("colors must be packed"));
    }
    if (!on_device_of(tensor, device)) {
      return holoscan::make_unexpected(invalid("colors are on another GPU than the operator"));
    }
    if (auto ready = wait_for_host(tensor); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    if (count != 0) {
      auto data = on_device(tensor, count * 3U, colors_staging_, stream);
      if (!data) {
        return holoscan::make_unexpected(std::move(data).error());
      }
      color_data = static_cast<const std::uint8_t*>(*data);
    }
  }
  const float* point_data = nullptr;
  if (count != 0) {
    auto data = on_device(point_tensor, count * 3U * sizeof(float), points_staging_, stream);
    if (!data) {
      return holoscan::make_unexpected(std::move(data).error());
    }
    point_data = static_cast<const float*>(*data);
  }
  auto zbuffer = reserve(
      zbuffer_, zbuffer_entries(config_.width, config_.height) * sizeof(std::uint64_t), stream);
  if (!zbuffer) {
    return holoscan::make_unexpected(std::move(zbuffer).error());
  }

  // The outputs.
  const bool uint16 = config_.depth_format == DepthFormat::kUint16;
  auto depth_loan = depth.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = std::array<std::int64_t, 2>{config_.height, config_.width},
      .dtype = uint16 ? kUInt16 : kFloat32});
  if (!depth_loan) {
    return holoscan::make_unexpected(std::move(depth_loan).error());
  }
  auto depth_writer = depth_loan->write(stream);
  if (!depth_writer) {
    return holoscan::make_unexpected(std::move(depth_writer).error());
  }
  std::optional<holoscan::TensorOutputLoan> color_loan;
  std::optional<holoscan::TensorOutputWriteGuard> color_writer;
  if (config_.color) {
    auto loan = color.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = std::array<std::int64_t, 3>{config_.height, config_.width, 3}, .dtype = kUInt8});
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

  if (const cudaError_t status =
          launch_project(point_data, color_data, static_cast<std::uint32_t>(count), params,
                         static_cast<std::uint64_t*>(*zbuffer), depth_writer->data_as<void>(),
                         color_writer ? color_writer->data_as<std::uint8_t>() : nullptr, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("projection kernels", status));
  }

  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*cloud);
  if (auto committed = std::move(*depth_writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  // Depth goes first: should the color then fail to go out, the frame's depth
  // is already delivered without it, and compute() fails (see the header).
  if (auto sent =
          depth.emit_tensor(std::move(*depth_loan),
                            image_descriptor(config_.width, config_.height,
                                             uint16 ? holoscan::schema::ImageEncoding_MONO16
                                                    : holoscan::schema::ImageEncoding_MONO32F,
                                             cloud->metadata, config_.frame_id),
                            emit);
      !sent) {
    return sent;
  }
  if (config_.color) {
    if (auto committed = std::move(*color_writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return color.emit_tensor(
        std::move(*color_loan),
        image_descriptor(config_.width, config_.height, holoscan::schema::ImageEncoding_RGB8,
                         cloud->metadata, config_.frame_id),
        emit);
  }
  return {};
}

}  // namespace point_cloud_to_depth
