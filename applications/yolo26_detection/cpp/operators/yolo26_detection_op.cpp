// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "yolo26_detection_op.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer/holoinfer.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace yolo26_detection {
namespace {

namespace infer = holoscan::inference;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kInt32{kDLInt, 32U, 1U};
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// HoloInfer's names for the model and its tensors. It binds tensors by
// position, so they need not match the names inside the model.
constexpr const char* kModelName = "yolo26";
constexpr const char* kInputName = "images";
constexpr const char* kOutputName = "output0";

holoscan::Error invalid(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "Yolo26DetectionOp: " + message};
}

holoscan::Error failure(const std::string& message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, "Yolo26DetectionOp: " + message};
}

holoscan::Error cuda_failure(const char* what, cudaError_t status) {
  return failure(std::string(what) + ": " + cudaGetErrorString(status));
}

holoscan::TensorOutputSpec device_tensor(DLDataType dtype, std::uint32_t rank,
                                         std::uint64_t bytes) {
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

bool is_onnx(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return extension == ".onnx";
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

std::string dims_text(const std::vector<std::int64_t>& dims) {
  std::string text = "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(dims[i]);
  }
  return text + "]";
}

// Keeps a pinned host tensor's memory from reuse until the work queued on
// `stream` so far is done: its allocator waits for the stream if it can,
// otherwise this waits.
holoscan::expected<void, holoscan::Error> hold_until_done(const holoscan::Tensor& tensor,
                                                          cudaStream_t stream) {
  holoscan::Tensor handle = tensor;
  if (handle.set_deallocation_stream(stream)) {
    return {};
  }
  if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
  }
  return {};
}

// HoloInfer's settings for `config`'s model, the same for prepare_engine() and
// start() so that both find the same cached engine.
std::shared_ptr<infer::InferenceSpecs> make_specs(const Yolo26DetectionConfig& config) {
  auto specs = std::make_shared<infer::InferenceSpecs>(
      "trt", infer::Mappings{}, infer::Mappings{{kModelName, config.model.string()}},
      infer::MultiMappings{{kModelName, {kInputName}}},
      infer::MultiMappings{{kModelName, {kOutputName}}}, infer::Mappings{}, infer::MultiMappings{},
      /*dynamic_input_dims=*/false, /*is_engine_path=*/!is_onnx(config.model),
      /*parallel_proc=*/false, config.fp16, /*cuda_buffer_in=*/true, /*cuda_buffer_out=*/true,
      config.cuda_graphs, /*dla_core=*/-1, /*dla_gpu_fallback=*/false,
      /*gpu_resident_inference=*/false, std::function<cudaStream_t(std::int32_t)>{});
  specs->max_memory_ = config.build_workspace_bytes;
  return specs;
}

template <std::size_t Rank>
holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error> allocate(
    holoscan::Output<holoscan::Tensor>& port, std::array<std::int64_t, Rank> shape,
    DLDataType dtype) {
  return port.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
}

}  // namespace

// The loaded model: HoloInfer's context and specifications, which own the
// input and output buffers that TensorRT reads and writes.
struct Yolo26DetectionOp::Engine {
  std::shared_ptr<infer::InferenceSpecs> specs;
  std::unique_ptr<infer::InferContext> context;
  float* input{};
  const float* output{};
  int rows{};

  ~Engine() {
    // The context's TensorRT objects go before the buffers they were bound to.
    context.reset();
    specs.reset();
  }
};

void validate_config(const Yolo26DetectionConfig& config) {
  if (config.model.empty()) {
    throw std::invalid_argument("a model path is required");
  }
  if (config.input_size < kMinInputSize || config.input_size > kMaxInputSize ||
      config.input_size % 32 != 0) {
    throw std::invalid_argument("input_size must be a multiple of 32 in " +
                                std::to_string(kMinInputSize) + ".." +
                                std::to_string(kMaxInputSize));
  }
  if (!(config.score_threshold >= 0.0F && config.score_threshold <= 1.0F)) {
    throw std::invalid_argument("score_threshold must be in [0, 1]");
  }
  if (config.max_detections < 1 || config.max_detections > kMaxRows) {
    throw std::invalid_argument("max_detections must be 1.." + std::to_string(kMaxRows));
  }
  if (config.num_classes < 1 || config.num_classes > 65536) {
    throw std::invalid_argument("num_classes must be 1..65536");
  }
  if (config.build_workspace_bytes < (std::size_t{1} << 24U)) {
    throw std::invalid_argument("build_workspace_bytes must be at least 16 MiB");
  }
}

void prepare_engine(const Yolo26DetectionConfig& config) {
  if (!is_onnx(config.model)) {
    return;
  }
  HOLOSCAN_LOG_INFO(
      "Yolo26DetectionOp: preparing {}; building its TensorRT engine takes a while "
      "when none is cached",
      config.model.string());
  auto specs = make_specs(config);
  infer::InferContext context;
  const infer::InferStatus status = context.set_inference_params(specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    throw std::runtime_error("Yolo26DetectionOp: cannot load " + config.model.string() + ": " +
                             status.get_message());
  }
}

Yolo26DetectionOp::Yolo26DetectionOp(Yolo26DetectionConfig config) : config_(std::move(config)) {
  validate_config(config_);
}

Yolo26DetectionOp::~Yolo26DetectionOp() { release(); }

void Yolo26DetectionOp::setup(holoscan::OperatorSpec& spec) {
  const auto capacity = static_cast<std::uint64_t>(config_.max_detections);
  // Two slots: one image in compute() while the next one arrives.
  spec.input(image, "image").queue_depth(2U);
  spec.output(boxes, "boxes").produces_tensor(device_tensor(kFloat32, 2U, capacity * 16U));
  spec.output(scores, "scores").produces_tensor(device_tensor(kFloat32, 1U, capacity * 4U));
  spec.output(class_ids, "class_ids").produces_tensor(device_tensor(kInt32, 1U, capacity * 4U));
  spec.output(count, "count").produces_tensor(device_tensor(kInt32, 1U, 4U));
}

holoscan::Contract Yolo26DetectionOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

void Yolo26DetectionOp::start() {
  release();
  const int size = config_.input_size;
  const bool onnx = is_onnx(config_.model);
  HOLOSCAN_LOG_INFO("Yolo26DetectionOp: loading {}{}", config_.model.string(),
                    onnx ? "; building its TensorRT engine takes a while when none is cached" : "");
  auto engine = std::make_unique<Engine>();
  engine->specs = make_specs(config_);
  engine->context = std::make_unique<infer::InferContext>();
  infer::InferStatus status = engine->context->set_inference_params(engine->specs);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS) {
    throw std::runtime_error("Yolo26DetectionOp: cannot load " + config_.model.string() + ": " +
                             status.get_message());
  }

  // The model must take [1, 3, S, S] and give [1, N, 6].
  const auto inputs = engine->context->get_input_dimensions();
  const auto outputs = engine->context->get_output_dimensions();
  const std::vector<std::int64_t> expected_input{1, 3, size, size};
  const auto in = inputs.find(kModelName);
  if (in == inputs.end() || in->second.size() != 1 || in->second[0] != expected_input) {
    throw std::runtime_error(
        "Yolo26DetectionOp: " + config_.model.string() + " takes " +
        (in == inputs.end() || in->second.size() != 1 ? std::string("other inputs")
                                                      : dims_text(in->second[0])) +
        ", not one " + dims_text(expected_input) +
        " input; export it for this input size, and delete any engine cached beside it");
  }
  const auto out = outputs.find(kModelName);
  if (out == outputs.end() || out->second.size() != 1 || out->second[0].size() != 3 ||
      out->second[0][0] != 1 || out->second[0][2] != kRowFields || out->second[0][1] < 1 ||
      out->second[0][1] > kMaxRows) {
    throw std::runtime_error(
        "Yolo26DetectionOp: " + config_.model.string() + " gives " +
        (out == outputs.end() || out->second.size() != 1 ? std::string("other outputs")
                                                         : dims_text(out->second[0])) +
        ", not one [1, N, 6] output of end-to-end detections; export it with nms=False");
  }
  engine->rows = static_cast<int>(out->second[0][1]);
  const auto output = engine->specs->output_per_model_.find(kOutputName);
  if (output == engine->specs->output_per_model_.end() || !output->second->device_buffer_ ||
      output->second->get_datatype() != infer::holoinfer_datatype::h_Float32) {
    throw std::runtime_error("Yolo26DetectionOp: " + config_.model.string() +
                             " must give float32 detections");
  }
  engine->output = static_cast<const float*>(output->second->device_buffer_->data());

  // The input buffer, which the letterbox writes and TensorRT reads in place.
  std::vector<std::int64_t> input_dims = expected_input;
  status = infer::allocate_buffers(engine->specs->data_per_tensor_, input_dims,
                                   infer::holoinfer_datatype::h_Float32, kInputName,
                                   /*allocate_cuda=*/true, /*device_id=*/0);
  const auto input = engine->specs->data_per_tensor_.find(kInputName);
  if (status.get_code() != infer::holoinfer_code::H_SUCCESS ||
      input == engine->specs->data_per_tensor_.end() || !input->second->device_buffer_) {
    throw std::runtime_error("Yolo26DetectionOp: cannot allocate the model input: " +
                             status.get_message());
  }
  engine->input = static_cast<float*>(input->second->device_buffer_->data());
  if (engine->input == nullptr || engine->output == nullptr) {
    throw std::runtime_error("Yolo26DetectionOp: the model buffers are not allocated");
  }
  if (const cudaError_t created = cudaEventCreateWithFlags(&frame_done_, cudaEventDisableTiming);
      created != cudaSuccess) {
    frame_done_ = nullptr;
    throw std::runtime_error(std::string("Yolo26DetectionOp: cudaEventCreateWithFlags: ") +
                             cudaGetErrorString(created));
  }
  void* passed = nullptr;
  if (const cudaError_t allocated = cudaHostAlloc(&passed, sizeof(std::int32_t), cudaHostAllocMapped);
      allocated != cudaSuccess) {
    throw std::runtime_error(std::string("Yolo26DetectionOp: cudaHostAlloc: ") +
                             cudaGetErrorString(allocated));
  }
  passed_ = static_cast<std::int32_t*>(passed);
  *passed_ = 0;
  warned_dropped_ = false;
  HOLOSCAN_LOG_INFO("Yolo26DetectionOp: {} ready, input [1, 3, {}, {}], {} output rows",
                    config_.model.string(), size, size, engine->rows);
  engine_ = std::move(engine);
}

void Yolo26DetectionOp::stop() {
  // The last frame's count, before its memory goes.
  if (frame_done_ != nullptr && cudaEventSynchronize(frame_done_) == cudaSuccess) {
    warn_if_dropped();
  }
  release();
}

void Yolo26DetectionOp::warn_if_dropped() {
  if (warned_dropped_ || passed_ == nullptr || *passed_ <= config_.max_detections) {
    return;
  }
  warned_dropped_ = true;
  HOLOSCAN_LOG_WARN(
      "Yolo26DetectionOp: {} detections passed the score threshold in a frame, more than "
      "max_detections ({}); the lowest-ranked were dropped. Raise max_detections to keep them",
      *passed_, config_.max_detections);
}

void Yolo26DetectionOp::release() noexcept {
  // Freeing waits for the GPU work that may still use the buffers.
  engine_.reset();
  if (frame_done_ != nullptr) {
    cudaEventDestroy(frame_done_);
    frame_done_ = nullptr;
  }
  if (passed_ != nullptr) {
    cudaFreeHost(passed_);
    passed_ = nullptr;
  }
  if (staging_ != nullptr) {
    cudaFree(staging_);
    staging_ = nullptr;
    staging_bytes_ = 0;
  }
}

holoscan::expected<const std::uint8_t*, holoscan::Error> Yolo26DetectionOp::stage(
    const holoscan::Tensor& tensor, int width, int height, int channels, std::size_t pitch,
    cudaStream_t stream) {
  const std::size_t row = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  const std::size_t bytes = row * static_cast<std::size_t>(height);
  if (staging_bytes_ < bytes) {
    // Grows a few times at most; the old buffer is freed once the stream is
    // done with it.
    if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_failure("cudaStreamSynchronize", status));
    }
    const std::size_t grown = std::max(bytes, staging_bytes_ * 2U);
    cudaFree(staging_);
    staging_ = nullptr;
    staging_bytes_ = 0;
    if (const cudaError_t status = cudaMalloc(&staging_, grown); status != cudaSuccess) {
      staging_ = nullptr;
      return holoscan::make_unexpected(cuda_failure("cudaMalloc", status));
    }
    staging_bytes_ = grown;
  }
  // Plain host memory: the runtime does not order its producer, so wait for
  // it; the copy then stages it before returning. Pinned memory: hold it
  // until the asynchronous copy has read it.
  const bool pinned = tensor.device().device_type == kDLCUDAHost;
  if (!pinned) {
    if (auto ready = tensor.synchronize_for_host(); !ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
  }
  if (const cudaError_t status =
          cudaMemcpy2DAsync(staging_, row, tensor.data(), pitch, row,
                            static_cast<std::size_t>(height), cudaMemcpyHostToDevice, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaMemcpy2DAsync", status));
  }
  if (pinned) {
    if (auto held = hold_until_done(tensor, stream); !held) {
      return holoscan::make_unexpected(std::move(held).error());
    }
  }
  return static_cast<const std::uint8_t*>(staging_);
}

holoscan::expected<void, holoscan::Error> Yolo26DetectionOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  if (!engine_) {
    return holoscan::make_unexpected(failure("the model is not loaded"));
  }
  // The runtime gives an operator with device outputs a stream, and writes
  // their loans on that stream only.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(failure("the runtime gave no CUDA stream"));
  }
  int device = 0;
  if (const cudaError_t status = cudaStreamGetDevice(stream, &device); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamGetDevice", status));
  }
  if (device != 0) {
    return holoscan::make_unexpected(invalid("runs on GPU 0, where HoloInfer loads the model"));
  }
  // A finished previous frame's count of passed rows; a frame still running
  // is checked by a later one, or by stop().
  if (const cudaError_t query = cudaEventQuery(frame_done_); query == cudaSuccess) {
    warn_if_dropped();
  } else if (query == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady) {
    cudaGetLastError();  // Not an error; keep it from the kernel launch checks.
  }
  // The previous frame may have run on another stream: wait for its work on
  // the buffers this one reuses, and mark this one's when it has been queued,
  // on every way out.
  if (const cudaError_t status = cudaStreamWaitEvent(stream, frame_done_, 0);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaStreamWaitEvent", status));
  }
  struct MarkDone {
    cudaEvent_t event;
    cudaStream_t stream;
    ~MarkDone() {
      if (event != nullptr) {
        cudaEventRecord(event, stream);
      }
    }
  } mark_done{frame_done_, stream};

  // The image.
  const holoscan::schema::ImageT& descriptor = sample->data;
  const auto format = pixel_format(descriptor.encoding);
  if (!format) {
    return holoscan::make_unexpected(
        invalid(std::string("unsupported image encoding ") +
                holoscan::schema::EnumNameImageEncoding(descriptor.encoding) +
                "; use RGB8, BGR8, RGBA8 or BGRA8"));
  }
  if (!descriptor.data) {
    return holoscan::make_unexpected(invalid("the image has no tensor"));
  }
  const holoscan::Tensor& tensor = *descriptor.data;
  const int pixel_channels = channels(*format);
  const DLDataType dtype = tensor.dtype();
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (dtype.code != kUInt8.code || dtype.bits != kUInt8.bits || dtype.lanes != kUInt8.lanes ||
      shape.size() != 3 || shape[2] != pixel_channels) {
    return holoscan::make_unexpected(
        invalid("the image tensor must be uint8 [height, width, channels] for its encoding"));
  }
  const std::vector<std::int64_t> strides = tensor.strides();
  const std::int64_t height = shape[0];
  const std::int64_t width = shape[1];
  if (height < 1 || width < 1 || height > kMaxImageSide || width > kMaxImageSide) {
    return holoscan::make_unexpected(
        invalid("image sides must be 1.." + std::to_string(kMaxImageSide)));
  }
  if (strides.size() != 3 || strides[2] != 1 || strides[1] != pixel_channels ||
      strides[0] < width * pixel_channels) {
    return holoscan::make_unexpected(invalid("image pixels must be packed within each row"));
  }
  if ((descriptor.width != 0 && descriptor.width != width) ||
      (descriptor.height != 0 && descriptor.height != height)) {
    return holoscan::make_unexpected(
        invalid("the image's width and height disagree with its tensor"));
  }
  const DLDevice where = tensor.device();
  const bool on_gpu = where.device_type == kDLCUDA || where.device_type == kDLCUDAManaged;
  if (!on_gpu && where.device_type != kDLCPU && where.device_type != kDLCUDAHost) {
    return holoscan::make_unexpected(invalid("the image must be in GPU or host memory"));
  }
  if (where.device_type == kDLCUDA && where.device_id != device) {
    return holoscan::make_unexpected(invalid("the image is on another GPU than the operator"));
  }
  const auto pitch = static_cast<std::size_t>(strides[0]);
  const std::uint8_t* pixels = nullptr;
  std::size_t pixels_pitch = pitch;
  if (on_gpu) {
    // The runtime has ordered the producer's writes before this stream.
    pixels = static_cast<const std::uint8_t*>(tensor.data());
  } else {
    auto staged = stage(tensor, static_cast<int>(width), static_cast<int>(height), pixel_channels,
                        pitch, stream);
    if (!staged) {
      return holoscan::make_unexpected(std::move(staged).error());
    }
    pixels = *staged;
    pixels_pitch = static_cast<std::size_t>(width) * static_cast<std::size_t>(pixel_channels);
  }

  // Letterbox into the model input, run the model, decode into the outputs,
  // all in order on the stream; HoloInfer joins it with events. Frames before
  // this one are done with the buffers: see frame_done_.
  const Letterbox letterbox =
      make_letterbox(static_cast<int>(width), static_cast<int>(height), config_.input_size);
  if (const cudaError_t status =
          launch_letterbox(ImageView{.data = pixels,
                                     .pitch = pixels_pitch,
                                     .width = static_cast<int>(width),
                                     .height = static_cast<int>(height),
                                     .format = *format},
                           letterbox, config_.pad_value, engine_->input, stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("letterbox kernel", status));
  }
  infer::InferStatus inferred = engine_->context->execute_inference(engine_->specs, stream);
  if (inferred.get_code() != infer::holoinfer_code::H_SUCCESS) {
    return holoscan::make_unexpected(failure("inference failed: " + inferred.get_message()));
  }

  const std::int64_t capacity = config_.max_detections;
  auto boxes_loan = allocate<2>(boxes, {capacity, 4}, kFloat32);
  auto scores_loan = allocate<1>(scores, {capacity}, kFloat32);
  auto class_ids_loan = allocate<1>(class_ids, {capacity}, kInt32);
  auto count_loan = allocate<1>(count, {1}, kInt32);
  for (const auto* loan : {&boxes_loan, &scores_loan, &class_ids_loan, &count_loan}) {
    if (!*loan) {
      return holoscan::make_unexpected(loan->error());
    }
  }
  auto boxes_writer = boxes_loan->write(stream);
  auto scores_writer = scores_loan->write(stream);
  auto class_ids_writer = class_ids_loan->write(stream);
  auto count_writer = count_loan->write(stream);
  for (const auto* writer : {&boxes_writer, &scores_writer, &class_ids_writer, &count_writer}) {
    if (!*writer) {
      return holoscan::make_unexpected(writer->error());
    }
  }
  const DecodeParams params{.rows = engine_->rows,
                            .score_threshold = config_.score_threshold,
                            .num_classes = config_.num_classes,
                            .capacity = config_.max_detections,
                            .letterbox = letterbox};
  if (const cudaError_t status =
          launch_decode(engine_->output, params,
                        DecodeOutputs{.boxes = boxes_writer->data_as<float>(),
                                      .scores = scores_writer->data_as<float>(),
                                      .class_ids = class_ids_writer->data_as<std::int32_t>(),
                                      .count = count_writer->data_as<std::int32_t>(),
                                      .passed = passed_},
                        stream);
      status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("decode kernel", status));
  }
  for (auto* writer : {&boxes_writer, &scores_writer, &class_ids_writer, &count_writer}) {
    if (auto committed = std::move(**writer).commit(); !committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
  }
  mark_done.event = nullptr;
  if (const cudaError_t status = cudaEventRecord(frame_done_, stream); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_failure("cudaEventRecord", status));
  }

  // One set of options, so every output carries the image's capture time and
  // frame id.
  const holoscan::EmitOptions emit = holoscan::sensor_io::derive_from(*sample);
  if (auto sent = boxes.emit(std::move(*boxes_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = scores.emit(std::move(*scores_loan), emit); !sent) {
    return sent;
  }
  if (auto sent = class_ids.emit(std::move(*class_ids_loan), emit); !sent) {
    return sent;
  }
  return count.emit(std::move(*count_loan), emit);
}

}  // namespace yolo26_detection
