// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The letterbox: its geometry, as the Ultralytics LetterBox computes it, for
// wide and tall images with even and odd padding; and the kernel against a
// host evaluation of the same expressions, bit for bit, for every encoding,
// with padded rows, downscaling and upscaling; plus what the values mean:
// pixel centers aligned, channel order, padding, and alpha ignored.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "detection/letterbox.hpp"
#include "test_support.hpp"

namespace yolo26_detection {
namespace {

// Pixels on the host, rows `pitch` bytes apart.
struct HostImage {
  int width{};
  int height{};
  PixelFormat format{};
  std::size_t pitch{};
  std::vector<std::uint8_t> bytes;
};

HostImage random_image(int width, int height, PixelFormat format, std::size_t row_padding,
                       std::uint32_t seed) {
  HostImage image{
      width, height, format, static_cast<std::size_t>(width) * channels(format) + row_padding, {}};
  image.bytes.resize(image.pitch * height);
  std::mt19937 random(seed);
  for (auto& byte : image.bytes) {
    byte = static_cast<std::uint8_t>(random() & 0xFFU);
  }
  return image;
}

// The kernel's output for `image`, as [3, size, size] floats.
std::vector<float> run_kernel(const HostImage& image, const Letterbox& lb, std::uint8_t pad) {
  void* pixels = nullptr;
  float* output = nullptr;
  const std::size_t out_count = 3U * lb.size * lb.size;
  if (cudaMalloc(&pixels, image.bytes.size()) != cudaSuccess ||
      cudaMalloc(&output, out_count * sizeof(float)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }
  cudaMemcpy(pixels, image.bytes.data(), image.bytes.size(), cudaMemcpyHostToDevice);
  cudaMemset(output, 0xFF, out_count * sizeof(float));  // NaN: unwritten values show.
  const cudaError_t status =
      launch_letterbox(ImageView{static_cast<const std::uint8_t*>(pixels), image.pitch, image.width,
                                 image.height, image.format},
                       lb, pad, output, nullptr);
  std::vector<float> result(out_count);
  cudaMemcpy(result.data(), output, out_count * sizeof(float), cudaMemcpyDeviceToHost);
  const cudaError_t synced = cudaDeviceSynchronize();
  cudaFree(pixels);
  cudaFree(output);
  if (status != cudaSuccess || synced != cudaSuccess) {
    throw std::runtime_error("letterbox kernel failed");
  }
  return result;
}

// The same expressions as the kernel, evaluated on the host.
std::vector<float> reference(const HostImage& image, const Letterbox& lb, std::uint8_t pad) {
  const int n = channels(image.format);
  const bool bgr = blue_first(image.format);
  const int offsets[3] = {bgr ? 2 : 0, 1, bgr ? 0 : 2};
  const float x_ratio = static_cast<float>(image.width) / static_cast<float>(lb.resized_width);
  const float y_ratio = static_cast<float>(image.height) / static_cast<float>(lb.resized_height);
  const std::size_t plane = static_cast<std::size_t>(lb.size) * lb.size;
  std::vector<float> out(3 * plane);
  for (int v = 0; v < lb.size; ++v) {
    for (int u = 0; u < lb.size; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * lb.size + u;
      const int x = u - lb.pad_left;
      const int y = v - lb.pad_top;
      if (x < 0 || y < 0 || x >= lb.resized_width || y >= lb.resized_height) {
        for (int c = 0; c < 3; ++c) out[c * plane + i] = static_cast<float>(pad) / 255.0F;
        continue;
      }
      const float sx = std::fmax((static_cast<float>(x) + 0.5F) * x_ratio - 0.5F, 0.0F);
      const float sy = std::fmax((static_cast<float>(y) + 0.5F) * y_ratio - 0.5F, 0.0F);
      const int x0 = std::min(static_cast<int>(sx), image.width - 1);
      const int y0 = std::min(static_cast<int>(sy), image.height - 1);
      const int x1 = std::min(x0 + 1, image.width - 1);
      const int y1 = std::min(y0 + 1, image.height - 1);
      const float fx = sx - static_cast<float>(x0);
      const float fy = sy - static_cast<float>(y0);
      const auto at = [&](int px, int py, int k) {
        return static_cast<float>(image.bytes[py * image.pitch + px * n + k]);
      };
      for (int c = 0; c < 3; ++c) {
        const int k = offsets[c];
        const float top = (1.0F - fx) * at(x0, y0, k) + fx * at(x1, y0, k);
        const float bottom = (1.0F - fx) * at(x0, y1, k) + fx * at(x1, y1, k);
        out[c * plane + i] = ((1.0F - fy) * top + fy * bottom) / 255.0F;
      }
    }
  }
  return out;
}

void check_geometry(int width, int height, int size, int resized_width, int resized_height,
                    int pad_left, int pad_top) {
  const Letterbox lb = make_letterbox(width, height, size);
  const std::string what = std::to_string(width) + "x" + std::to_string(height) + " in " +
                           std::to_string(size) + ": " + std::to_string(lb.resized_width) + "x" +
                           std::to_string(lb.resized_height) + " at (" +
                           std::to_string(lb.pad_left) + ", " + std::to_string(lb.pad_top) + ")";
  VERIFY_MSG(lb.resized_width == resized_width && lb.resized_height == resized_height &&
                 lb.pad_left == pad_left && lb.pad_top == pad_top,
             what);
  VERIFY(lb.scale == static_cast<float>(std::min(static_cast<double>(size) / width,
                                                 static_cast<double>(size) / height)));
}

void check_geometries() {
  check_geometry(640, 480, 640, 640, 480, 0, 80);    // Wide, even padding.
  check_geometry(810, 1080, 640, 480, 640, 80, 0);   // Tall, even padding.
  check_geometry(1000, 333, 640, 640, 213, 0, 213);  // Odd padding: 213 above, 214 below.
  check_geometry(333, 1000, 640, 213, 640, 213, 0);  // And 213 left, 214 right.
  // 721 * 0.5 = 360.5 rounds half to even, to 360, as Python's round() does.
  check_geometry(1280, 721, 640, 640, 360, 0, 140);
  check_geometry(1280, 723, 640, 640, 362, 0, 139);  // 361.5 rounds to 362.
  check_geometry(100, 37, 640, 640, 237, 0, 201);    // Upscaled.
  check_geometry(1, 1, 32, 32, 32, 0, 0);
  check_geometry(16384, 1, 4096, 4096, 1, 0, 2047);  // A side never rounds to 0.
  for (const auto& [w, h, s] : {std::array<int, 3>{0, 10, 640}, std::array<int, 3>{10, 0, 640},
                                std::array<int, 3>{16385, 10, 640}, std::array<int, 3>{10, 10, 31},
                                std::array<int, 3>{10, 10, 4097}}) {
    VERIFY_MSG(
        testing::throws<std::invalid_argument>([&] { (void)make_letterbox(w, h, s); }),
        "rejects " + std::to_string(w) + "x" + std::to_string(h) + " in " + std::to_string(s));
  }
}

// The kernel equals the host evaluation exactly.
void check_against_reference(int width, int height, int size, PixelFormat format,
                             std::size_t row_padding) {
  const HostImage image = random_image(width, height, format, row_padding, width * 31U + height);
  const Letterbox lb = make_letterbox(width, height, size);
  const auto gpu = run_kernel(image, lb, 114);
  const auto cpu = reference(image, lb, 114);
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    wrong += gpu[i] != cpu[i];
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " of " + std::to_string(gpu.size()) +
                             " values differ for " + std::to_string(width) + "x" +
                             std::to_string(height) + " in " + std::to_string(size));
}

// What the values mean, independently of the expressions: a black image in
// gray padding fills exactly its letterboxed rectangle, in every channel.
void check_placement(int width, int height, int size) {
  HostImage image{width, height, PixelFormat::kRgb8, static_cast<std::size_t>(width) * 3, {}};
  image.bytes.assign(image.pitch * height, 0);
  const Letterbox lb = make_letterbox(width, height, size);
  const auto out = run_kernel(image, lb, 114);
  const std::size_t plane = static_cast<std::size_t>(size) * size;
  std::size_t wrong = 0;
  for (int c = 0; c < 3; ++c) {
    for (int v = 0; v < size; ++v) {
      for (int u = 0; u < size; ++u) {
        const bool inside = u >= lb.pad_left && u < lb.pad_left + lb.resized_width &&
                            v >= lb.pad_top && v < lb.pad_top + lb.resized_height;
        wrong += out[c * plane + v * size + u] != (inside ? 0.0F : 114.0F / 255.0F);
      }
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " values misplaced for " + std::to_string(width) +
                             "x" + std::to_string(height));
}

// Each encoding's red, green and blue land in planes 0, 1 and 2; alpha is
// ignored; an image of the input's size passes through unscaled.
void check_channels() {
  for (const PixelFormat format :
       {PixelFormat::kRgb8, PixelFormat::kBgr8, PixelFormat::kRgba8, PixelFormat::kBgra8}) {
    const int n = channels(format);
    HostImage image{64, 64, format, 64U * n, {}};
    image.bytes.resize(image.pitch * 64);
    std::mt19937 random(7);
    for (std::size_t p = 0; p < 64U * 64U; ++p) {
      std::uint8_t* px = image.bytes.data() + p * n;
      const std::uint8_t r = static_cast<std::uint8_t>(p % 251U);
      const std::uint8_t g = static_cast<std::uint8_t>((p * 7U) % 253U);
      const std::uint8_t b = static_cast<std::uint8_t>((p * 13U) % 255U);
      px[0] = blue_first(format) ? b : r;
      px[1] = g;
      px[2] = blue_first(format) ? r : b;
      if (n == 4) px[3] = static_cast<std::uint8_t>(random());
    }
    const auto out = run_kernel(image, make_letterbox(64, 64, 64), 114);
    std::size_t wrong = 0;
    for (std::size_t p = 0; p < 64U * 64U; ++p) {
      wrong += out[p] != static_cast<float>(p % 251U) / 255.0F;
      wrong += out[4096 + p] != static_cast<float>((p * 7U) % 253U) / 255.0F;
      wrong += out[8192 + p] != static_cast<float>((p * 13U) % 255U) / 255.0F;
    }
    VERIFY_MSG(wrong == 0, std::to_string(wrong) + " values wrong for encoding " +
                               std::to_string(static_cast<int>(format)));
  }
}

// Halving an image with alternating black and white columns gives gray:
// each output pixel center lies halfway between two input centers, as in
// OpenCV's INTER_LINEAR, not on one of them.
void check_pixel_centers() {
  HostImage image{128, 64, PixelFormat::kRgb8, 128U * 3U, {}};
  image.bytes.resize(image.pitch * 64);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 128; ++x) {
      for (int c = 0; c < 3; ++c) image.bytes[y * image.pitch + x * 3 + c] = x % 2 ? 255 : 0;
    }
  }
  const Letterbox lb = make_letterbox(128, 64, 64);  // 64x32, 16 rows of padding above.
  const auto out = run_kernel(image, lb, 0);
  std::size_t wrong = 0;
  for (int v = lb.pad_top; v < lb.pad_top + lb.resized_height; ++v) {
    for (int u = 0; u < 64; ++u) {
      wrong += out[v * 64 + u] != 0.5F;
    }
  }
  VERIFY_MSG(wrong == 0, std::to_string(wrong) + " pixels are not the mean of two columns");
}

void check_rejects_bad_arguments() {
  float* output = nullptr;
  std::uint8_t* pixels = nullptr;
  cudaMalloc(&output, 3 * 32 * 32 * sizeof(float));
  cudaMalloc(&pixels, 32 * 32 * 3);
  const Letterbox lb = make_letterbox(32, 32, 32);
  const ImageView good{pixels, 96, 32, 32, PixelFormat::kRgb8};
  VERIFY(launch_letterbox(good, lb, 114, output, nullptr) == cudaSuccess);
  VERIFY(launch_letterbox(ImageView{pixels, 95, 32, 32, PixelFormat::kRgb8}, lb, 114, output,
                          nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_letterbox(ImageView{pixels, 96, 31, 32, PixelFormat::kRgb8}, lb, 114, output,
                          nullptr) == cudaErrorInvalidValue);
  VERIFY(launch_letterbox(good, lb, 114, nullptr, nullptr) == cudaErrorInvalidValue);
  VERIFY(cudaDeviceSynchronize() == cudaSuccess);
  cudaFree(output);
  cudaFree(pixels);
}

}  // namespace
}  // namespace yolo26_detection

int main() {
  using yolo26_detection::PixelFormat;
  namespace yd = yolo26_detection;
  std::vector<yd::testing::TestCase> cases{
      {"geometry", yd::check_geometries},
      {"channels and alpha", yd::check_channels},
      {"pixel centers", yd::check_pixel_centers},
      {"rejects bad arguments", yd::check_rejects_bad_arguments},
  };
  // Wide and tall, even and odd padding, down- and upscaled, every encoding,
  // rows packed and padded.
  struct Case {
    int width, height, size;
    PixelFormat format;
    std::size_t padding;
  };
  for (const Case c :
       {Case{1280, 720, 640, PixelFormat::kRgb8, 0}, Case{720, 1280, 640, PixelFormat::kBgr8, 5},
        Case{1000, 333, 640, PixelFormat::kRgba8, 0}, Case{333, 1000, 320, PixelFormat::kBgra8, 12},
        Case{100, 37, 640, PixelFormat::kRgb8, 3}, Case{1921, 1081, 1280, PixelFormat::kBgr8, 1},
        Case{64, 64, 64, PixelFormat::kRgba8, 0}}) {
    cases.push_back(
        {"kernel = reference " + std::to_string(c.width) + "x" + std::to_string(c.height) + " in " +
             std::to_string(c.size),
         [c] { yd::check_against_reference(c.width, c.height, c.size, c.format, c.padding); }});
  }
  for (const auto& [w, h] : {std::pair{1000, 333}, std::pair{333, 1000}, std::pair{100, 37}}) {
    cases.push_back({"placement " + std::to_string(w) + "x" + std::to_string(h),
                     [w, h] { yd::check_placement(w, h, 640); }});
  }
  return yd::testing::run(cases);
}
