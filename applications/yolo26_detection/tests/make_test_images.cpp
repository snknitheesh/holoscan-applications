// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Writes synthetic scenes as image files for the tests, with the stub model's
// detections of each listed in expected.csv:
//
//   scene_0.jpg  1280x720, baseline JPEG      (frame 0)
//   scene_1.ppm  333x500, PPM                 (frame 1)
//   scene_2.JPEG 500x333, progressive JPEG    (frame 2)
//
// Usage: yolo26_make_test_images DIR

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "detection/image_files.hpp"
#include "detection/synthetic_scene.hpp"

namespace {

namespace yd = yolo26_detection;

void check(nvjpegStatus_t status, const char* what) {
  if (status != NVJPEG_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) +
                             " failed: " + std::to_string(static_cast<int>(status)));
  }
}

// Encodes packed RGB8 pixels as a JPEG of quality 95 without chroma
// subsampling, so that colors keep to their rectangles.
std::vector<std::uint8_t> encode_jpeg(const std::vector<std::uint8_t>& rgb, int width, int height,
                                      bool progressive) {
  nvjpegHandle_t handle{};
  check(nvjpegCreateSimple(&handle), "nvjpegCreateSimple");
  nvjpegEncoderState_t state{};
  nvjpegEncoderParams_t params{};
  check(nvjpegEncoderStateCreate(handle, &state, nullptr), "nvjpegEncoderStateCreate");
  check(nvjpegEncoderParamsCreate(handle, &params, nullptr), "nvjpegEncoderParamsCreate");
  check(nvjpegEncoderParamsSetQuality(params, 95, nullptr), "nvjpegEncoderParamsSetQuality");
  check(nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_444, nullptr),
        "nvjpegEncoderParamsSetSamplingFactors");
  if (progressive) {
    check(nvjpegEncoderParamsSetOptimizedHuffman(params, 1, nullptr),
          "nvjpegEncoderParamsSetOptimizedHuffman");
    check(nvjpegEncoderParamsSetEncoding(params, NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, nullptr),
          "nvjpegEncoderParamsSetEncoding");
  }
  void* device = nullptr;
  if (cudaMalloc(&device, rgb.size()) != cudaSuccess ||
      cudaMemcpy(device, rgb.data(), rgb.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cannot copy the pixels to the GPU");
  }
  nvjpegImage_t source{};
  source.channel[0] = static_cast<unsigned char*>(device);
  source.pitch[0] = static_cast<std::size_t>(width) * 3U;
  check(
      nvjpegEncodeImage(handle, state, params, &source, NVJPEG_INPUT_RGBI, width, height, nullptr),
      "nvjpegEncodeImage");
  std::size_t length = 0;
  check(nvjpegEncodeRetrieveBitstream(handle, state, nullptr, &length, nullptr),
        "nvjpegEncodeRetrieveBitstream");
  std::vector<std::uint8_t> jpeg(length);
  check(nvjpegEncodeRetrieveBitstream(handle, state, jpeg.data(), &length, nullptr),
        "nvjpegEncodeRetrieveBitstream");
  cudaDeviceSynchronize();
  jpeg.resize(length);
  cudaFree(device);
  nvjpegEncoderParamsDestroy(params);
  nvjpegEncoderStateDestroy(state);
  nvjpegDestroy(handle);
  return jpeg;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  const std::filesystem::path directory = argv[1];
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  struct Image {
    const char* name;
    int width;
    int height;
    enum { kJpeg, kProgressive, kPpm } kind;
  };
  std::ofstream expected(directory / "expected.csv");
  expected << "# image,class_id,x1,y1,x2,y2: the stub model's detections\n";
  std::uint32_t frame = 0;
  for (const Image& image :
       {Image{"scene_0.jpg", 1280, 720, Image::kJpeg}, Image{"scene_1.ppm", 333, 500, Image::kPpm},
        Image{"scene_2.JPEG", 500, 333, Image::kProgressive}}) {
    const yd::SyntheticScene scene = yd::make_scene(image.width, image.height, frame++);
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(image.width) * image.height * 3U);
    yd::render_scene(scene, yd::PixelFormat::kRgb8, rgb.data(),
                     static_cast<std::size_t>(image.width) * 3U);
    const std::filesystem::path path = directory / image.name;
    if (image.kind == Image::kPpm) {
      yd::write_ppm(path, image.width, image.height, rgb);
    } else {
      const auto jpeg =
          encode_jpeg(rgb, image.width, image.height, image.kind == Image::kProgressive);
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(jpeg.data()),
                static_cast<std::streamsize>(jpeg.size()));
      if (!out.flush()) throw std::runtime_error("cannot write " + path.string());
    }
    for (const yd::BoxDetection& d : yd::scene_truth(scene)) {
      expected << image.name << ',' << d.class_id << ',' << d.x1 << ',' << d.y1 << ',' << d.x2
               << ',' << d.y2 << '\n';
    }
  }
  // Not an image: the sources skip it.
  std::ofstream(directory / "notes.txt") << "not an image\n";
  if (!expected.flush()) throw std::runtime_error("cannot write expected.csv");
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
