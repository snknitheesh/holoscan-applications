// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_files.hpp"

#include <nvjpeg.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace yolo26_detection {
namespace {

std::runtime_error file_error(const std::filesystem::path& path, const std::string& what) {
  return std::runtime_error(path.string() + ": " + what);
}

std::string lower_extension(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return extension;
}

bool is_image_name(const std::filesystem::path& path) {
  const std::string extension = lower_extension(path);
  return extension == ".jpg" || extension == ".jpeg" || extension == ".ppm";
}

// Reads the binary PPM header: magic, width, height and maxval, each after
// optional whitespace and comments, then exactly one whitespace byte.
// Leaves `in` at the first pixel byte.
void read_ppm_header(std::istream& in, const std::filesystem::path& path, int& width, int& height) {
  char magic[2] = {};
  if (!in.read(magic, 2) || magic[0] != 'P' || magic[1] != '6') {
    throw file_error(path, "not a binary PPM (P6) file");
  }
  const auto number = [&]() {
    int c = in.get();
    while (c != EOF && (std::isspace(c) || c == '#')) {
      if (c == '#') {
        while (c != EOF && c != '\n') c = in.get();
      }
      c = in.get();
    }
    std::int64_t value = 0;
    bool any = false;
    while (c != EOF && std::isdigit(c)) {
      value = value * 10 + (c - '0');
      any = true;
      if (value > 1'000'000) {
        throw file_error(path, "PPM header value out of range");
      }
      c = in.get();
    }
    if (!any || c == EOF || !std::isspace(c)) {
      throw file_error(path, "malformed PPM header");
    }
    return static_cast<int>(value);
  };
  width = number();
  height = number();
  const int maxval = number();
  if (maxval != 255) {
    throw file_error(path, "only PPM files with a maxval of 255 are supported");
  }
  if (width < 1 || height < 1 || width > kMaxImageSide || height > kMaxImageSide) {
    throw file_error(path, "image sides must be 1.." + std::to_string(kMaxImageSide));
  }
}

// The frame header of a JPEG: walks the marker segments up to the first
// start-of-frame and reads its size.
ImageFileInfo read_jpeg_info(std::istream& in, const std::filesystem::path& path) {
  const auto byte = [&]() {
    const int c = in.get();
    if (c == EOF) {
      throw file_error(path, "truncated JPEG header");
    }
    return c;
  };
  if (byte() != 0xFF || byte() != 0xD8) {
    throw file_error(path, "not a JPEG file");
  }
  while (true) {
    if (byte() != 0xFF) {
      throw file_error(path, "malformed JPEG marker");
    }
    int marker = byte();
    while (marker == 0xFF) {
      marker = byte();  // Fill bytes.
    }
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
      continue;  // Markers without a segment.
    }
    if (marker == 0xD9 || marker == 0xDA) {
      throw file_error(path, "JPEG has no frame header before its scan");
    }
    const int length = (byte() << 8) | byte();
    if (length < 2) {
      throw file_error(path, "malformed JPEG segment");
    }
    const bool huffman_frame = marker == 0xC0 || marker == 0xC1 || marker == 0xC2;
    const bool other_frame =
        marker >= 0xC3 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
    if (other_frame) {
      throw file_error(path, "only baseline, extended and progressive Huffman JPEGs are supported");
    }
    if (huffman_frame) {
      if (length < 8) {
        throw file_error(path, "malformed JPEG frame header");
      }
      const int precision = byte();
      const int height = (byte() << 8) | byte();
      const int width = (byte() << 8) | byte();
      const int components = byte();
      if (precision != 8) {
        throw file_error(path, "only 8-bit JPEGs are supported");
      }
      if (components != 1 && components != 3) {
        throw file_error(path, "only grayscale and three-component JPEGs are supported");
      }
      if (width < 1 || height < 1 || width > kMaxImageSide || height > kMaxImageSide) {
        throw file_error(path, "image sides must be 1.." + std::to_string(kMaxImageSide));
      }
      return ImageFileInfo{.kind = ImageFileKind::kJpeg, .width = width, .height = height};
    }
    in.ignore(length - 2);
    if (!in) {
      throw file_error(path, "truncated JPEG header");
    }
  }
}

}  // namespace

ImageFileInfo read_image_info(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw file_error(path, "cannot open");
  }
  const std::string extension = lower_extension(path);
  if (extension == ".ppm") {
    ImageFileInfo info{.kind = ImageFileKind::kPpm};
    read_ppm_header(in, path, info.width, info.height);
    return info;
  }
  if (extension == ".jpg" || extension == ".jpeg") {
    return read_jpeg_info(in, path);
  }
  throw file_error(path, "not a .jpg, .jpeg or .ppm file");
}

std::vector<std::filesystem::path> list_image_files(const std::filesystem::path& path) {
  std::error_code error;
  if (std::filesystem::is_regular_file(path, error)) {
    if (!is_image_name(path)) {
      throw file_error(path, "not a .jpg, .jpeg or .ppm file");
    }
    return {path};
  }
  std::vector<std::filesystem::path> files;
  std::filesystem::directory_iterator it(path, error);
  if (error) {
    throw file_error(path, "cannot list: " + error.message());
  }
  for (const auto& entry : it) {
    if (entry.is_regular_file(error) && is_image_name(entry.path())) {
      files.push_back(entry.path());
    }
  }
  if (files.empty()) {
    throw file_error(path, "no .jpg, .jpeg or .ppm files");
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw file_error(path, "cannot open");
  }
  std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
  if (in.bad()) {
    throw file_error(path, "cannot read");
  }
  return bytes;
}

std::vector<std::uint8_t> read_ppm(const std::filesystem::path& path, int& width, int& height) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw file_error(path, "cannot open");
  }
  read_ppm_header(in, path, width, height);
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3U);
  if (!in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()))) {
    throw file_error(path, "truncated PPM pixels");
  }
  return rgb;
}

void write_ppm(const std::filesystem::path& path, int width, int height,
               std::span<const std::uint8_t> rgb) {
  if (rgb.size() != static_cast<std::size_t>(width) * height * 3U) {
    throw file_error(path, "pixel count does not match the image size");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << "P6\n" << width << ' ' << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  if (!out.flush()) {
    throw file_error(path, "cannot write");
  }
}

std::vector<std::uint8_t> to_rgb(const std::uint8_t* pixels, std::size_t pitch, int width,
                                 int height, PixelFormat format) {
  const int n = channels(format);
  const bool bgr = blue_first(format);
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3U);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = pixels + static_cast<std::size_t>(y) * pitch;
    std::uint8_t* out = rgb.data() + static_cast<std::size_t>(y) * width * 3U;
    for (int x = 0; x < width; ++x) {
      out[3 * x + 0] = row[n * x + (bgr ? 2 : 0)];
      out[3 * x + 1] = row[n * x + 1];
      out[3 * x + 2] = row[n * x + (bgr ? 0 : 2)];
    }
  }
  return rgb;
}

void draw_boxes(std::span<std::uint8_t> rgb, int width, int height,
                std::span<const BoxDetection> detections, int thickness) {
  static constexpr std::array<std::array<std::uint8_t, 3>, 6> kColors{
      {{255, 56, 56}, {56, 255, 56}, {56, 128, 255}, {255, 214, 0}, {255, 56, 255}, {0, 230, 230}}};
  for (const BoxDetection& d : detections) {
    const auto& color = kColors[static_cast<std::size_t>(std::abs(d.class_id)) % kColors.size()];
    const int x0 = std::clamp(static_cast<int>(std::floor(d.x1)), 0, width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(d.y1)), 0, height - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(d.x2)) - 1, 0, width - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil(d.y2)) - 1, 0, height - 1);
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const bool edge =
            x - x0 < thickness || x1 - x < thickness || y - y0 < thickness || y1 - y < thickness;
        if (edge) {
          std::uint8_t* p = rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3U;
          p[0] = color[0];
          p[1] = color[1];
          p[2] = color[2];
        }
      }
    }
  }
}

struct JpegDecoder::State {
  nvjpegHandle_t handle{};
  nvjpegJpegState_t jpeg{};
};

JpegDecoder::JpegDecoder() : state_(new State) {
  if (nvjpegCreateSimple(&state_->handle) != NVJPEG_STATUS_SUCCESS) {
    delete state_;
    throw std::runtime_error("nvJPEG: cannot create a handle");
  }
  if (nvjpegJpegStateCreate(state_->handle, &state_->jpeg) != NVJPEG_STATUS_SUCCESS) {
    nvjpegDestroy(state_->handle);
    delete state_;
    throw std::runtime_error("nvJPEG: cannot create a decoder state");
  }
}

JpegDecoder::~JpegDecoder() {
  nvjpegJpegStateDestroy(state_->jpeg);
  nvjpegDestroy(state_->handle);
  delete state_;
}

std::string JpegDecoder::decode(std::span<const std::uint8_t> jpeg, int width, int height, bool bgr,
                                std::uint8_t* pixels, std::size_t pitch, cudaStream_t stream) {
  int components = 0;
  nvjpegChromaSubsampling_t subsampling{};
  std::array<int, NVJPEG_MAX_COMPONENT> widths{};
  std::array<int, NVJPEG_MAX_COMPONENT> heights{};
  if (nvjpegGetImageInfo(state_->handle, jpeg.data(), jpeg.size(), &components, &subsampling,
                         widths.data(), heights.data()) != NVJPEG_STATUS_SUCCESS) {
    return "nvJPEG cannot read the image header";
  }
  if (widths[0] != width || heights[0] != height) {
    return "the image is " + std::to_string(widths[0]) + "x" + std::to_string(heights[0]) +
           ", not the " + std::to_string(width) + "x" + std::to_string(height) + " its header gave";
  }
  nvjpegImage_t destination{};
  destination.channel[0] = pixels;
  destination.pitch[0] = pitch;
  const nvjpegStatus_t status =
      nvjpegDecode(state_->handle, state_->jpeg, jpeg.data(), jpeg.size(),
                   bgr ? NVJPEG_OUTPUT_BGRI : NVJPEG_OUTPUT_RGBI, &destination, stream);
  if (status != NVJPEG_STATUS_SUCCESS) {
    return "nvJPEG decode failed (status " + std::to_string(static_cast<int>(status)) + ")";
  }
  return {};
}

}  // namespace yolo26_detection
