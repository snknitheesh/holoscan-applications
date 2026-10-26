// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace depth_to_point_cloud {
namespace {

// Reads one header integer, skipping whitespace and comments.
int read_header_value(std::istream& in, const std::filesystem::path& path) {
  int c = in.peek();
  while (c != EOF && (std::isspace(c) || c == '#')) {
    if (c == '#') {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else {
      in.get();
    }
    c = in.peek();
  }
  int value = -1;
  if (!(in >> value) || value < 0) {
    throw std::runtime_error(path.string() + ": malformed PGM header");
  }
  return value;
}

}  // namespace

DepthImage read_pgm_depth(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(path.string() + ": cannot open");
  }
  char magic[2] = {};
  in.read(magic, 2);
  if (!in || magic[0] != 'P' || magic[1] != '5') {
    throw std::runtime_error(path.string() + ": not a binary PGM (P5) image");
  }
  DepthImage image;
  image.width = read_header_value(in, path);
  image.height = read_header_value(in, path);
  const int max_value = read_header_value(in, path);
  if (image.width <= 0 || image.height <= 0 || image.width > 16384 || image.height > 16384) {
    throw std::runtime_error(path.string() + ": image size out of range");
  }
  if (max_value < 1 || max_value > 65535) {
    throw std::runtime_error(path.string() + ": maxval must be 1..65535");
  }
  if (!std::isspace(in.get())) {
    throw std::runtime_error(path.string() + ": malformed PGM header");
  }
  const std::size_t count = static_cast<std::size_t>(image.width) * image.height;
  const std::size_t sample_bytes = max_value > 255 ? 2 : 1;
  std::vector<std::uint8_t> raw(count * sample_bytes);
  in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (static_cast<std::size_t>(in.gcount()) != raw.size()) {
    throw std::runtime_error(path.string() + ": truncated pixel data");
  }
  image.pixels.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    image.pixels[i] =
        sample_bytes == 2 ? static_cast<std::uint16_t>((raw[2 * i] << 8) | raw[2 * i + 1]) : raw[i];
  }
  return image;
}

void write_pgm_depth(const std::filesystem::path& path, const DepthImage& image) {
  std::ofstream out(path, std::ios::binary);
  out << "P5\n" << image.width << ' ' << image.height << "\n65535\n";
  for (const std::uint16_t value : image.pixels) {
    const char bytes[2] = {static_cast<char>(value >> 8), static_cast<char>(value & 0xFF)};
    out.write(bytes, 2);
  }
  if (!out) {
    throw std::runtime_error(path.string() + ": write failed");
  }
}

std::vector<std::filesystem::path> list_pgm_files(const std::filesystem::path& directory) {
  std::error_code error;
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
      files.push_back(entry.path());
    }
  }
  if (error) {
    throw std::runtime_error(directory.string() + ": cannot list: " + error.message());
  }
  if (files.empty()) {
    throw std::runtime_error(directory.string() + ": no .pgm images");
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::size_t write_ply(const std::filesystem::path& path, const std::vector<float>& xyz,
                      const std::vector<std::uint8_t>* rgb) {
  const std::size_t total = xyz.size() / 3;
  std::size_t finite = 0;
  for (std::size_t i = 0; i < total; ++i) {
    finite +=
        std::isfinite(xyz[3 * i]) && std::isfinite(xyz[3 * i + 1]) && std::isfinite(xyz[3 * i + 2]);
  }
  std::ofstream out(path, std::ios::binary);
  out << "ply\nformat binary_little_endian 1.0\nelement vertex " << finite
      << "\nproperty float x\nproperty float y\nproperty float z\n";
  if (rgb != nullptr) {
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  }
  out << "end_header\n";
  for (std::size_t i = 0; i < total; ++i) {
    const float* p = xyz.data() + 3 * i;
    if (!(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]))) {
      continue;
    }
    out.write(reinterpret_cast<const char*>(p), 3 * sizeof(float));
    if (rgb != nullptr) {
      out.write(reinterpret_cast<const char*>(rgb->data() + 3 * i), 3);
    }
  }
  if (!out) {
    throw std::runtime_error(path.string() + ": write failed");
  }
  return finite;
}

}  // namespace depth_to_point_cloud
