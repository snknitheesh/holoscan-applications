// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Writes synthetic depth frames as 16-bit PGM files (millimeters), for the
// tests that feed the application a folder of depth images.
//
// Usage: write_test_images DIR COUNT WIDTH HEIGHT

#include <charconv>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string_view>

#include "deproject/image_io.hpp"
#include "deproject/synthetic_depth.hpp"

namespace {

int number(const char* text) {
  const std::string_view view(text);
  int value = 0;
  std::from_chars(view.data(), view.data() + view.size(), value);
  return value;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " DIR COUNT WIDTH HEIGHT\n";
    return 2;
  }
  const std::filesystem::path directory(argv[1]);
  const int count = number(argv[2]);
  const int width = number(argv[3]);
  const int height = number(argv[4]);
  std::filesystem::create_directories(directory);
  for (int frame = 0; frame < count; ++frame) {
    const auto f = depth_to_point_cloud::synthetic_frame(width, height,
                                                         static_cast<std::uint32_t>(frame), false);
    depth_to_point_cloud::DepthImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * height);
    for (int v = 0; v < height; ++v) {
      for (int u = 0; u < width; ++u) {
        image.pixels[static_cast<std::size_t>(v) * width + u] =
            depth_to_point_cloud::synthetic_depth_mm(f, u, v);
      }
    }
    char name[32];
    std::snprintf(name, sizeof(name), "depth_%04d.pgm", frame);
    depth_to_point_cloud::write_pgm_depth(directory / name, image);
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
