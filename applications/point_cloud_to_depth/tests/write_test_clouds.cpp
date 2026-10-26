// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Writes synthetic clouds (frames 0, 1, ...) as PLY files, in the body frame
// and with colors unless `xyz` is given, for the tests that feed the
// application a folder of PLY files.
//
// Usage: write_test_clouds DIR COUNT WIDTH HEIGHT [xyz]

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

#include "projection/cloud_io.hpp"
#include "projection/synthetic_scene.hpp"

namespace {

int number(const char* text) {
  const std::string_view view(text);
  int value = 0;
  std::from_chars(view.data(), view.data() + view.size(), value);
  return value;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 5 && !(argc == 6 && std::string_view(argv[5]) == "xyz")) {
    std::cerr << "usage: " << argv[0] << " DIR COUNT WIDTH HEIGHT [xyz]\n";
    return 2;
  }
  const bool colors = argc == 5;
  const std::filesystem::path directory(argv[1]);
  const int count = number(argv[2]);
  const int width = number(argv[3]);
  const int height = number(argv[4]);
  std::filesystem::create_directories(directory);
  for (int frame = 0; frame < count; ++frame) {
    std::vector<float> xyz;
    std::vector<std::uint8_t> rgb;
    point_cloud_to_depth::synthetic_cloud(
        point_cloud_to_depth::synthetic_frame(width, height, static_cast<std::uint32_t>(frame),
                                              false),
        false, xyz, rgb);
    char name[32];
    std::snprintf(name, sizeof(name), "cloud_%04d.ply", frame);
    point_cloud_to_depth::write_ply(directory / name, xyz,
                                    colors ? rgb : std::vector<std::uint8_t>{});
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
