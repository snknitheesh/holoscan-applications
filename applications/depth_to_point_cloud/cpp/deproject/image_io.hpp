// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Files the application reads and writes: binary PGM depth images (8- or
// 16-bit, as depth cameras and image tools save them) and PLY point clouds,
// which any point-cloud viewer opens.

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace depth_to_point_cloud {

struct DepthImage {
  int width{};
  int height{};
  std::vector<std::uint16_t> pixels;  // Rows packed; 8-bit images are widened.
};

// Reads a binary PGM (P5) with maxval up to 65535; 16-bit samples are
// big-endian, as the format specifies.
// @throws std::runtime_error naming the file and the problem.
[[nodiscard]] DepthImage read_pgm_depth(const std::filesystem::path& path);

// Writes a 16-bit binary PGM.
// @throws std::runtime_error if the file cannot be written.
void write_pgm_depth(const std::filesystem::path& path, const DepthImage& image);

// The .pgm files directly in `directory`, sorted by name.
// @throws std::runtime_error if `directory` cannot be listed or holds none.
[[nodiscard]] std::vector<std::filesystem::path> list_pgm_files(
    const std::filesystem::path& directory);

// Writes the finite points of an organized cloud (`xyz` holds width x height
// x 3 floats) as a binary little-endian PLY, with colors when `rgb` (width x
// height x 3 bytes) is given. Returns the number of points written.
// @throws std::runtime_error if the file cannot be written.
std::size_t write_ply(const std::filesystem::path& path, const std::vector<float>& xyz,
                      const std::vector<std::uint8_t>* rgb);

}  // namespace depth_to_point_cloud
