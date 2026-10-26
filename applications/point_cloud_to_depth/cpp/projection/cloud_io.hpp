// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Files the application reads and writes: binary little-endian PLY point
// clouds (float x, y, z per vertex, optionally uchar red, green, blue), as
// point-cloud tools save them, and the depth and color images it makes, as
// 16-bit PGM and 8-bit PPM files that any image viewer opens.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace point_cloud_to_depth {

// The largest PLY file the reader accepts, in points.
inline constexpr std::uint64_t kMaxPlyPoints = std::uint64_t{1} << 30;

struct PointCloud {
  std::vector<float> xyz;         // x, y, z per point.
  std::vector<std::uint8_t> rgb;  // Red, green, blue per point; empty without colors.
  bool color{};                   // The file has colors, even with no points.

  [[nodiscard]] std::size_t size() const { return xyz.size() / 3U; }
};

struct PlyHeader {
  std::uint64_t points{};
  bool color{};
};

// Reads the header of a binary little-endian PLY file whose only element is
// `vertex` with the properties float x, y, z, optionally followed by uchar
// red, green, blue (float32 and uint8 are accepted as type names).
// @throws std::runtime_error naming the file and what it does not support.
[[nodiscard]] PlyHeader read_ply_header(const std::filesystem::path& path);

// Reads a PLY file read_ply_header() accepts.
// @throws std::runtime_error naming the file and the problem.
[[nodiscard]] PointCloud read_ply(const std::filesystem::path& path);

// Writes `xyz` (x, y, z per point) and, unless empty, `rgb` (red, green, blue
// per point) as a binary little-endian PLY file. Every point is written, the
// non-finite ones included.
// @throws std::invalid_argument if the sizes disagree.
// @throws std::runtime_error if the file cannot be written.
void write_ply(const std::filesystem::path& path, std::span<const float> xyz,
               std::span<const std::uint8_t> rgb);

// The .ply files directly in `directory`, sorted by name.
// @throws std::runtime_error if `directory` cannot be listed or holds none.
[[nodiscard]] std::vector<std::filesystem::path> list_ply_files(
    const std::filesystem::path& directory);

// Writes `width` x `height` packed samples as a 16-bit binary PGM (P5).
// @throws std::runtime_error if the file cannot be written.
void write_pgm(const std::filesystem::path& path, int width, int height,
               std::span<const std::uint16_t> samples);

// Writes `width` x `height` packed RGB8 pixels as a binary PPM (P6).
// @throws std::runtime_error if the file cannot be written.
void write_ppm(const std::filesystem::path& path, int width, int height,
               std::span<const std::uint8_t> rgb);

}  // namespace point_cloud_to_depth
