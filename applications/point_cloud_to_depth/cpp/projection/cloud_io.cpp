// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cloud_io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace point_cloud_to_depth {
namespace {

// The vertex data is read and written as the host stores it.
static_assert(std::endian::native == std::endian::little,
              "PLY files are read and written as little-endian");

// Bounds on a header, so a file that is not PLY is not read to its end.
constexpr int kMaxHeaderLines = 1000;
constexpr std::size_t kMaxHeaderLineBytes = 4096;

constexpr std::array<std::string_view, 3> kPositionNames{"x", "y", "z"};
constexpr std::array<std::string_view, 3> kColorNames{"red", "green", "blue"};

std::runtime_error ply_error(const std::filesystem::path& path, const std::string& message) {
  return std::runtime_error(path.string() + ": " + message);
}

std::vector<std::string> split(const std::string& line) {
  std::istringstream words(line);
  std::vector<std::string> result;
  for (std::string word; words >> word;) {
    result.push_back(word);
  }
  return result;
}

// Reads one header line without its line ending, never more than
// kMaxHeaderLineBytes of it: a longer line sets failbit.
std::string header_line(std::istream& in, const std::filesystem::path& path) {
  std::array<char, kMaxHeaderLineBytes + 1U> buffer{};
  if (!in.getline(buffer.data(), static_cast<std::streamsize>(buffer.size()))) {
    throw ply_error(path, "malformed PLY header");
  }
  // gcount() includes the line ending, which a last line at the end of the
  // file lacks.
  std::string line(buffer.data(), static_cast<std::size_t>(in.gcount()) - (in.eof() ? 0U : 1U));
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  return line;
}

struct ParsedHeader {
  PlyHeader header;
  std::streamoff data_offset{};
};

ParsedHeader parse_header(std::istream& in, const std::filesystem::path& path) {
  const std::string expected_properties =
      "vertex properties must be float x, y, z, optionally followed by uchar red, green, blue";
  if (header_line(in, path) != "ply") {
    throw ply_error(path, "not a PLY file");
  }
  bool format_seen = false;
  bool vertex_seen = false;
  std::vector<std::pair<std::string, std::string>> properties;  // Type and name.
  ParsedHeader parsed;
  for (int lines = 0;; ++lines) {
    if (lines == kMaxHeaderLines) {
      throw ply_error(path, "malformed PLY header");
    }
    const std::string line = header_line(in, path);
    const std::vector<std::string> words = split(line);
    if (words.empty() || words[0] == "comment" || words[0] == "obj_info") {
      continue;
    }
    if (words[0] == "end_header") {
      if (words.size() != 1) {
        throw ply_error(path, "malformed PLY header line '" + line + "'");
      }
      break;
    }
    if (words[0] == "format") {
      if (words.size() != 3 || words[1] != "binary_little_endian" || words[2] != "1.0") {
        throw ply_error(path, "only binary_little_endian 1.0 PLY is supported, not '" + line + "'");
      }
      format_seen = true;
    } else if (words[0] == "element") {
      if (words.size() != 3 || words[1] != "vertex" || vertex_seen) {
        throw ply_error(path, "only one element, vertex, is supported, not '" + line + "'");
      }
      std::uint64_t count = 0;
      const auto [end, error] =
          std::from_chars(words[2].data(), words[2].data() + words[2].size(), count);
      if (error != std::errc() || end != words[2].data() + words[2].size()) {
        throw ply_error(path, "malformed vertex count '" + words[2] + "'");
      }
      if (count > kMaxPlyPoints) {
        throw ply_error(path, "more than " + std::to_string(kMaxPlyPoints) + " points");
      }
      parsed.header.points = count;
      vertex_seen = true;
    } else if (words[0] == "property") {
      if (!vertex_seen || words.size() != 3) {
        throw ply_error(path, expected_properties + ", not '" + line + "'");
      }
      properties.emplace_back(words[1], words[2]);
    } else {
      throw ply_error(path, "malformed PLY header line '" + line + "'");
    }
  }
  if (!format_seen || !vertex_seen) {
    throw ply_error(path, "a PLY header needs a format and a vertex element");
  }
  const auto is_float = [](const std::string& type) {
    return type == "float" || type == "float32";
  };
  const auto is_uchar = [](const std::string& type) { return type == "uchar" || type == "uint8"; };
  bool layout_ok = properties.size() == 3 || properties.size() == 6;
  for (std::size_t i = 0; layout_ok && i < properties.size(); ++i) {
    const auto& [type, name] = properties[i];
    layout_ok = i < 3 ? is_float(type) && name == kPositionNames[i]
                      : is_uchar(type) && name == kColorNames[i - 3];
  }
  if (!layout_ok) {
    throw ply_error(path, expected_properties);
  }
  parsed.header.color = properties.size() == 6;
  // end_header must end its line, so that the data has a known start.
  parsed.data_offset = in.tellg();
  if (in.eof() || parsed.data_offset < 0) {
    throw ply_error(path, "malformed PLY header: end_header must end its line");
  }
  return parsed;
}

void open_for_writing(std::ofstream& out, const std::filesystem::path& path) {
  out.open(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error(path.string() + ": cannot open for writing");
  }
}

void finish_writing(std::ofstream& out, const std::filesystem::path& path) {
  out.flush();
  if (!out) {
    throw std::runtime_error(path.string() + ": write failed");
  }
}

}  // namespace

PlyHeader read_ply_header(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw ply_error(path, "cannot open");
  }
  return parse_header(in, path).header;
}

PointCloud read_ply(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw ply_error(path, "cannot open");
  }
  const ParsedHeader parsed = parse_header(in, path);
  const std::size_t count = parsed.header.points;
  const std::size_t stride = 3U * sizeof(float) + (parsed.header.color ? 3U : 0U);
  // Checked before allocating, so a short file with a large count fails fast.
  std::error_code error;
  const std::uintmax_t file_bytes = std::filesystem::file_size(path, error);
  const std::uintmax_t expected_bytes =
      static_cast<std::uintmax_t>(parsed.data_offset) + count * stride;
  if (error || file_bytes < expected_bytes) {
    throw ply_error(path, "truncated vertex data");
  }
  // The vertex element is the only one, so nothing may follow it.
  if (file_bytes > expected_bytes) {
    throw ply_error(path, "data after the vertex data");
  }
  std::vector<char> data(count * stride);
  in.read(data.data(), static_cast<std::streamsize>(data.size()));
  if (static_cast<std::size_t>(in.gcount()) != data.size()) {
    throw ply_error(path, "truncated vertex data");
  }
  PointCloud cloud;
  cloud.color = parsed.header.color;
  cloud.xyz.resize(count * 3U);
  if (parsed.header.color) {
    cloud.rgb.resize(count * 3U);
  }
  for (std::size_t i = 0; i < count; ++i) {
    const char* vertex = data.data() + i * stride;
    std::memcpy(&cloud.xyz[3U * i], vertex, 3U * sizeof(float));
    if (parsed.header.color) {
      std::memcpy(&cloud.rgb[3U * i], vertex + 3U * sizeof(float), 3U);
    }
  }
  return cloud;
}

void write_ply(const std::filesystem::path& path, std::span<const float> xyz,
               std::span<const std::uint8_t> rgb) {
  if (xyz.size() % 3U != 0U || (!rgb.empty() && rgb.size() != xyz.size())) {
    throw std::invalid_argument("write_ply: need 3 floats, and 0 or 3 colors, per point");
  }
  const std::size_t count = xyz.size() / 3U;
  std::ofstream out;
  open_for_writing(out, path);
  out << "ply\nformat binary_little_endian 1.0\nelement vertex " << count
      << "\nproperty float x\nproperty float y\nproperty float z\n";
  if (!rgb.empty()) {
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  }
  out << "end_header\n";
  for (std::size_t i = 0; i < count; ++i) {
    out.write(reinterpret_cast<const char*>(xyz.data() + 3U * i), 3U * sizeof(float));
    if (!rgb.empty()) {
      out.write(reinterpret_cast<const char*>(rgb.data() + 3U * i), 3U);
    }
  }
  finish_writing(out, path);
}

std::vector<std::filesystem::path> list_ply_files(const std::filesystem::path& directory) {
  std::error_code error;
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.is_regular_file() && entry.path().extension() == ".ply") {
      files.push_back(entry.path());
    }
  }
  if (error) {
    throw std::runtime_error(directory.string() + ": cannot list: " + error.message());
  }
  if (files.empty()) {
    throw std::runtime_error(directory.string() + ": no .ply files");
  }
  std::sort(files.begin(), files.end());
  return files;
}

void write_pgm(const std::filesystem::path& path, int width, int height,
               std::span<const std::uint16_t> samples) {
  std::ofstream out;
  open_for_writing(out, path);
  out << "P5\n" << width << ' ' << height << "\n65535\n";
  // Samples are big-endian, as the format specifies.
  std::vector<char> bytes(samples.size() * 2U);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    bytes[2U * i] = static_cast<char>(samples[i] >> 8U);
    bytes[2U * i + 1U] = static_cast<char>(samples[i] & 0xFFU);
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  finish_writing(out, path);
}

void write_ppm(const std::filesystem::path& path, int width, int height,
               std::span<const std::uint8_t> rgb) {
  std::ofstream out;
  open_for_writing(out, path);
  out << "P6\n" << width << ' ' << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  finish_writing(out, path);
}

}  // namespace point_cloud_to_depth
