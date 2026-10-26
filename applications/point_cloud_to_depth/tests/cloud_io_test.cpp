// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The PLY reader against hand-written files: points with and without colors,
// comments and type-name synonyms, and every unsupported or malformed file
// rejected with an error naming the problem, the header's bounds included. Also the PLY writer, the
// folder listing, and the PGM and PPM writers, byte for byte.
//
// Usage: pcd_cloud_io_test DIR (a scratch folder for the files)

#include "projection/cloud_io.hpp"

#include <unistd.h>

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "test_support.hpp"

namespace point_cloud_to_depth {
namespace {

std::filesystem::path g_directory;

// Little-endian bytes of `values`, as a PLY body holds them.
std::string float_bytes(const std::vector<float>& values) {
  std::string bytes(values.size() * sizeof(float), '\0');
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::filesystem::path write_file(const std::string& name, const std::string& contents) {
  const std::filesystem::path path = g_directory / name;
  std::ofstream(path, std::ios::binary) << contents;
  return path;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string xyz_properties() { return "property float x\nproperty float y\nproperty float z\n"; }

std::string rgb_properties() {
  return "property uchar red\nproperty uchar green\nproperty uchar blue\n";
}

std::string header(const std::string& count, const std::string& properties) {
  return "ply\nformat binary_little_endian 1.0\nelement vertex " + count + "\n" + properties +
         "end_header\n";
}

// Checks that reading `path` fails with a message containing `what`.
void check_rejected(const std::filesystem::path& path, const std::string& what) {
  try {
    static_cast<void>(read_ply(path));
    VERIFY_MSG(false, path.filename().string() + " is rejected");
  } catch (const std::runtime_error& error) {
    VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
               path.filename().string() + ": '" + error.what() + "' names '" + what + "'");
  }
}

void check_points_only() {
  const auto path = write_file("xyz.ply", header("2", xyz_properties()) +
                                              float_bytes({1.0F, -2.0F, 3.5F, 0.25F, 0.0F, -7.0F}));
  const PlyHeader head = read_ply_header(path);
  VERIFY(head.points == 2 && !head.color);
  const PointCloud cloud = read_ply(path);
  VERIFY(cloud.size() == 2);
  VERIFY((cloud.xyz == std::vector<float>{1.0F, -2.0F, 3.5F, 0.25F, 0.0F, -7.0F}));
  VERIFY(cloud.rgb.empty() && !cloud.color);
}

void check_points_with_colors() {
  std::string body = float_bytes({1.0F, 2.0F, 3.0F});
  body += std::string("\x0A\x80\xFF", 3);
  body += float_bytes({4.0F, 5.0F, 6.0F});
  body += std::string("\x00\x01\x02", 3);
  // Comments, obj_info, CRLF line ends and the float32/uint8 synonyms.
  const auto path = write_file(
      "xyzrgb.ply",
      "ply\r\nformat binary_little_endian 1.0\r\ncomment made by hand\r\nobj_info test\r\n"
      "element vertex 2\r\nproperty float32 x\r\nproperty float32 y\r\nproperty float32 z\r\n"
      "property uint8 red\r\nproperty uint8 green\r\nproperty uint8 blue\r\nend_header\r\n" +
          body);
  const PlyHeader head = read_ply_header(path);
  VERIFY(head.points == 2 && head.color);
  const PointCloud cloud = read_ply(path);
  VERIFY((cloud.xyz == std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
  VERIFY((cloud.rgb == std::vector<std::uint8_t>{10, 128, 255, 0, 1, 2}));
  VERIFY(cloud.color);
}

void check_non_finite_points_kept() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const PointCloud cloud = read_ply(
      write_file("nan.ply", header("1", xyz_properties()) + float_bytes({nan, 1.0F, 2.0F})));
  VERIFY(cloud.size() == 1 && std::isnan(cloud.xyz[0]) && cloud.xyz[2] == 2.0F);
}

void check_empty_cloud() {
  const PointCloud plain = read_ply(write_file("empty.ply", header("0", xyz_properties())));
  VERIFY(plain.size() == 0 && !plain.color);
  const PointCloud colored =
      read_ply(write_file("empty_rgb.ply", header("0", xyz_properties() + rgb_properties())));
  VERIFY(colored.size() == 0 && colored.color && colored.rgb.empty());
}

void check_round_trip() {
  const std::vector<float> xyz{1.5F, -2.0F, 3.0F, std::numeric_limits<float>::infinity(),
                               0.0F, 1.0F};
  const std::vector<std::uint8_t> rgb{1, 2, 3, 250, 251, 252};
  write_ply(g_directory / "written.ply", xyz, rgb);
  const PointCloud colored = read_ply(g_directory / "written.ply");
  VERIFY(colored.xyz == xyz && colored.rgb == rgb);
  write_ply(g_directory / "written_xyz.ply", xyz, {});
  const PointCloud plain = read_ply(g_directory / "written_xyz.ply");
  VERIFY(plain.xyz == xyz && plain.rgb.empty());
  VERIFY(testing::throws<std::invalid_argument>(
      [] { write_ply(g_directory / "bad.ply", std::vector<float>{1.0F, 2.0F}, {}); }));
  VERIFY(testing::throws<std::invalid_argument>(
      [&] { write_ply(g_directory / "bad.ply", xyz, std::vector<std::uint8_t>{1, 2, 3}); }));
}

void check_malformed() {
  const std::string one_point = float_bytes({1.0F, 2.0F, 3.0F});
  check_rejected(g_directory / "missing.ply", "cannot open");
  check_rejected(write_file("not_ply.ply", "PLY\n"), "not a PLY file");
  check_rejected(write_file("empty_file.ply", ""), "malformed PLY header");
  check_rejected(write_file("ascii.ply", "ply\nformat ascii 1.0\nelement vertex 1\n" +
                                             xyz_properties() + "end_header\n1 2 3\n"),
                 "only binary_little_endian 1.0");
  check_rejected(write_file("big_endian.ply",
                            "ply\nformat binary_big_endian 1.0\nelement vertex "
                            "1\n" +
                                xyz_properties() + "end_header\n" + one_point),
                 "only binary_little_endian 1.0");
  check_rejected(
      write_file("no_format.ply", "ply\nelement vertex 1\n" + xyz_properties() + "end_header\n"),
      "needs a format and a vertex element");
  check_rejected(write_file("no_vertex.ply", "ply\nformat binary_little_endian 1.0\nend_header\n"),
                 "needs a format and a vertex element");
  check_rejected(write_file("faces.ply", header("1", xyz_properties() +
                                                         "element face 0\nproperty list uchar int "
                                                         "vertex_indices\n") +
                                             one_point),
                 "only one element, vertex");
  check_rejected(
      write_file("double.ply", header("1",
                                      "property double x\nproperty double y\nproperty double "
                                      "z\n")),
      "vertex properties must be float x, y, z");
  check_rejected(write_file("normals.ply", header("1", xyz_properties() + "property float nx\n")),
                 "vertex properties must be float x, y, z");
  check_rejected(write_file("no_z.ply", header("1", "property float x\nproperty float y\n")),
                 "vertex properties must be float x, y, z");
  check_rejected(
      write_file("swapped.ply", header("1",
                                       "property float y\nproperty float x\nproperty float "
                                       "z\n")),
      "vertex properties must be float x, y, z");
  check_rejected(write_file("red_only.ply", header("1", xyz_properties() + "property uchar red\n")),
                 "vertex properties must be float x, y, z");
  check_rejected(
      write_file("float_color.ply", header("1", xyz_properties() + "property float red\nproperty "
                                                                   "float green\nproperty float "
                                                                   "blue\n")),
      "vertex properties must be float x, y, z");
  check_rejected(write_file("bad_count.ply", header("x1", xyz_properties())),
                 "malformed vertex count");
  check_rejected(write_file("negative_count.ply", header("-1", xyz_properties())),
                 "malformed vertex count");
  check_rejected(write_file("huge_count.ply", header("2000000000", xyz_properties())), "more than");
  check_rejected(write_file("unknown_line.ply", header("1", xyz_properties() + "texture foo\n")),
                 "malformed PLY header line");
  check_rejected(write_file("no_end.ply",
                            "ply\nformat binary_little_endian 1.0\nelement vertex "
                            "1\n" +
                                xyz_properties()),
                 "malformed PLY header");
  check_rejected(write_file("truncated.ply", header("2", xyz_properties()) + one_point),
                 "truncated");
  check_rejected(write_file("truncated_color.ply",
                            header("1", xyz_properties() + rgb_properties()) + one_point + "\x01"),
                 "truncated");
  // A large count in a short file is caught before reading.
  check_rejected(write_file("short_large.ply", header("100000000", xyz_properties()) + one_point),
                 "truncated");
}

// The header is read within its bounds: a line or a header too long, or a
// file with no line end at all, fails without reading further.
void check_header_bounds() {
  const std::string body = float_bytes({1.0F, 2.0F, 3.0F});
  check_rejected(write_file("long_first_line.ply", std::string(5000, 'p') + "\n"),
                 "malformed PLY header");
  const std::string long_comment = "comment " + std::string(4089, 'c') + "\n";  // 4097 bytes.
  check_rejected(write_file("long_line.ply", "ply\n" + long_comment +
                                                 header("1", xyz_properties()).substr(4) + body),
                 "malformed PLY header");
  // A line of exactly 4096 bytes is accepted.
  const std::string longest_comment = "comment " + std::string(4088, 'c') + "\n";
  VERIFY(
      read_ply(write_file("longest_line.ply", "ply\n" + longest_comment +
                                                  header("1", xyz_properties()).substr(4) + body))
          .size() == 1);
  std::string comments;
  for (int i = 0; i < 1000; ++i) comments += "comment " + std::to_string(i) + "\n";
  check_rejected(write_file("long_header.ply",
                            "ply\n" + comments + header("1", xyz_properties()).substr(4) + body),
                 "malformed PLY header");
}

// A first line that never ends is not read past the line bound: from a pipe
// offering a megabyte without a line end, the reader takes no more than the
// bound, its stream buffer and the pipe's own buffer hold, and stops.
void check_endless_line_not_read() {
  int ends[2];
  VERIFY(pipe(ends) == 0);
  std::signal(SIGPIPE, SIG_IGN);  // A write with no reader left fails instead.
  constexpr std::size_t kOffered = std::size_t{1} << 20;
  std::size_t written = 0;
  std::thread writer([&ends, &written] {
    const std::string chunk(4096, 'p');
    while (written < kOffered) {
      const ssize_t sent = write(ends[1], chunk.data(), chunk.size());
      if (sent <= 0) {
        break;
      }
      written += static_cast<std::size_t>(sent);
    }
    close(ends[1]);
  });
  check_rejected("/dev/fd/" + std::to_string(ends[0]), "malformed PLY header");
  close(ends[0]);  // The last reader: the writer, if it still waits, fails.
  writer.join();
  VERIFY_MSG(written < kOffered / 4,
             std::to_string(written) + " bytes taken from a line that never ends");
}

// The vertex count: beyond 64 bits, and either side of the largest accepted.
void check_vertex_counts() {
  check_rejected(
      write_file("count_beyond_64_bits.ply", header("99999999999999999999999", xyz_properties())),
      "malformed vertex count");
  const PlyHeader largest = read_ply_header(
      write_file("count_largest.ply", header(std::to_string(kMaxPlyPoints), xyz_properties())));
  VERIFY(largest.points == kMaxPlyPoints);
  check_rejected(write_file("count_too_large.ply",
                            header(std::to_string(kMaxPlyPoints + 1U), xyz_properties())),
                 "more than");
}

// Nothing may follow end_header on its line, nor the vertex data in the file.
void check_trailing_data() {
  const std::string body = float_bytes({1.0F, 2.0F, 3.0F});
  std::string garbage_header = header("1", xyz_properties());
  garbage_header.insert(garbage_header.size() - 1, " garbage");
  check_rejected(write_file("end_header_garbage.ply", garbage_header + body),
                 "malformed PLY header line 'end_header garbage'");
  check_rejected(write_file("trailing.ply", header("1", xyz_properties()) + body + "x"),
                 "data after the vertex data");
  std::string unended = header("1", xyz_properties());
  unended.pop_back();
  check_rejected(write_file("end_header_unended.ply", unended), "end_header must end its line");
}

void check_listing() {
  const std::filesystem::path folder = g_directory / "listing";
  std::filesystem::create_directories(folder);
  for (const char* name : {"b.ply", "a.ply", "c.pcd"}) {
    std::ofstream(folder / name) << "ply\n";
  }
  const auto files = list_ply_files(folder);
  VERIFY(files.size() == 2 && files[0].filename() == "a.ply" && files[1].filename() == "b.ply");
  const std::filesystem::path empty = g_directory / "empty_folder";
  std::filesystem::create_directories(empty);
  VERIFY(testing::throws<std::runtime_error>([&] { static_cast<void>(list_ply_files(empty)); }));
  VERIFY(testing::throws<std::runtime_error>(
      [&] { static_cast<void>(list_ply_files(g_directory / "no_such_folder")); }));
}

void check_images() {
  write_pgm(g_directory / "depth.pgm", 3, 1, std::vector<std::uint16_t>{0, 258, 65535});
  VERIFY(read_file(g_directory / "depth.pgm") ==
         std::string("P5\n3 1\n65535\n\x00\x00\x01\x02\xFF\xFF", 19));
  write_ppm(g_directory / "color.ppm", 2, 1, std::vector<std::uint8_t>{1, 2, 3, 250, 251, 252});
  VERIFY(read_file(g_directory / "color.ppm") ==
         std::string("P6\n2 1\n255\n\x01\x02\x03\xFA\xFB\xFC", 17));
  // A directory in the file's place cannot be written.
  std::filesystem::create_directories(g_directory / "blocked.pgm");
  VERIFY(testing::throws<std::runtime_error>(
      [] { write_pgm(g_directory / "blocked.pgm", 1, 1, std::vector<std::uint16_t>{1}); }));
}

}  // namespace
}  // namespace point_cloud_to_depth

int main(int argc, char** argv) try {
  namespace pcd = point_cloud_to_depth;
  if (argc != 2) {
    return 2;
  }
  pcd::g_directory = argv[1];
  std::filesystem::remove_all(pcd::g_directory);
  std::filesystem::create_directories(pcd::g_directory);
  return pcd::testing::run({
      {"points without colors", pcd::check_points_only},
      {"points with colors, comments and synonyms", pcd::check_points_with_colors},
      {"non-finite points are read as they are", pcd::check_non_finite_points_kept},
      {"a cloud without points", pcd::check_empty_cloud},
      {"a written cloud reads back", pcd::check_round_trip},
      {"unsupported and malformed files are rejected", pcd::check_malformed},
      {"the header is read within its bounds", pcd::check_header_bounds},
      {"a line that never ends is not read on", pcd::check_endless_line_not_read},
      {"vertex counts at and beyond the limits", pcd::check_vertex_counts},
      {"nothing may follow the header line or the data", pcd::check_trailing_data},
      {"folders are listed in name order", pcd::check_listing},
      {"PGM and PPM images", pcd::check_images},
  });
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
