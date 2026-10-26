// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The PGM reader against hand-written files: 8-bit samples widened, 16-bit
// samples decoded big-endian, header comments, and every malformed file
// rejected with an error naming the problem. Also the folder listing.
//
// Usage: depth_image_io_test DIR (a scratch folder for the files)

#include "deproject/image_io.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "test_support.hpp"

namespace depth_to_point_cloud {
namespace {

std::filesystem::path g_directory;

// Writes `header` followed by `pixels` to DIR/`name`.
std::filesystem::path write_file(const std::string& name, const std::string& header,
                                 const std::vector<std::uint8_t>& pixels) {
  const std::filesystem::path path = g_directory / name;
  std::ofstream out(path, std::ios::binary);
  out << header;
  out.write(reinterpret_cast<const char*>(pixels.data()),
            static_cast<std::streamsize>(pixels.size()));
  return path;
}

// Checks that reading `path` fails with a message containing `what`.
void check_rejected(const std::filesystem::path& path, const std::string& what) {
  try {
    static_cast<void>(read_pgm_depth(path));
    VERIFY_MSG(false, path.filename().string() + " is rejected");
  } catch (const std::runtime_error& error) {
    VERIFY_MSG(std::string(error.what()).find(what) != std::string::npos,
               path.filename().string() + ": '" + error.what() + "' names '" + what + "'");
  }
}

void check_8bit() {
  const DepthImage image =
      read_pgm_depth(write_file("8bit.pgm", "P5\n3 2\n255\n", {0, 1, 2, 127, 128, 255}));
  VERIFY(image.width == 3 && image.height == 2);
  VERIFY((image.pixels == std::vector<std::uint16_t>{0, 1, 2, 127, 128, 255}));
}

void check_16bit_big_endian() {
  // Most significant byte first: 0x0102 = 258, 0xFF00 = 65280, 0x00FF = 255.
  const DepthImage image = read_pgm_depth(
      write_file("16bit.pgm", "P5\n3 1\n65535\n", {0x01, 0x02, 0xFF, 0x00, 0x00, 0xFF}));
  VERIFY(image.width == 3 && image.height == 1);
  VERIFY((image.pixels == std::vector<std::uint16_t>{258, 65280, 255}));
  // A maxval above 255 means two bytes per sample, whatever its value.
  const DepthImage small =
      read_pgm_depth(write_file("16bit_small.pgm", "P5\n2 1\n256\n", {0x01, 0x00, 0x00, 0x07}));
  VERIFY((small.pixels == std::vector<std::uint16_t>{256, 7}));
}

void check_header_comments() {
  const DepthImage image = read_pgm_depth(
      write_file("comments.pgm", "P5\n# a comment\n2 # width\n1\n# maxval next\n255\n", {9, 8}));
  VERIFY(image.width == 2 && image.height == 1);
  VERIFY((image.pixels == std::vector<std::uint16_t>{9, 8}));
}

void check_round_trip() {
  const DepthImage written{.width = 2, .height = 2, .pixels = {0, 1, 4095, 65535}};
  write_pgm_depth(g_directory / "round_trip.pgm", written);
  const DepthImage read = read_pgm_depth(g_directory / "round_trip.pgm");
  VERIFY(read.width == 2 && read.height == 2 && read.pixels == written.pixels);
}

void check_malformed() {
  check_rejected(g_directory / "missing.pgm", "cannot open");
  check_rejected(write_file("ascii.pgm", "P2\n1 1\n255\n", {0}), "not a binary PGM");
  check_rejected(write_file("empty.pgm", "", {}), "not a binary PGM");
  check_rejected(write_file("no_size.pgm", "P5\nx 1\n255\n", {0}), "malformed PGM header");
  check_rejected(write_file("negative.pgm", "P5\n-1 1\n255\n", {0}), "malformed PGM header");
  check_rejected(write_file("zero_width.pgm", "P5\n0 1\n255\n", {}), "image size out of range");
  check_rejected(write_file("huge.pgm", "P5\n16385 1\n255\n", {}), "image size out of range");
  check_rejected(write_file("maxval_zero.pgm", "P5\n1 1\n0\n", {0}), "maxval must be 1..65535");
  check_rejected(write_file("maxval_big.pgm", "P5\n1 1\n65536\n", {0, 0}),
                 "maxval must be 1..65535");
  check_rejected(write_file("no_separator.pgm", "P5\n1 1\n255", {}), "malformed PGM header");
  check_rejected(write_file("truncated_8bit.pgm", "P5\n2 2\n255\n", {1, 2, 3}),
                 "truncated pixel data");
  check_rejected(write_file("truncated_16bit.pgm", "P5\n2 1\n65535\n", {1, 2, 3}),
                 "truncated pixel data");
}

void check_listing() {
  const std::filesystem::path folder = g_directory / "listing";
  std::filesystem::create_directories(folder);
  for (const char* name : {"b.pgm", "a.pgm", "c.png"}) {
    std::ofstream(folder / name) << "P5\n1 1\n255\n";
  }
  const auto files = list_pgm_files(folder);
  VERIFY(files.size() == 2 && files[0].filename() == "a.pgm" && files[1].filename() == "b.pgm");
  const std::filesystem::path empty = g_directory / "empty_folder";
  std::filesystem::create_directories(empty);
  VERIFY(testing::throws<std::runtime_error>([&] { static_cast<void>(list_pgm_files(empty)); }));
  VERIFY(testing::throws<std::runtime_error>(
      [&] { static_cast<void>(list_pgm_files(g_directory / "no_such_folder")); }));
}

}  // namespace
}  // namespace depth_to_point_cloud

int main(int argc, char** argv) try {
  namespace dpc = depth_to_point_cloud;
  if (argc != 2) {
    return 2;
  }
  dpc::g_directory = argv[1];
  std::filesystem::remove_all(dpc::g_directory);
  std::filesystem::create_directories(dpc::g_directory);
  return dpc::testing::run({
      {"8-bit samples are widened", dpc::check_8bit},
      {"16-bit samples are big-endian", dpc::check_16bit_big_endian},
      {"header comments are skipped", dpc::check_header_comments},
      {"a written image reads back", dpc::check_round_trip},
      {"malformed files are rejected", dpc::check_malformed},
      {"folders are listed in name order", dpc::check_listing},
  });
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
