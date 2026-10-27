// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The host-side pieces, without a GPU: matching detections to truth and to
// expectations, the expectation and label files, image file headers, PPM
// reading and writing, encodings to RGB, box drawing, the stub model file and
// the synthetic scene.
//
// Usage: yolo26_core_test DIR (a scratch folder for the files)

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "detection/image_files.hpp"
#include "detection/matching.hpp"
#include "detection/stub_model.hpp"
#include "detection/synthetic_scene.hpp"
#include "test_support.hpp"

namespace yolo26_detection {
namespace {

std::filesystem::path g_directory;

std::filesystem::path write(const std::string& name, const std::string& content) {
  const auto path = g_directory / name;
  std::ofstream(path, std::ios::binary) << content;
  return path;
}

BoxDetection box(float x1, float y1, float x2, float y2, int class_id = 0, float score = 0.9F) {
  return BoxDetection{x1, y1, x2, y2, score, class_id};
}

void check_iou() {
  VERIFY(iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0F);
  VERIFY(iou(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0F);
  VERIFY(iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == 50.0F / 150.0F);
  VERIFY(iou(box(0, 0, 0, 10), box(0, 0, 0, 10)) == 0.0F);
  VERIFY(edge_error(box(0, 0, 10, 10), box(1, -2, 10.5F, 10)) == 2.0F);
}

void check_match_truth() {
  const std::vector<BoxDetection> truth{box(0, 0, 10, 10, 0), box(20, 20, 40, 40, 1, 0.8F)};
  // Exact within the tolerance.
  TruthMatch m = match_truth(std::vector{box(0.5F, 0, 10, 9.5F, 0), box(20, 21, 40, 40, 1, 0.8F)},
                             truth, 1.0F, 1e-6F);
  VERIFY(m.missed == 0 && m.extra == 0 && m.max_edge_error == 1.0F);
  // An edge too far off: one missed, one extra, and how far.
  m = match_truth(std::vector{box(0, 0, 13, 10, 0), box(20, 20, 40, 40, 1, 0.8F)}, truth, 1.0F,
                  1e-6F);
  VERIFY(m.missed == 1 && m.extra == 1 && m.max_edge_error == 3.0F);
  // The wrong class, the wrong score, a duplicate.
  m = match_truth(std::vector{box(0, 0, 10, 10, 1), box(20, 20, 40, 40, 1, 0.7F)}, truth, 1.0F,
                  1e-6F);
  VERIFY(m.missed == 2 && m.extra == 2);
  m = match_truth(
      std::vector{box(0, 0, 10, 10, 0), box(0, 0, 10, 10, 0), box(20, 20, 40, 40, 1, 0.8F)}, truth,
      1.0F, 1e-6F);
  VERIFY(m.missed == 0 && m.extra == 1);
  m = match_truth(std::vector<BoxDetection>{}, truth, 1.0F, 1e-6F);
  VERIFY(m.missed == 2 && m.extra == 0);
}

void check_count_matched() {
  const std::vector<Expectation> expected{{"a.jpg", box(0, 0, 100, 100, 15)},
                                          {"a.jpg", box(200, 0, 300, 100, 15)}};
  // One detection cannot match two expectations.
  VERIFY(count_matched(std::vector{box(0, 0, 100, 100, 15)}, expected, 0.8F) == 1);
  VERIFY(count_matched(std::vector{box(5, 5, 100, 100, 15), box(200, 0, 290, 100, 15)}, expected,
                       0.8F) == 2);
  VERIFY(count_matched(std::vector{box(0, 0, 100, 100, 16)}, expected, 0.8F) == 0);
  VERIFY(count_matched(std::vector{box(0, 0, 50, 100, 15)}, expected, 0.8F) == 0);
}

void check_expectation_file() {
  const auto good = write("good.csv",
                          "# image,class_id,x1,y1,x2,y2\n"
                          "\n"
                          "a.jpg,15,344,24.5,640,374.8\r\n"
                          " b.ppm , 0 , 1 , 2 , 3 , 4 \n");
  const auto e = read_expectations(good);
  VERIFY(e.size() == 2);
  if (e.size() == 2) {
    VERIFY(e[0].image == "a.jpg" && e[0].box.class_id == 15 && e[0].box.x1 == 344.0F &&
           e[0].box.y2 == 374.8F);
    VERIFY(e[1].image == "b.ppm" && e[1].box.x2 == 3.0F);
  }
  for (const char* bad : {"a.jpg,15,1,2,3\n", "a.jpg,x,1,2,3,4\n", "a.jpg,1,3,2,1,4\n",
                          ",1,1,2,3,4\n", "a.jpg,-1,1,2,3,4\n", "a.jpg,1,1,2,3,4,5\n"}) {
    const auto path = write("bad.csv", std::string("# header\n") + bad);
    VERIFY_MSG(testing::throws<std::runtime_error>([&] { (void)read_expectations(path); }),
               std::string("rejects ") + bad);
  }
  VERIFY(testing::throws<std::runtime_error>(
      [] { (void)read_expectations(g_directory / "missing.csv"); }));
}

void check_labels() {
  const auto labels = read_labels(write("labels.txt", "person\nbicycle\r\ntraffic light\n\n"));
  VERIFY(labels == (std::vector<std::string>{"person", "bicycle", "traffic light"}));
  VERIFY(testing::throws<std::runtime_error>([] { (void)read_labels(write("empty.txt", "\n")); }));
}

// A minimal JPEG header: SOI, an APP0 segment, then a frame header.
std::string jpeg_header(unsigned char frame_marker, int width, int height, int precision = 8,
                        int components = 3) {
  std::string bytes = "\xFF\xD8";
  bytes += std::string("\xFF\xE0\x00\x04JF", 6);
  bytes += '\xFF';
  bytes += static_cast<char>(frame_marker);
  bytes += std::string("\x00\x11", 2);
  bytes += static_cast<char>(precision);
  bytes += static_cast<char>(height >> 8);
  bytes += static_cast<char>(height & 0xFF);
  bytes += static_cast<char>(width >> 8);
  bytes += static_cast<char>(width & 0xFF);
  bytes += static_cast<char>(components);
  bytes += std::string(9, '\0');
  return bytes;
}

void check_image_headers() {
  ImageFileInfo info = read_image_info(write("baseline.jpg", jpeg_header(0xC0, 810, 1080)));
  VERIFY(info.kind == ImageFileKind::kJpeg && info.width == 810 && info.height == 1080);
  info = read_image_info(write("progressive.JPEG", jpeg_header(0xC2, 640, 480, 8, 1)));
  VERIFY(info.width == 640 && info.height == 480);
  info = read_image_info(write("comment.ppm", "P6\n# a comment\n 33 \t17\n255\n"));
  VERIFY(info.kind == ImageFileKind::kPpm && info.width == 33 && info.height == 17);
  const std::vector<std::pair<std::string, std::string>> bad{
      {"arithmetic.jpg", jpeg_header(0xC9, 10, 10)},
      {"twelve_bit.jpg", jpeg_header(0xC0, 10, 10, 12)},
      {"cmyk.jpg", jpeg_header(0xC0, 10, 10, 8, 4)},
      {"zero.jpg", jpeg_header(0xC0, 0, 10)},
      {"truncated.jpg", jpeg_header(0xC0, 10, 10).substr(0, 9)},
      {"scan_first.jpg", std::string("\xFF\xD8\xFF\xDA\x00\x02", 6)},
      {"png.jpg", "\x89PNG\r\n"},
      {"ascii.ppm", "P3\n2 2\n255\n"},
      {"sixteen_bit.ppm", "P6\n2 2\n65535\n"},
      {"huge.ppm", "P6\n20000 2\n255\n"},
      {"image.png", "P6\n2 2\n255\n"},
  };
  for (const auto& [name, content] : bad) {
    const auto path = write(name, content);
    VERIFY_MSG(testing::throws<std::runtime_error>([&] { (void)read_image_info(path); }),
               "rejects " + name);
  }
}

void check_listing() {
  const auto folder = g_directory / "listing";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder / "sub.jpg");
  for (const char* name : {"b.JPG", "a.ppm", "c.jpeg", "d.png", "e.txt"}) {
    std::ofstream(folder / name) << "x";
  }
  const auto files = list_image_files(folder);
  VERIFY(files == (std::vector<std::filesystem::path>{folder / "a.ppm", folder / "b.JPG",
                                                      folder / "c.jpeg"}));
  VERIFY(list_image_files(folder / "a.ppm") == std::vector{folder / "a.ppm"});
  VERIFY(testing::throws<std::runtime_error>([&] { (void)list_image_files(folder / "d.png"); }));
  VERIFY(testing::throws<std::runtime_error>([&] { (void)list_image_files(folder / "missing"); }));
  std::filesystem::create_directories(folder / "empty");
  VERIFY(testing::throws<std::runtime_error>([&] { (void)list_image_files(folder / "empty"); }));
}

void check_ppm_round_trip() {
  std::vector<std::uint8_t> rgb(5 * 3 * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i * 5);
  const auto path = g_directory / "round_trip.ppm";
  write_ppm(path, 5, 3, rgb);
  int width = 0;
  int height = 0;
  VERIFY(read_ppm(path, width, height) == rgb && width == 5 && height == 3);
  const auto truncated = write("truncated.ppm", "P6\n5 3\n255\n" + std::string(10, 'x'));
  VERIFY(testing::throws<std::runtime_error>([&] { (void)read_ppm(truncated, width, height); }));
  VERIFY(testing::throws<std::runtime_error>([&] { write_ppm(path, 5, 4, rgb); }));
}

void check_to_rgb() {
  // One pixel (10, 20, 30) in each layout, rows padded to 8 bytes.
  const std::array<std::array<std::uint8_t, 8>, 4> pixels{{{10, 20, 30, 0, 0, 0, 0, 0},
                                                           {30, 20, 10, 0, 0, 0, 0, 0},
                                                           {10, 20, 30, 255, 0, 0, 0, 0},
                                                           {30, 20, 10, 255, 0, 0, 0, 0}}};
  const std::array<PixelFormat, 4> formats{PixelFormat::kRgb8, PixelFormat::kBgr8,
                                           PixelFormat::kRgba8, PixelFormat::kBgra8};
  for (std::size_t i = 0; i < formats.size(); ++i) {
    VERIFY(to_rgb(pixels[i].data(), 8, 1, 1, formats[i]) ==
           (std::vector<std::uint8_t>{10, 20, 30}));
  }
}

void check_draw_boxes() {
  constexpr int kW = 20;
  constexpr int kH = 10;
  std::vector<std::uint8_t> rgb(kW * kH * 3, 0);
  const std::vector<BoxDetection> boxes{box(2, 2, 12, 8, 0), box(15, -5, 40, 40, 1)};
  draw_boxes(rgb, kW, kH, boxes, 1);
  const auto lit = [&](int x, int y) { return rgb[(y * kW + x) * 3] | rgb[(y * kW + x) * 3 + 1]; };
  VERIFY(lit(2, 2) && lit(11, 7) && lit(6, 2) && lit(2, 5));  // Edges of the first box.
  VERIFY(!lit(6, 5) && !lit(1, 1) && !lit(12, 8));            // Inside it, and outside.
  VERIFY(lit(15, 0) && lit(19, 9) && !lit(17, 5));            // The second, clipped.
}

void check_stub_model_file() {
  VERIFY(stub_model_onnx(640) == stub_model_onnx(640));
  VERIFY(stub_model_onnx(640) != stub_model_onnx(320));
  VERIFY(testing::throws<std::invalid_argument>([] { (void)stub_model_onnx(16); }));
  const auto folder = g_directory / "stub";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder);
  const auto path = write_stub_model(folder, 320);
  VERIFY(path == folder / "yolo26_stub_320.onnx");
  // An unchanged file is left alone, so an engine cached beside it stays
  // valid; a changed one is replaced.
  const auto written = std::filesystem::last_write_time(path);
  VERIFY(write_stub_model(folder, 320) == path &&
         std::filesystem::last_write_time(path) == written);
  std::ofstream(path, std::ios::binary) << "corrupt";
  (void)write_stub_model(folder, 320);
  std::ifstream in(path, std::ios::binary);
  const std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  VERIFY(content == stub_model_onnx(320));
  VERIFY(std::distance(std::filesystem::directory_iterator(folder),
                       std::filesystem::directory_iterator{}) == 1);
}

void check_scene() {
  for (const auto& [w, h] : {std::pair{64, 48}, std::pair{1280, 720}, std::pair{333, 1000}}) {
    for (std::uint32_t frame = 0; frame < 60; ++frame) {
      const SyntheticScene scene = make_scene(w, h, frame);
      for (const SceneRect& r : scene.rects) {
        VERIFY(r.x0 >= 0 && r.y0 >= 0 && r.x1 <= w && r.y1 <= h && r.x1 > r.x0 && r.y1 > r.y0);
      }
      // The detected rectangles are apart, so neither covers the other.
      VERIFY(scene.rects[0].x1 <= scene.rects[1].x0);
      const auto truth = scene_truth(scene);
      VERIFY(truth.size() == 2 && truth[0].class_id == 0 && truth[1].class_id == 1);
      VERIFY(truth[0].x1 == static_cast<float>(scene.rects[0].x0) &&
             truth[1].y2 == static_cast<float>(scene.rects[1].y1));
    }
  }
  VERIFY(make_scene(640, 480, 3).rects[0].x0 != make_scene(640, 480, 4).rects[0].x0);
  VERIFY(testing::throws<std::invalid_argument>([] { (void)make_scene(63, 48, 0); }));
  VERIFY(testing::throws<std::invalid_argument>([] { (void)make_scene(64, 4097, 0); }));
  // The colors land in the right channels and the rectangles in the right
  // places; alpha is opaque.
  const SyntheticScene scene = make_scene(200, 100, 0);
  std::vector<std::uint8_t> bgra(200 * 100 * 4);
  render_scene(scene, PixelFormat::kBgra8, bgra.data(), 200 * 4);
  const SceneRect& red = scene.rects[0];
  const std::uint8_t* inside = bgra.data() + (red.y0 * 200 + red.x0) * 4;
  VERIFY(inside[0] == red.rgb[2] && inside[1] == red.rgb[1] && inside[2] == red.rgb[0] &&
         inside[3] == 255);
  const std::uint8_t* corner = bgra.data();
  VERIFY(corner[0] == 40 && corner[1] == 40 && corner[2] == 40);
}

}  // namespace
}  // namespace yolo26_detection

int main(int argc, char** argv) {
  namespace yd = yolo26_detection;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " DIR\n";
    return 2;
  }
  yd::g_directory = argv[1];
  std::filesystem::create_directories(yd::g_directory);
  return yd::testing::run({
      {"iou and edge error", yd::check_iou},
      {"truth matching", yd::check_match_truth},
      {"expectation matching", yd::check_count_matched},
      {"expectation file", yd::check_expectation_file},
      {"labels file", yd::check_labels},
      {"image headers", yd::check_image_headers},
      {"image listing", yd::check_listing},
      {"PPM round trip", yd::check_ppm_round_trip},
      {"encodings to RGB", yd::check_to_rgb},
      {"box drawing", yd::check_draw_boxes},
      {"stub model file", yd::check_stub_model_file},
      {"synthetic scene", yd::check_scene},
  });
}
