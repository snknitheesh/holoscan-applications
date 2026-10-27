// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The model is written as ONNX protobuf directly: the few message types it
// needs are encoded here, so the application does not depend on ONNX or
// protobuf libraries.

#include "stub_model.hpp"

#include <unistd.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "detection/letterbox.hpp"

namespace yolo26_detection {
namespace {

static_assert(std::endian::native == std::endian::little,
              "ONNX raw tensor data is little-endian; this writer copies host bytes");

// ONNX enumerations.
constexpr int kFloat = 1;     // TensorProto.DataType.FLOAT
constexpr int kInt64 = 7;     // TensorProto.DataType.INT64
constexpr int kAttrInt = 2;   // AttributeProto.AttributeType.INT
constexpr int kAttrInts = 7;  // AttributeProto.AttributeType.INTS

// One protobuf message under construction.
class Message {
 public:
  Message& varint(int field, std::uint64_t value) {
    tag(field, 0);
    put_varint(value);
    return *this;
  }
  Message& integer(int field, std::int64_t value) {
    return varint(field, static_cast<std::uint64_t>(value));
  }
  Message& bytes(int field, std::string_view value) {
    tag(field, 2);
    put_varint(value.size());
    out_.append(value);
    return *this;
  }
  Message& message(int field, const Message& value) { return bytes(field, value.out_); }
  [[nodiscard]] const std::string& str() const { return out_; }

 private:
  void tag(int field, int wire_type) {
    put_varint((static_cast<std::uint64_t>(field) << 3U) | static_cast<std::uint64_t>(wire_type));
  }
  void put_varint(std::uint64_t value) {
    while (value >= 0x80U) {
      out_.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
      value >>= 7U;
    }
    out_.push_back(static_cast<char>(value));
  }
  std::string out_;
};

template <typename T>
std::string raw(const std::vector<T>& values) {
  std::string bytes(values.size() * sizeof(T), '\0');
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

// TensorProto: dims = 1, data_type = 2, name = 8, raw_data = 9.
Message tensor(std::string_view name, const std::vector<std::int64_t>& dims, int type,
               std::string data) {
  Message t;
  for (const std::int64_t dim : dims) {
    t.integer(1, dim);
  }
  t.integer(2, type);
  t.bytes(8, name);
  t.bytes(9, data);
  return t;
}

Message int64_tensor(std::string_view name, std::vector<std::int64_t> values) {
  return tensor(name, {static_cast<std::int64_t>(values.size())}, kInt64, raw(values));
}

Message float_scalar(std::string_view name, float value) {
  return tensor(name, {}, kFloat, raw(std::vector<float>{value}));
}

// AttributeProto: name = 1, i = 3, ints = 8, type = 20.
Message int_attribute(std::string_view name, std::int64_t value) {
  Message a;
  a.bytes(1, name).integer(3, value).integer(20, kAttrInt);
  return a;
}

Message ints_attribute(std::string_view name, std::initializer_list<std::int64_t> values) {
  Message a;
  a.bytes(1, name);
  for (const std::int64_t value : values) {
    a.integer(8, value);
  }
  a.integer(20, kAttrInts);
  return a;
}

// NodeProto: input = 1, output = 2, name = 3, op_type = 4, attribute = 5.
Message node(std::string_view op, std::initializer_list<std::string_view> inputs,
             std::string_view output, std::initializer_list<Message> attributes = {}) {
  Message n;
  for (const std::string_view input : inputs) {
    n.bytes(1, input);
  }
  n.bytes(2, output);
  n.bytes(3, output);
  n.bytes(4, op);
  for (const Message& attribute : attributes) {
    n.message(5, attribute);
  }
  return n;
}

// ValueInfoProto for a float tensor: name = 1, type = 2 { tensor_type = 1 {
// elem_type = 1, shape = 2 { dim = 1 { dim_value = 1 } } } }.
Message float_value(std::string_view name, std::initializer_list<std::int64_t> dims) {
  Message shape;
  for (const std::int64_t dim : dims) {
    Message d;
    d.integer(1, dim);
    shape.message(1, d);
  }
  Message tensor_type;
  tensor_type.integer(1, kFloat).message(2, shape);
  Message type;
  type.message(1, tensor_type);
  Message value;
  value.bytes(1, name).message(2, type);
  return value;
}

// The fixed low-score rows that follow the two detections.
std::vector<float> filler_rows(int size) {
  std::vector<float> rows;
  const int count = kStubRows - kStubClasses;
  rows.reserve(static_cast<std::size_t>(count) * 6U);
  for (int i = 0; i < count; ++i) {
    const float x = static_cast<float>((i * 37) % (size / 2));
    const float y = static_cast<float>((i * 53) % (size / 2));
    rows.insert(
        rows.end(),
        {x, y, x + static_cast<float>(size) / 4.0F, y + static_cast<float>(size) / 4.0F,
         kStubMaxFillerScore * static_cast<float>(count - i) / static_cast<float>(count + 1),
         static_cast<float>(i % kStubClasses)});
  }
  return rows;
}

}  // namespace

std::string stub_model_onnx(int size) {
  if (size < kMinInputSize || size > kMaxInputSize) {
    throw std::invalid_argument("stub model input size must be " + std::to_string(kMinInputSize) +
                                ".." + std::to_string(kMaxInputSize));
  }
  Message graph;
  // GraphProto: node = 1, name = 2, initializer = 5, input = 11, output = 12.
  graph.bytes(2, "yolo26_stub");
  const std::vector<Message> initializers{
      int64_tensor("channel_axis", {1}),
      int64_tensor("red_start", {0}),
      int64_tensor("red_end", {1}),
      int64_tensor("green_start", {1}),
      int64_tensor("green_end", {2}),
      int64_tensor("blue_start", {2}),
      int64_tensor("blue_end", {3}),
      int64_tensor("reverse_start", {-1}),
      // Past the first element, so the reversed profile includes it.
      int64_tensor("reverse_end", {-static_cast<std::int64_t>(size) - 1}),
      int64_tensor("profile_axis", {2}),
      int64_tensor("reverse_step", {-1}),
      float_scalar("half", 0.5F),
      float_scalar("size", static_cast<float>(size)),
      float_scalar("class_score_0", kStubScores[0]),
      float_scalar("class_score_1", kStubScores[1]),
      tensor("class_0", {1, 1, 1}, kFloat, raw(std::vector<float>{0.0F})),
      tensor("class_1", {1, 1, 1}, kFloat, raw(std::vector<float>{1.0F})),
      tensor("filler", {1, kStubRows - kStubClasses, 6}, kFloat, raw(filler_rows(size))),
  };
  for (const Message& initializer : initializers) {
    graph.message(5, initializer);
  }
  std::vector<Message> nodes{
      node("Slice", {"images", "red_start", "red_end", "channel_axis"}, "red"),
      node("Slice", {"images", "green_start", "green_end", "channel_axis"}, "green"),
      node("Slice", {"images", "blue_start", "blue_end", "channel_axis"}, "blue"),
  };
  // One detection per class: the bounding box of the pixels whose channel
  // exceeds green by more than one half.
  for (const std::string c : {"0", "1"}) {
    const std::string channel = c == "0" ? "red" : "blue";
    const auto name = [&c](std::string_view stem) { return std::string(stem) + "_" + c; };
    nodes.push_back(node("Sub", {channel, "green"}, name("excess")));
    nodes.push_back(node("Greater", {name("excess"), "half"}, name("above")));
    nodes.push_back(node("Cast", {name("above")}, name("mask"), {int_attribute("to", kFloat)}));
    // Columns ([1, 1, S], over the rows) and rows (over the columns) that hold
    // any qualifying pixel.
    nodes.push_back(node("ReduceMax", {name("mask")}, name("columns"),
                         {ints_attribute("axes", {2}), int_attribute("keepdims", 0)}));
    nodes.push_back(node("ReduceMax", {name("mask")}, name("rows"),
                         {ints_attribute("axes", {3}), int_attribute("keepdims", 0)}));
    for (const std::string axis : {"columns", "rows"}) {
      const std::string profile = name(axis);
      // The first set entry from the start gives the low edge; from the end,
      // the exclusive high edge.
      nodes.push_back(node("ArgMax", {profile}, profile + "_first",
                           {int_attribute("axis", 2), int_attribute("keepdims", 1)}));
      nodes.push_back(
          node("Slice", {profile, "reverse_start", "reverse_end", "profile_axis", "reverse_step"},
               profile + "_reversed"));
      nodes.push_back(node("ArgMax", {profile + "_reversed"}, profile + "_last",
                           {int_attribute("axis", 2), int_attribute("keepdims", 1)}));
      nodes.push_back(
          node("Cast", {profile + "_first"}, profile + "_low", {int_attribute("to", kFloat)}));
      nodes.push_back(
          node("Cast", {profile + "_last"}, profile + "_from_end", {int_attribute("to", kFloat)}));
      nodes.push_back(node("Sub", {"size", profile + "_from_end"}, profile + "_high"));
    }
    nodes.push_back(node("ReduceMax", {name("columns")}, name("present"),
                         {ints_attribute("axes", {2}), int_attribute("keepdims", 1)}));
    nodes.push_back(node("Mul", {name("present"), "class_score_" + c}, name("score")));
    nodes.push_back(
        node("Concat",
             {name("columns") + "_low", name("rows") + "_low", name("columns") + "_high",
              name("rows") + "_high", name("score"), "class_" + c},
             name("detection"), {int_attribute("axis", 2)}));
  }
  nodes.push_back(node("Concat", {"detection_0", "detection_1", "filler"}, "output0",
                       {int_attribute("axis", 1)}));
  for (const Message& n : nodes) {
    graph.message(1, n);
  }
  graph.message(11, float_value("images", {1, 3, size, size}));
  graph.message(12, float_value("output0", {1, kStubRows, 6}));

  // ModelProto: ir_version = 1, producer_name = 2, graph = 7, opset_import =
  // 8 { domain = 1, version = 2 }.
  Message opset;
  opset.bytes(1, "").integer(2, 13);
  Message model;
  model.integer(1, 7).bytes(2, "yolo26_detection stub").message(7, graph).message(8, opset);
  return model.str();
}

std::filesystem::path write_stub_model(const std::filesystem::path& directory, int size) {
  const std::string bytes = stub_model_onnx(size);
  const std::filesystem::path path = directory / ("yolo26_stub_" + std::to_string(size) + ".onnx");
  {
    std::ifstream existing(path, std::ios::binary);
    if (existing) {
      const std::string current{std::istreambuf_iterator<char>(existing),
                                std::istreambuf_iterator<char>()};
      if (current == bytes) {
        return path;
      }
    }
  }
  static std::atomic<unsigned> sequence{0};
  const std::filesystem::path temporary =
      directory / (path.filename().string() + ".tmp." + std::to_string(::getpid()) + "." +
                   std::to_string(sequence.fetch_add(1)));
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out.flush()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      throw std::runtime_error("cannot write the stub model to " + temporary.string());
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    throw std::runtime_error("cannot write the stub model to " + path.string());
  }
  return path;
}

}  // namespace yolo26_detection
