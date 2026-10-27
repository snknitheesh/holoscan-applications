# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Select the current GPU unless the caller supplied an architecture list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(yolo26_detection LANGUAGES CXX CUDA)

option(YOLO26_DETECTION_ENABLE_VISUALIZATION "Build the Holoscan Visualization display" ON)

# The wheel's CTest driver does not forward --local-sdk-root into CMake.
# Also honor the standard SDK environment variable for direct CMake consumers.
if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  get_filename_component(_yolo26_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_yolo26_sdk_root}")
endif()
# HoloInfer, the SDK's TensorRT inference library, comes with its inference
# component.
set(_yolo26_holoscan_components core sensor_io inference)
if(YOLO26_DETECTION_ENABLE_VISUALIZATION)
  list(APPEND _yolo26_holoscan_components visualization)
endif()
if(BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _yolo26_holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_yolo26_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# The letterbox and decode kernels, the stub model, the synthetic scene, the
# image files and the matching of detections. No Holoscan.
add_library(yolo26_detection_core STATIC
  detection/decode.cu
  detection/image_files.cpp
  detection/letterbox.cu
  detection/matching.cpp
  detection/stub_model.cpp
  detection/synthetic_scene.cpp)
target_compile_features(yolo26_detection_core PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(yolo26_detection_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yolo26_detection_core PUBLIC CUDA::cudart PRIVATE CUDA::nvjpeg)
# Without fused multiply-adds the kernels compute bit for bit what the host
# computes from the same expressions, so the tests compare them exactly.
set_source_files_properties(detection/decode.cu detection/letterbox.cu
  PROPERTIES COMPILE_OPTIONS "-fmad=false")
# The host side of that: every target that computes with these functions links
# this library, so none of them may contract either.
target_compile_options(yolo26_detection_core PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
set_target_properties(yolo26_detection_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF)

# The Holoscan operators.
add_library(yolo26_detection_operators STATIC
  operators/detection_report_op.cpp
  operators/image_sources.cpp
  operators/yolo26_detection_op.cpp)
target_include_directories(yolo26_detection_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yolo26_detection_operators PUBLIC
  yolo26_detection_core holoscan::core holoscan::sensor_io PRIVATE holoscan::infer)
set_target_properties(yolo26_detection_operators PROPERTIES CXX_EXTENSIONS OFF)
target_compile_definitions(yolo26_detection_operators PUBLIC
  YOLO26_DETECTION_ENABLE_VISUALIZATION=$<BOOL:${YOLO26_DETECTION_ENABLE_VISUALIZATION}>)
if(YOLO26_DETECTION_ENABLE_VISUALIZATION)
  target_sources(yolo26_detection_operators PRIVATE operators/detection_overlay_op.cpp)
  target_link_libraries(yolo26_detection_operators PUBLIC holoscan::visualization::op_holoviz)
endif()

add_executable(yolo26_detection main.cpp)
target_link_libraries(yolo26_detection PRIVATE yolo26_detection_operators)
# Place the executable in the application's build directory, where the CLI
# starts the run commands from metadata.json.
set_target_properties(yolo26_detection PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_EXTENSIONS OFF)
install(TARGETS yolo26_detection RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  # Keep tests in the C++ dependency scope and preserve their build directory.
  add_subdirectory(../tests "${CMAKE_CURRENT_BINARY_DIR}/../tests")
endif()
