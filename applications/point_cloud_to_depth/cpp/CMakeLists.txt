# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Select the current GPU unless the caller supplied an architecture list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(point_cloud_to_depth LANGUAGES CXX CUDA)

# The wheel's CTest driver does not forward --local-sdk-root into CMake.
# Also honor the standard SDK environment variable for direct CMake consumers.
if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  get_filename_component(_pcd_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_pcd_sdk_root}")
endif()
set(_pcd_holoscan_components core sensor_io)
if(BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _pcd_holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_pcd_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# The projection kernels, the synthetic scene, its ground truth and file I/O.
# No Holoscan.
add_library(point_cloud_to_depth_core STATIC
  projection/cloud_io.cpp
  projection/depth_truth.cpp
  projection/project.cu
  projection/synthetic_scene.cu)
target_compile_features(point_cloud_to_depth_core PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(point_cloud_to_depth_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(point_cloud_to_depth_core PUBLIC CUDA::cudart)
# Without fused multiply-adds the kernels compute bit for bit what the host
# computes from the same expressions: the synthetic clouds match their host
# copies, and the projection matches the tests' CPU reference pixel for pixel.
# The kernels are bound by memory and atomics, not arithmetic.
set_source_files_properties(projection/project.cu projection/synthetic_scene.cu
  PROPERTIES COMPILE_OPTIONS "-fmad=false")
# The host side of that: every target that computes with these functions links
# this library, so none of them may contract either, whatever the compiler's
# default.
target_compile_options(point_cloud_to_depth_core PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
set_target_properties(point_cloud_to_depth_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF)

# The Holoscan operators.
add_library(point_cloud_to_depth_operators STATIC
  operators/depth_image_report_op.cpp
  operators/point_cloud_sources.cpp
  operators/point_cloud_to_depth_op.cpp)
target_include_directories(point_cloud_to_depth_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(point_cloud_to_depth_operators PUBLIC
  point_cloud_to_depth_core holoscan::core holoscan::sensor_io)
set_target_properties(point_cloud_to_depth_operators PROPERTIES CXX_EXTENSIONS OFF)

add_executable(point_cloud_to_depth main.cpp)
target_link_libraries(point_cloud_to_depth PRIVATE point_cloud_to_depth_operators)
# Place the executable in the application's build directory, where the CLI
# starts the run commands from metadata.json.
set_target_properties(point_cloud_to_depth PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_EXTENSIONS OFF)
install(TARGETS point_cloud_to_depth RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  # Keep tests in the C++ dependency scope and preserve their build directory.
  add_subdirectory(../tests "${CMAKE_CURRENT_BINARY_DIR}/../tests")
endif()
