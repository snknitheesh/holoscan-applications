# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Select the current GPU unless the caller supplied an architecture list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile")
endif()
project(depth_to_point_cloud LANGUAGES CXX CUDA)

# The wheel's CTest driver does not forward --local-sdk-root into CMake.
# Also honor the standard SDK environment variable for direct CMake consumers.
if(DEFINED ENV{HOLOSCAN_SDK_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOLOSCAN_SDK_ROOT}")
elseif(DEFINED ENV{HOLOSCAN_LIB_PATH})
  get_filename_component(_depth_sdk_root "$ENV{HOLOSCAN_LIB_PATH}" DIRECTORY)
  list(PREPEND CMAKE_PREFIX_PATH "${_depth_sdk_root}")
endif()
set(_depth_holoscan_components core sensor_io)
if(BUILD_TESTING)
  # The operator tests use the SDK's operator test harness.
  list(APPEND _depth_holoscan_components testing)
endif()
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ${_depth_holoscan_components})
find_package(CUDAToolkit REQUIRED)

# The deprojection kernel, the synthetic camera and file I/O. No Holoscan.
add_library(depth_to_point_cloud_core STATIC
  deproject/deproject.cu
  deproject/image_io.cpp
  deproject/synthetic_depth.cu)
target_compile_features(depth_to_point_cloud_core PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(depth_to_point_cloud_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(depth_to_point_cloud_core PUBLIC CUDA::cudart)
# Without fused multiply-adds the synthetic camera renders the same depths the
# host computes, so the ground-truth check can be tight.
set_source_files_properties(deproject/synthetic_depth.cu PROPERTIES COMPILE_OPTIONS "-fmad=false")
# The host side of that check: every target that computes the truth with these
# functions links this library, so none of them may contract either, whatever
# the compiler's default.
target_compile_options(depth_to_point_cloud_core PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
set_target_properties(depth_to_point_cloud_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF)

# The Holoscan operators.
add_library(depth_to_point_cloud_operators STATIC
  operators/depth_sources.cpp
  operators/depth_to_point_cloud_op.cpp
  operators/point_cloud_report_op.cpp)
target_include_directories(depth_to_point_cloud_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(depth_to_point_cloud_operators PUBLIC
  depth_to_point_cloud_core holoscan::core holoscan::sensor_io)
set_target_properties(depth_to_point_cloud_operators PROPERTIES CXX_EXTENSIONS OFF)

add_executable(depth_to_point_cloud main.cpp)
target_link_libraries(depth_to_point_cloud PRIVATE depth_to_point_cloud_operators)
# Place the executable in the application's build directory, where the CLI
# starts the run commands from metadata.json.
set_target_properties(depth_to_point_cloud PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_EXTENSIONS OFF)
install(TARGETS depth_to_point_cloud RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  # Keep tests in the C++ dependency scope and preserve their build directory.
  add_subdirectory(../tests "${CMAKE_CURRENT_BINARY_DIR}/../tests")
endif()
