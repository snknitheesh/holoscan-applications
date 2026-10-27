# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(depth_to_point_cloud_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# The PGM and PLY file code needs no SDK or GPU. Keep its test outside the C++
# dependency scope so hosted CI runs it.
if(BUILD_TESTING)
  enable_language(CXX)
  add_executable(depth_image_io_test tests/image_io_test.cpp cpp/deproject/image_io.cpp)
  target_include_directories(depth_image_io_test PRIVATE cpp tests)
  target_compile_features(depth_image_io_test PRIVATE cxx_std_20)
  add_test(NAME depth_image_io
    COMMAND depth_image_io_test "${CMAKE_CURRENT_BINARY_DIR}/image_io_files")
  set_tests_properties(depth_image_io PROPERTIES LABELS "depth_to_point_cloud" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md LICENSE DESTINATION share/depth_to_point_cloud)
