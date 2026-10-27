# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(point_cloud_to_depth_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# The PLY reader needs no SDK or GPU. Keep its test outside the C++ dependency
# scope so hosted CI runs it.
if(BUILD_TESTING)
  enable_language(CXX)
  add_executable(pcd_cloud_io_test tests/cloud_io_test.cpp cpp/projection/cloud_io.cpp)
  target_include_directories(pcd_cloud_io_test PRIVATE cpp tests)
  target_compile_features(pcd_cloud_io_test PRIVATE cxx_std_20)
  add_test(NAME pcd_cloud_io
    COMMAND pcd_cloud_io_test "${CMAKE_CURRENT_BINARY_DIR}/cloud_io_files")
  set_tests_properties(pcd_cloud_io PROPERTIES LABELS "point_cloud_to_depth" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md LICENSE DESTINATION share/point_cloud_to_depth)
