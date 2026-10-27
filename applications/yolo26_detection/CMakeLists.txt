# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(yolo26_detection_apps LANGUAGES NONE)
include(CTest)

# Also provide the CLI language switch for standalone application builds.
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)

# The model preparation tests use fixtures and run without the SDK, a GPU,
# Ultralytics or the network, and the webcam's YUYV conversion is header-only.
# Keep their registration outside the C++ dependency scope for hosted CI.
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME yolo26_prepare_model
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_prepare_model.py -v)
  enable_language(CXX)
  add_executable(yolo26_yuyv_test tests/yuyv_test.cpp)
  target_include_directories(yolo26_yuyv_test PRIVATE cpp tests)
  target_compile_features(yolo26_yuyv_test PRIVATE cxx_std_20)
  add_test(NAME yolo26_yuyv COMMAND yolo26_yuyv_test)
  set_tests_properties(yolo26_prepare_model yolo26_yuyv PROPERTIES
    LABELS "yolo26_detection" TIMEOUT 60)
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()

install(FILES README.md prepare_model.py LICENSE DESTINATION share/yolo26_detection)
