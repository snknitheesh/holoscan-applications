# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# These tests need the SDK, its CUDA toolchain and a GPU. The application has
# no SDK-free tests to register at the top level.
add_executable(depth_deproject_test deproject_test.cu)
target_link_libraries(depth_deproject_test PRIVATE depth_to_point_cloud_core)

add_executable(depth_image_io_test image_io_test.cpp)
target_link_libraries(depth_image_io_test PRIVATE depth_to_point_cloud_core)

add_executable(depth_operator_test operator_test.cpp)
target_link_libraries(depth_operator_test PRIVATE
  depth_to_point_cloud_operators holoscan::testing)

add_executable(depth_write_test_images write_test_images.cpp)
target_link_libraries(depth_write_test_images PRIVATE depth_to_point_cloud_core)

add_library(depth_spawn_thread_preload SHARED spawn_thread_preload.cpp)

add_test(NAME depth_deproject COMMAND depth_deproject_test)
add_test(NAME depth_image_io
  COMMAND depth_image_io_test "${CMAKE_CURRENT_BINARY_DIR}/image_io_files")
add_test(NAME depth_operators COMMAND depth_operator_test)

# Runs the application through run_and_match.cmake, which checks its exit
# status as well as its output: depth_app_test(<name> <exit status> <output
# regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(depth_app_test name status output)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}"
      -P "${_run_and_match}" -- $<TARGET_FILE:depth_to_point_cloud> ${ARGN})
endfunction()

# The application on the synthetic camera, every point and color checked
# against ground truth: float32 and uint16 depth, from GPU and host memory,
# with configured and per-frame intrinsics.
set(_pass "wrong_points=0 wrong_colors=0 .*verdict=PASS")
depth_app_test(depth_app_meters 0 "${_pass}" --count 20 --period-ms 5)
depth_app_test(depth_app_millimeters 0 "${_pass}"
  --count 20 --period-ms 5 --depth-format millimeters)
depth_app_test(depth_app_host_input 0 "${_pass}" --count 20 --period-ms 5 --host-input true)
depth_app_test(depth_app_intrinsics_input 0 "${_pass}"
  --count 20 --period-ms 5 --intrinsics-input true)
depth_app_test(depth_app_without_color 0 "wrong_points=0 .*verdict=PASS"
  --count 20 --period-ms 5 --color false)
# A narrower valid range reaches the operator: the truth applies the same one,
# and the reported depths stay inside it.
depth_app_test(depth_app_depth_range 0
  "depth_m=\\[1\\.200, 1\\.800\\] truth: wrong_points=0 wrong_colors=0 .*verdict=PASS"
  --count 20 --period-ms 5 --depth-min 1.2 --depth-max 1.8)

# PLY output: one file per frame.
set(_ply "${CMAKE_CURRENT_BINARY_DIR}/ply")
add_test(NAME depth_app_ply
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=ply_files=3 .*verdict=PASS"
    "-DMAKE_DIRECTORY=${_ply}" -P "${_run_and_match}" --
    $<TARGET_FILE:depth_to_point_cloud> --count 3 --period-ms 5 --ply-dir "${_ply}")

# A folder of 16-bit PGM depth images written by the fixture, read back with
# explicit intrinsics. The valid points and depth range are those of the
# synthetic frames 0..4 at 320x240 in millimeters, valid in 0.1..10 m.
set(_images "${CMAKE_CURRENT_BINARY_DIR}/depth_images")
add_test(NAME depth_images_setup
  COMMAND $<TARGET_FILE:depth_write_test_images> "${_images}" 5 320 240)
set_tests_properties(depth_images_setup PROPERTIES FIXTURES_SETUP depth_images)
set(_image_intrinsics --fx 256 --fy 256 --cx 159.5 --cy 119.5)
depth_app_test(depth_app_images 0
  "frames=5/5 points=384000 valid=369920 depth_m=\\[1\\.040, 1\\.956\\].*verdict=PASS"
  --source images --images "${_images}" ${_image_intrinsics} --period-ms 5)
# --loop starts over after the last image: 12 frames from 5 images.
depth_app_test(depth_app_images_loop 0 "frames=12/12 points=921600 .*verdict=PASS"
  --source images --images "${_images}" ${_image_intrinsics} --period-ms 5
  --loop true --count 12)
depth_app_test(depth_app_rejects_count_beyond_images 2 "^error: .*--loop true"
  --source images --images "${_images}" ${_image_intrinsics} --count 6)

# Invalid input fails fast with a usage error (exit status 2) instead of
# starting a run.
depth_app_test(depth_app_rejects_unknown_option 2 "^error: unknown option" --colour true)
depth_app_test(depth_app_rejects_zero_count 2 "^error: " --count 0)
depth_app_test(depth_app_rejects_images_without_intrinsics 2 "^error: .*--fx"
  --source images --images "${_images}")
depth_app_test(depth_app_rejects_partial_intrinsics 2 "^error: .*go together"
  --source images --images "${_images}" --fx 256)
depth_app_test(depth_app_rejects_size_with_images 2 "^error: .*--width and --height"
  --source images --images "${_images}" --fx 256 --fy 256 --cx 159.5 --cy 119.5 --width 320)
depth_app_test(depth_app_rejects_zero_focal_length 2 "^error: intrinsics need"
  --source images --images "${_images}" --fx 0 --fy 256 --cx 159.5 --cy 119.5)
depth_app_test(depth_app_rejects_zero_depth_scale 2 "^error: depth_scale"
  --source images --images "${_images}" ${_image_intrinsics} --depth-scale 0)
set_tests_properties(depth_app_images depth_app_images_loop depth_app_rejects_count_beyond_images
  depth_app_rejects_images_without_intrinsics depth_app_rejects_partial_intrinsics
  depth_app_rejects_size_with_images
  depth_app_rejects_zero_focal_length depth_app_rejects_zero_depth_scale
  PROPERTIES FIXTURES_REQUIRED depth_images)
depth_app_test(depth_app_rejects_zero_width 2 "^error: synthetic depth images" --width 0)
depth_app_test(depth_app_rejects_inverted_range 2 "^error: depth range"
  --depth-min 5 --depth-max 1)
depth_app_test(depth_app_rejects_missing_images 2 "^error: .*cannot list"
  --source images --images "${CMAKE_CURRENT_BINARY_DIR}/no_such_folder" ${_image_intrinsics})
set(_no_images "${CMAKE_CURRENT_BINARY_DIR}/no_images")
add_test(NAME depth_app_rejects_empty_images
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=2 "-DEXPECT_OUTPUT=^error: .*no \\.pgm images"
    "-DMAKE_DIRECTORY=${_no_images}" -P "${_run_and_match}" --
    $<TARGET_FILE:depth_to_point_cloud> --source images --images "${_no_images}"
    ${_image_intrinsics})

# A failed run reports its cause and exits 1.
set(_blocked "${CMAKE_CURRENT_BINARY_DIR}/blocked_ply")
add_test(NAME depth_app_failed_run
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=1 "-DEXPECT_OUTPUT=frame_000000.ply.*verdict=FAIL"
    "-DMAKE_DIRECTORY=${_blocked}/frame_000000.ply" -P "${_run_and_match}" --
    $<TARGET_FILE:depth_to_point_cloud> --count 5 --period-ms 5 --ply-dir "${_blocked}")

# A run that outlives its deadline is stopped and fails: 100 frames of
# 4096x4096, each checked point by point, take far longer than the one second
# allowed beyond the 0.2 s the frames need.
depth_app_test(depth_app_timed_out 1 "timed out.*frames=[0-9]+/100 .*verdict=FAIL"
  --width 4096 --height 4096 --count 100 --period-ms 1 --timeout-seconds 1)

# A stop request ends the run early with its own verdict, and the warning it
# leaves is shown. The signal goes out once the first cloud is written, so the
# run has started. The test sends SIGTERM, as a shell starts background jobs
# with SIGINT ignored, and its script has no semicolons, where CMake would
# split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped_ply")
add_test(NAME depth_app_stopped
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/100000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     \"$1\" --count 100000 --period-ms 10 --width 64 --height 48 --ply-dir \"$2\" &
     for i in $(seq 600)
     do
       [ -e \"$2/frame_000000.ply\" ] && break
       sleep 0.05
     done
     kill -TERM $! && wait $!"
    sh $<TARGET_FILE:depth_to_point_cloud> "${_stopped}")

# Another thread running before main(), as under a profiler or a sanitizer:
# the run goes ahead without managed signal handling, and says so.
add_test(NAME depth_app_without_signal_domain
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS"
    -P "${_run_and_match}" -- ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:depth_spawn_thread_preload>
    $<TARGET_FILE:depth_to_point_cloud> --count 10 --period-ms 5)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "depth_to_point_cloud;gpu" TIMEOUT 300)
