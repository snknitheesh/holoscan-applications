# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# These tests build with the SDK and its CUDA toolchain. All but the PLY reader
# tests and the fixtures that write PLY files also need a GPU and its driver,
# which the application executable loads at start, and carry the `gpu` label.
# The application has no SDK-free tests to register at the top level.
add_executable(pcd_project_test project_test.cu)
target_link_libraries(pcd_project_test PRIVATE point_cloud_to_depth_core)

add_executable(pcd_cloud_io_test cloud_io_test.cpp)
target_link_libraries(pcd_cloud_io_test PRIVATE point_cloud_to_depth_core)

add_executable(pcd_operator_test operator_test.cpp)
target_link_libraries(pcd_operator_test PRIVATE
  point_cloud_to_depth_operators holoscan::testing)

add_executable(pcd_write_test_clouds write_test_clouds.cpp)
target_link_libraries(pcd_write_test_clouds PRIVATE point_cloud_to_depth_core)

add_library(pcd_spawn_thread_preload SHARED spawn_thread_preload.cpp)

add_test(NAME pcd_project COMMAND pcd_project_test)
add_test(NAME pcd_cloud_io
  COMMAND pcd_cloud_io_test "${CMAKE_CURRENT_BINARY_DIR}/cloud_io_files")
add_test(NAME pcd_operators
  COMMAND pcd_operator_test "${CMAKE_CURRENT_BINARY_DIR}/operator_files")

# Runs the application through run_and_match.cmake, which checks its exit
# status as well as its output: pcd_app_test(<name> <exit status> <output
# regex> <arguments>...).
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(pcd_app_test name status output)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}"
      -P "${_run_and_match}" -- $<TARGET_FILE:point_cloud_to_depth> ${ARGN})
endfunction()

# The application on the synthetic cloud, every pixel and color checked
# against ground truth: float32 and uint16 depth, from GPU and host memory,
# with configured and per-frame intrinsics, and every splat size.
set(_pass "wrong_depths=0 wrong_colors=0 .*verdict=PASS")
pcd_app_test(pcd_app_float32 0 "${_pass}" --count 20 --period-ms 5)
pcd_app_test(pcd_app_uint16 0 "${_pass}" --count 20 --period-ms 5 --depth-format uint16)
pcd_app_test(pcd_app_host_input 0 "${_pass}" --count 20 --period-ms 5 --host-input true)
pcd_app_test(pcd_app_intrinsics_input 0 "${_pass}"
  --count 20 --period-ms 5 --intrinsics-input true)
pcd_app_test(pcd_app_without_color 0 "wrong_depths=0 .*verdict=PASS"
  --count 20 --period-ms 5 --color false)
pcd_app_test(pcd_app_point_size_3 0 "${_pass}" --count 20 --period-ms 5 --point-size 3)
pcd_app_test(pcd_app_point_size_5_uint16 0 "${_pass}"
  --count 20 --period-ms 5 --point-size 5 --depth-format uint16)
# A narrower valid range reaches the operator: the truth applies the same one,
# and the reported depths stay inside it. Duplicates of nearer points, twice as
# far, now fall inside the range where their own points do not.
pcd_app_test(pcd_app_depth_range 0
  "depth_m=\\[1\\.200, 1\\.800\\] truth: ${_pass}"
  --count 20 --period-ms 5 --depth-min 1.2 --depth-max 1.8)
# A wider one lets the far patch at 50 m in, which uint16 millimeters hold.
pcd_app_test(pcd_app_far_patch_uint16 0 "depth_m=\\[1\\.040, 50\\.000\\] truth: ${_pass}"
  --count 20 --period-ms 5 --depth-max 100 --depth-format uint16)

# Image output: a depth and a color image per frame.
set(_images "${CMAKE_CURRENT_BINARY_DIR}/images")
add_test(NAME pcd_app_images
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=files_written=6 .*verdict=PASS"
    "-DMAKE_DIRECTORY=${_images}" -P "${_run_and_match}" --
    $<TARGET_FILE:point_cloud_to_depth> --count 3 --period-ms 5 --image-dir "${_images}")

# Folders of PLY files written by the fixture: synthetic frames 0..4 at
# 320x240, in the body frame, with and without colors. Read back with the
# camera's intrinsics and the body-to-camera extrinsics, every pixel with a
# valid scene depth is filled: 73984 per frame with 768 of the 76800 pixels in
# the far patch, and holes; 369920 over the five frames, with depths of
# 1.040..1.956 m, as an independent evaluation of the scene gives.
set(_clouds "${CMAKE_CURRENT_BINARY_DIR}/clouds")
set(_clouds_xyz "${CMAKE_CURRENT_BINARY_DIR}/clouds_xyz")
add_test(NAME pcd_clouds_setup
  COMMAND $<TARGET_FILE:pcd_write_test_clouds> "${_clouds}" 5 320 240)
add_test(NAME pcd_clouds_xyz_setup
  COMMAND $<TARGET_FILE:pcd_write_test_clouds> "${_clouds_xyz}" 5 320 240 xyz)
set_tests_properties(pcd_clouds_setup pcd_clouds_xyz_setup PROPERTIES FIXTURES_SETUP pcd_clouds)
# A folder mixing a cloud with colors and one without.
set(_clouds_mixed "${CMAKE_CURRENT_BINARY_DIR}/clouds_mixed")
file(MAKE_DIRECTORY "${_clouds_mixed}")
add_test(NAME pcd_clouds_mixed_setup
  COMMAND ${CMAKE_COMMAND} -E copy "${_clouds}/cloud_0000.ply" "${_clouds_xyz}/cloud_0001.ply"
    "${_clouds_mixed}")
set_tests_properties(pcd_clouds_mixed_setup PROPERTIES
  FIXTURES_REQUIRED pcd_clouds FIXTURES_SETUP pcd_clouds_mixed)
# A folder with a cloud and an empty one after it, which gives an empty image.
set(_clouds_empty "${CMAKE_CURRENT_BINARY_DIR}/clouds_with_empty")
file(MAKE_DIRECTORY "${_clouds_empty}")
file(WRITE "${_clouds_empty}/cloud_0001.ply"
  "ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\n"
  "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
  "property uchar blue\nend_header\n")
add_test(NAME pcd_clouds_empty_setup
  COMMAND ${CMAKE_COMMAND} -E copy "${_clouds}/cloud_0000.ply" "${_clouds_empty}")
set_tests_properties(pcd_clouds_empty_setup PROPERTIES
  FIXTURES_REQUIRED pcd_clouds FIXTURES_SETUP pcd_clouds_empty)
# A folder holding an ASCII PLY file.
set(_clouds_ascii "${CMAKE_CURRENT_BINARY_DIR}/clouds_ascii")
file(MAKE_DIRECTORY "${_clouds_ascii}")
file(WRITE "${_clouds_ascii}/ascii.ply"
  "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
  "property float z\nend_header\n1 2 3\n")

set(_camera --width 320 --height 240 --fx 256 --fy 248.32 --cx 159.5 --cy 119.5
  --extrinsics 0,-1,0,-0.04,0,0,-1,0.35,1,0,0,-0.12)
set(_files_pass "frames=5/5 pixels=384000 filled=369920 depth_m=\\[1\\.040, 1\\.956\\].*verdict=PASS")
pcd_app_test(pcd_app_files 0 "${_files_pass}"
  --source files --clouds "${_clouds}" ${_camera} --period-ms 5)
pcd_app_test(pcd_app_files_without_colors 0 "${_files_pass}"
  --source files --clouds "${_clouds_xyz}" ${_camera} --period-ms 5)
pcd_app_test(pcd_app_files_uint16 0 "${_files_pass}"
  --source files --clouds "${_clouds}" ${_camera} --period-ms 5 --depth-format uint16
  --depth-scale 0.001)
# --loop starts over after the last file: 12 clouds from 5 files.
pcd_app_test(pcd_app_files_loop 0 "frames=12/12 pixels=921600 .*verdict=PASS"
  --source files --clouds "${_clouds}" ${_camera} --period-ms 5 --loop true --count 12)
pcd_app_test(pcd_app_rejects_count_beyond_files 2 "^error: .*--loop true"
  --source files --clouds "${_clouds}" ${_camera} --count 6)
pcd_app_test(pcd_app_rejects_files_without_intrinsics 2 "^error: .*--fx"
  --source files --clouds "${_clouds}" --width 320 --height 240)
pcd_app_test(pcd_app_rejects_files_without_size 2 "^error: .*--width and --height"
  --source files --clouds "${_clouds}" --fx 256 --fy 248.32 --cx 159.5 --cy 119.5)
pcd_app_test(pcd_app_rejects_partial_intrinsics 2 "^error: .*go together"
  --source files --clouds "${_clouds}" --width 320 --height 240 --fx 256)
pcd_app_test(pcd_app_rejects_short_extrinsics 2 "^error: .*12 comma-separated numbers"
  --source files --clouds "${_clouds}" --width 320 --height 240 --fx 256 --fy 248.32
  --cx 159.5 --cy 119.5 --extrinsics 1,0,0,0,0,1,0,0,0,0,1)
pcd_app_test(pcd_app_rejects_non_finite_extrinsics 2 "^error: extrinsics must be finite"
  --source files --clouds "${_clouds}" --width 320 --height 240 --fx 256 --fy 248.32
  --cx 159.5 --cy 119.5 --extrinsics 1,0,0,0,0,1,0,0,0,0,1,inf)
pcd_app_test(pcd_app_rejects_zero_depth_scale 2 "^error: depth_scale"
  --source files --clouds "${_clouds}" ${_camera} --depth-format uint16 --depth-scale 0)
pcd_app_test(pcd_app_rejects_depth_scale_with_float32 2 "^error: --depth-scale applies to"
  --source files --clouds "${_clouds}" ${_camera} --depth-scale 0.001)
pcd_app_test(pcd_app_rejects_synthetic_options_with_files 2 "^error: .*apply to synthetic"
  --source files --clouds "${_clouds}" ${_camera} --host-input true)
pcd_app_test(pcd_app_rejects_color_without_file_colors 2 "^error: --color true needs"
  --source files --clouds "${_clouds_xyz}" ${_camera} --color true)
pcd_app_test(pcd_app_rejects_zero_count_with_files 2 "^error: --count must be positive"
  --source files --clouds "${_clouds}" ${_camera} --count 0)
# The extrinsics must be a camera pose: no scale, and no mirror.
pcd_app_test(pcd_app_rejects_scaled_extrinsics 2 "^error: --extrinsics must be a rigid transform"
  --source files --clouds "${_clouds}" --width 320 --height 240 --fx 256 --fy 248.32
  --cx 159.5 --cy 119.5 --extrinsics 0,-2,0,-0.04,0,0,-2,0.35,2,0,0,-0.12)
pcd_app_test(pcd_app_rejects_mirrored_extrinsics 2 "^error: --extrinsics must be a rigid transform"
  --source files --clouds "${_clouds}" --width 320 --height 240 --fx 256 --fy 248.32
  --cx 159.5 --cy 119.5 --extrinsics 0,1,0,-0.04,0,0,-1,0.35,1,0,0,-0.12)
# Without a truth to check them against, images in which no point landed fail
# the run: here the camera looks away from the points, 100 m behind them.
pcd_app_test(pcd_app_files_all_empty 1
  "warning: no point reached the image.*filled=0 .*verdict=FAIL"
  --source files --clouds "${_clouds}" --width 320 --height 240 --fx 256 --fy 248.32
  --cx 159.5 --cy 119.5 --extrinsics 0,-1,0,-0.04,0,0,-1,0.35,1,0,0,-100 --period-ms 5)
set_tests_properties(pcd_app_files pcd_app_files_without_colors pcd_app_files_uint16
  pcd_app_files_loop pcd_app_rejects_count_beyond_files
  pcd_app_rejects_files_without_intrinsics pcd_app_rejects_files_without_size
  pcd_app_rejects_partial_intrinsics pcd_app_rejects_short_extrinsics
  pcd_app_rejects_non_finite_extrinsics pcd_app_rejects_zero_depth_scale
  pcd_app_rejects_depth_scale_with_float32
  pcd_app_rejects_synthetic_options_with_files pcd_app_rejects_color_without_file_colors
  pcd_app_rejects_zero_count_with_files pcd_app_rejects_scaled_extrinsics
  pcd_app_rejects_mirrored_extrinsics pcd_app_files_all_empty
  PROPERTIES FIXTURES_REQUIRED pcd_clouds)
pcd_app_test(pcd_app_files_with_empty_cloud 0
  "frames=2/2 pixels=153600 filled=73984 .*verdict=PASS"
  --source files --clouds "${_clouds_empty}" ${_camera} --period-ms 5)
set_tests_properties(pcd_app_files_with_empty_cloud PROPERTIES FIXTURES_REQUIRED pcd_clouds_empty)
pcd_app_test(pcd_app_rejects_mixed_colors 2 "^error: either every PLY file has colors"
  --source files --clouds "${_clouds_mixed}" ${_camera})
set_tests_properties(pcd_app_rejects_mixed_colors PROPERTIES FIXTURES_REQUIRED pcd_clouds_mixed)
pcd_app_test(pcd_app_rejects_ascii_file 2 "^error: .*only binary_little_endian"
  --source files --clouds "${_clouds_ascii}" ${_camera})
pcd_app_test(pcd_app_rejects_missing_clouds 2 "^error: .*cannot list"
  --source files --clouds "${CMAKE_CURRENT_BINARY_DIR}/no_such_folder" ${_camera})
set(_no_clouds "${CMAKE_CURRENT_BINARY_DIR}/no_clouds")
add_test(NAME pcd_app_rejects_empty_clouds
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=2 "-DEXPECT_OUTPUT=^error: .*no \\.ply files"
    "-DMAKE_DIRECTORY=${_no_clouds}" -P "${_run_and_match}" --
    $<TARGET_FILE:point_cloud_to_depth> --source files --clouds "${_no_clouds}" ${_camera})

# Invalid input fails fast with a usage error (exit status 2) instead of
# starting a run.
pcd_app_test(pcd_app_rejects_unknown_option 2 "^error: unknown option" --colour true)
pcd_app_test(pcd_app_rejects_zero_count 2 "^error: " --count 0)
pcd_app_test(pcd_app_rejects_zero_width 2 "^error: synthetic images" --width 0)
pcd_app_test(pcd_app_rejects_inverted_range 2 "^error: depth range" --depth-min 5 --depth-max 1)
pcd_app_test(pcd_app_rejects_point_size 2 "^error: point_size must be 1, 3 or 5" --point-size 2)
pcd_app_test(pcd_app_rejects_depth_format 2 "^error: --depth-format" --depth-format meters)
pcd_app_test(pcd_app_rejects_extrinsics_with_synthetic 2 "^error: the synthetic camera"
  --extrinsics 1,0,0,0,0,1,0,0,0,0,1,0)
pcd_app_test(pcd_app_rejects_depth_scale_with_synthetic 2 "^error: .*drop --depth-scale"
  --depth-format uint16 --depth-scale 0.001)
# An image folder that cannot be looked into is a usage error, not a failed
# run: here a path through a link to itself, which no user can resolve.
set(_looped "${CMAKE_CURRENT_BINARY_DIR}/looped")
file(MAKE_DIRECTORY "${_looped}")
file(CREATE_LINK loop "${_looped}/loop" SYMBOLIC)
pcd_app_test(pcd_app_rejects_unreadable_image_dir 2 "^error: --image-dir is not a directory"
  --count 1 --image-dir "${_looped}/loop/images")
# Only the plan's verdict: an accepted plan has no warnings.
pcd_app_test(pcd_app_compile_only 0 "^point_cloud_to_depth: plan=OK\n*$" --compile-only true)

# A failed run reports its cause and exits 1.
set(_blocked "${CMAKE_CURRENT_BINARY_DIR}/blocked_images")
add_test(NAME pcd_app_failed_run
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=1 "-DEXPECT_OUTPUT=depth_000000.pgm.*verdict=FAIL"
    "-DMAKE_DIRECTORY=${_blocked}/depth_000000.pgm" -P "${_run_and_match}" --
    $<TARGET_FILE:point_cloud_to_depth> --count 5 --period-ms 5 --image-dir "${_blocked}")

# The timeout is the longest wait for an image, not for the whole run. Checking
# 2048x2048 images with 5x5 splats against the truth takes about 0.5 s each
# here: eight of them outlast a 2 s timeout, yet the run goes on and passes.
# A 4096x4096 image takes about 2 s, beyond a 1 s timeout: that run is stopped
# and fails. Each holds 1 to 3 GiB of GPU memory, so it runs alone.
pcd_app_test(pcd_app_slow_but_progressing 0 "frames=8/8 .*verdict=PASS"
  --width 2048 --height 2048 --point-size 5 --count 8 --period-ms 1 --timeout-seconds 2)
pcd_app_test(pcd_app_timed_out 1 "timed out: no image for 1 s.*frames=[01]/2 .*verdict=FAIL"
  --width 4096 --height 4096 --point-size 5 --count 2 --period-ms 1 --timeout-seconds 1)
set_tests_properties(pcd_app_slow_but_progressing pcd_app_timed_out PROPERTIES RUN_SERIAL TRUE)

# A stop request ends the run early with its own verdict, and the warning it
# leaves is shown. The signal goes out once the first image is written, so the
# run has started. The test sends SIGTERM, as a shell starts background jobs
# with SIGINT ignored, and its script has no semicolons, where CMake would
# split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped_images")
add_test(NAME pcd_app_stopped
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/100000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     \"$1\" --count 100000 --period-ms 10 --width 64 --height 48 --image-dir \"$2\" &
     for i in $(seq 600)
     do
       [ -e \"$2/depth_000000.pgm\" ] && break
       sleep 0.05
     done
     kill -TERM $! && wait $!"
    sh $<TARGET_FILE:point_cloud_to_depth> "${_stopped}")

# Another thread running before main(), as under a profiler or a sanitizer:
# the run goes ahead without managed signal handling, and says so.
add_test(NAME pcd_app_without_signal_domain
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS"
    -P "${_run_and_match}" -- ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:pcd_spawn_thread_preload>
    $<TARGET_FILE:point_cloud_to_depth> --count 10 --period-ms 5)

get_property(_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_tests} PROPERTIES LABELS "point_cloud_to_depth;gpu" TIMEOUT 300)
set_tests_properties(pcd_cloud_io pcd_clouds_setup pcd_clouds_xyz_setup pcd_clouds_mixed_setup
  pcd_clouds_empty_setup PROPERTIES LABELS "point_cloud_to_depth")
