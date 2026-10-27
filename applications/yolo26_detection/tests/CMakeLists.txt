# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# These tests build with the SDK and its CUDA toolchain. All but the host-side
# core tests also need a GPU and its driver, and carry the `gpu` label. None
# needs a download: the stub model stands in for YOLO26. The tests of the real
# model are skipped unless it has been prepared in YOLO26_DATA_DIR (see the
# README).
set(YOLO26_DATA_DIR "${CMAKE_SOURCE_DIR}/data/yolo26_detection" CACHE PATH
  "Where prepare_model.py wrote the YOLO26 model, for the real-model tests")

add_executable(yolo26_letterbox_test letterbox_test.cu)
target_link_libraries(yolo26_letterbox_test PRIVATE yolo26_detection_core)
add_executable(yolo26_decode_test decode_test.cu)
target_link_libraries(yolo26_decode_test PRIVATE yolo26_detection_core)
# The tests' own evaluations of the kernels' expressions must not contract
# either.
set_source_files_properties(letterbox_test.cu decode_test.cu PROPERTIES COMPILE_OPTIONS
  "-fmad=false")
add_executable(yolo26_core_test core_test.cpp)
target_link_libraries(yolo26_core_test PRIVATE yolo26_detection_core)
add_executable(yolo26_operator_test operator_test.cpp)
target_link_libraries(yolo26_operator_test PRIVATE yolo26_detection_operators holoscan::testing)
add_executable(yolo26_make_test_images make_test_images.cpp)
target_link_libraries(yolo26_make_test_images PRIVATE yolo26_detection_core CUDA::nvjpeg)
add_library(yolo26_spawn_thread_preload SHARED spawn_thread_preload.cpp)

add_test(NAME yolo26_letterbox COMMAND yolo26_letterbox_test)
add_test(NAME yolo26_decode COMMAND yolo26_decode_test)
add_test(NAME yolo26_core COMMAND yolo26_core_test "${CMAKE_CURRENT_BINARY_DIR}/core_files")
add_test(NAME yolo26_operators
  COMMAND yolo26_operator_test "${CMAKE_CURRENT_BINARY_DIR}/operator_models")

# Runs the application through run_and_match.cmake, which checks its exit
# status as well as its output: yolo26_app_test(<name> <exit status> <output
# regex> <arguments>...). It runs in this directory, where `--model stub`
# writes the stub model and its engine is cached.
set(_run_and_match "${CMAKE_CURRENT_SOURCE_DIR}/run_and_match.cmake")
function(yolo26_app_test name status output)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=${status} "-DEXPECT_OUTPUT=${output}"
      -P "${_run_and_match}" -- $<TARGET_FILE:yolo26_detection> ${ARGN})
endfunction()

# Builds the stub model's engine once, for the tests that use its files.
add_test(NAME yolo26_stub_setup
  COMMAND yolo26_detection --model stub --count 1 --period-ms 1)
set_tests_properties(yolo26_stub_setup PROPERTIES FIXTURES_SETUP yolo26_stub)

# Synthetic frames with the stub model, every detection checked against the
# frame's known ones: every encoding, GPU and host memory, wide, tall, odd
# padding, small and large frames, and another input size.
set(_truth "missed=0 extra=0 .*verdict=PASS")
set(_stub --model stub --period-ms 5)
yolo26_app_test(yolo26_app_synthetic 0 "frames=20/20 detections=40 .*${_truth}"
  ${_stub} --count 20)
yolo26_app_test(yolo26_app_bgr8 0 "frames=20/20 .*${_truth}" ${_stub} --count 20 --encoding bgr8)
yolo26_app_test(yolo26_app_rgba8_host 0 "frames=20/20 .*${_truth}"
  ${_stub} --count 20 --encoding rgba8 --host-input true)
yolo26_app_test(yolo26_app_bgra8_tall 0 "frames=20/20 .*${_truth}"
  ${_stub} --count 20 --encoding bgra8 --width 720 --height 1280)
yolo26_app_test(yolo26_app_odd_padding 0 "frames=20/20 .*${_truth}"
  ${_stub} --count 20 --width 1000 --height 333)
yolo26_app_test(yolo26_app_small 0 "frames=20/20 .*${_truth}"
  ${_stub} --count 20 --width 64 --height 48)
yolo26_app_test(yolo26_app_input_320 0 "frames=20/20 .*${_truth}"
  ${_stub} --count 20 --width 1920 --height 1080 --input-size 320)
yolo26_app_test(yolo26_app_fp16 0 "frames=20/20 .*${_truth}" ${_stub} --count 20 --fp16 true)
# Wrong detections fail the run: too small a capacity drops the second one of
# every frame, too high a threshold both.
yolo26_app_test(yolo26_app_capacity_too_small 1 "missed=20 extra=0 .*verdict=FAIL"
  ${_stub} --count 20 --max-detections 1)
yolo26_app_test(yolo26_app_threshold_too_high 1 "detections=0 .*missed=40 .*verdict=FAIL"
  ${_stub} --count 20 --score-threshold 0.95)

# Image output: each frame with its boxes drawn.
set(_images "${CMAKE_CURRENT_BINARY_DIR}/images")
add_test(NAME yolo26_app_image_dir
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=files_written=3 .*verdict=PASS"
    "-DMAKE_DIRECTORY=${_images}" -P "${_run_and_match}" --
    $<TARGET_FILE:yolo26_detection> ${_stub} --count 3 --image-dir "${_images}")

# Image files of synthetic scenes, written by the fixture: a baseline JPEG, a
# PPM and a progressive JPEG, decoded as RGB8 or BGR8, each with the stub
# model's two detections expected.
set(_files "${CMAKE_CURRENT_BINARY_DIR}/test_images")
add_test(NAME yolo26_images_setup COMMAND yolo26_make_test_images "${_files}")
set_tests_properties(yolo26_images_setup PROPERTIES FIXTURES_SETUP yolo26_images)
set(_from_files ${_stub} --source files --images "${_files}" --expect "${_files}/expected.csv"
  --expect-iou 0.9)
yolo26_app_test(yolo26_app_files 0
  "scene_0\\.jpg: 2 detections\n  class 0 \\(0\\) 0\\.900 .*scene_1\\.ppm: 2 detections.*scene_2\\.JPEG: 2 detections.*frames=3/3 detections=6 .*expected: matched=6/6\nyolo26_detection: verdict=PASS"
  ${_from_files})
yolo26_app_test(yolo26_app_files_bgr8 0 "expected: matched=6/6\n.*verdict=PASS"
  ${_from_files} --encoding bgr8)
yolo26_app_test(yolo26_app_files_loop 0 "frames=7/7 .*expected: matched=14/14\n.*verdict=PASS"
  ${_from_files} --loop true --count 7)
yolo26_app_test(yolo26_app_one_file 0 "frames=1/1 .*expected: matched=2/2\n.*verdict=PASS"
  ${_stub} --source files --images "${_files}/scene_2.JPEG" --expect "${_files}/expected.csv")
# Expectations the detections do not meet, or files without any detection,
# fail the run.
set(_shifted "${CMAKE_CURRENT_BINARY_DIR}/shifted.csv")
file(WRITE "${_shifted}" "scene_0.jpg,0,600,300,700,400\nscene_0.jpg,1,0,0,10,10\n")
yolo26_app_test(yolo26_app_files_unmatched 1 "expected: matched=0/2\n.*verdict=FAIL"
  ${_stub} --source files --images "${_files}/scene_0.jpg" --expect "${_shifted}")
yolo26_app_test(yolo26_app_files_none_found 1
  "warning: no detection in any image.*detections=0 .*verdict=FAIL"
  ${_stub} --source files --images "${_files}" --score-threshold 0.95)
yolo26_app_test(yolo26_app_rejects_count_beyond_files 2 "^error: .*--loop true"
  ${_stub} --source files --images "${_files}" --count 4)
yolo26_app_test(yolo26_app_rejects_rgba8_files 2 "^error: image files are decoded as rgb8"
  ${_stub} --source files --images "${_files}" --encoding rgba8)
yolo26_app_test(yolo26_app_rejects_size_with_files 2 "^error: --width, --height"
  ${_stub} --source files --images "${_files}" --width 640)
yolo26_app_test(yolo26_app_rejects_verify_with_files 2 "^error: --verify checks synthetic"
  ${_stub} --source files --images "${_files}" --verify true)
set_tests_properties(yolo26_app_files yolo26_app_files_bgr8 yolo26_app_files_loop
  yolo26_app_one_file yolo26_app_files_unmatched yolo26_app_files_none_found
  yolo26_app_rejects_count_beyond_files yolo26_app_rejects_rgba8_files
  yolo26_app_rejects_size_with_files yolo26_app_rejects_verify_with_files
  PROPERTIES FIXTURES_REQUIRED yolo26_images)

# A serialized engine instead of an ONNX file: the one cached for the stub.
# And a model exported for another input size than configured.
set(_stub_onnx "${CMAKE_CURRENT_BINARY_DIR}/yolo26_stub_640.onnx")
add_test(NAME yolo26_app_engine_file
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=expected: matched=6/6\n.*verdict=PASS"
    -P "${_run_and_match}" -- sh -c
    "exec \"$1\" --model \"$(ls yolo26_stub_640.*.engine.fp32 | head -n 1)\" --num-classes 2 --source files --images \"$2\" --expect \"$2/expected.csv\" --period-ms 5"
    sh $<TARGET_FILE:yolo26_detection> "${_files}")
yolo26_app_test(yolo26_app_rejects_other_input_size 1
  "export it for this input size.*verdict=FAIL" --model "${_stub_onnx}" --input-size 320
  --num-classes 2 --count 1)
set_tests_properties(yolo26_app_engine_file PROPERTIES FIXTURES_REQUIRED "yolo26_stub;yolo26_images")
set_tests_properties(yolo26_app_rejects_other_input_size PROPERTIES FIXTURES_REQUIRED yolo26_stub)
# A file that is no model fails the run when the detector starts.
set(_junk "${CMAKE_CURRENT_BINARY_DIR}/junk.onnx")
file(WRITE "${_junk}" "not a model")
yolo26_app_test(yolo26_app_rejects_junk_model 1 "cannot load .*junk\\.onnx.*verdict=FAIL"
  --model "${_junk}" --count 1)

# Invalid input fails fast with a usage error (exit status 2) instead of
# starting a run.
yolo26_app_test(yolo26_app_rejects_unknown_option 2 "^error: unknown option" ${_stub} --colour 1)
yolo26_app_test(yolo26_app_rejects_no_model 2 "^error: --model is required" --count 1)
yolo26_app_test(yolo26_app_rejects_missing_model 2 "^error: --model: no such file"
  --model "${CMAKE_CURRENT_BINARY_DIR}/missing.onnx")
yolo26_app_test(yolo26_app_rejects_verify_without_stub 2 "^error: --verify needs --model stub"
  --model "${_junk}" --verify true)
yolo26_app_test(yolo26_app_rejects_input_size 2 "^error: input_size must be a multiple of 32"
  ${_stub} --input-size 100)
yolo26_app_test(yolo26_app_rejects_threshold 2 "^error: score_threshold" ${_stub}
  --score-threshold 2)
yolo26_app_test(yolo26_app_rejects_expect_with_synthetic 2 "^error: --expect applies to image"
  ${_stub} --expect "${_shifted}")
yolo26_app_test(yolo26_app_rejects_files_without_images 2 "^error: --source files needs --images"
  ${_stub} --source files)
yolo26_app_test(yolo26_app_rejects_missing_images 2 "^error: .*cannot list"
  ${_stub} --source files --images "${CMAKE_CURRENT_BINARY_DIR}/no_such_folder")
yolo26_app_test(yolo26_app_rejects_missing_labels 2 "^error: .*labels.txt: cannot open"
  ${_stub} --labels "${CMAKE_CURRENT_BINARY_DIR}/labels.txt")
yolo26_app_test(yolo26_app_rejects_num_classes_with_stub 2 "^error: the stub model has 2 classes"
  ${_stub} --num-classes 80)
yolo26_app_test(yolo26_app_rejects_small_frames 2 "^error: synthetic images must be" ${_stub}
  --width 32)
yolo26_app_test(yolo26_app_rejects_zero_period 2 "^error: --period-ms" ${_stub} --period-ms 0)
yolo26_app_test(yolo26_app_rejects_encoding 2 "^error: --encoding" ${_stub} --encoding mono8)
yolo26_app_test(yolo26_app_rejects_image_dir 2 "^error: --image-dir is not a directory"
  ${_stub} --image-dir "${_junk}")
yolo26_app_test(yolo26_app_help 0 "^Usage: .*--model PATH\\|stub" --help)
# Only the plan's verdict: an accepted plan has no warnings.
yolo26_app_test(yolo26_app_compile_only 0 "^yolo26_detection: plan=OK\n*$"
  ${_stub} --compile-only true)

# A stop request ends the run early with its own verdict, and the warning it
# leaves is shown. The signal goes out once the first frame is written, so the
# run has started. The test sends SIGTERM, as a shell starts background jobs
# with SIGINT ignored, and its script has no semicolons, where CMake would
# split it.
set(_stopped "${CMAKE_CURRENT_BINARY_DIR}/stopped_images")
add_test(NAME yolo26_app_stopped
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=1
    "-DEXPECT_OUTPUT=warning RUNTIME_MANAGED_SIGNAL_RECEIVED.*frames=[0-9]+/100000 .*verdict=STOPPED"
    -P "${_run_and_match}" -- sh -c
    "rm -rf \"$2\" && mkdir -p \"$2\" || exit 1
     \"$1\" --model stub --count 100000 --period-ms 10 --width 64 --height 48 --image-dir \"$2\" &
     for i in $(seq 600)
     do
       [ -e \"$2/detections_000001.ppm\" ] && break
       sleep 0.05
     done
     kill -TERM $! && wait $!"
    sh $<TARGET_FILE:yolo26_detection> "${_stopped}")

# Another thread running before main(), as under a profiler or a sanitizer:
# the run goes ahead without managed signal handling, and says so.
add_test(NAME yolo26_app_without_signal_domain
  COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0
    "-DEXPECT_OUTPUT=without managed SIGINT/SIGTERM handling.*verdict=PASS"
    -P "${_run_and_match}" -- ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:yolo26_spawn_thread_preload>
    $<TARGET_FILE:yolo26_detection> --model stub --count 10 --period-ms 5)

# The real YOLO26n model on a COCO validation image, in RGB8 and BGR8, and
# with FP16: the two cats and the remote that Ultralytics' own predictor finds
# in it must be found here, overlapping its boxes by an IoU of at least 0.9.
# Skipped (exit 77) unless prepare_model.py has written the model and image.
# The first run builds and caches the engines, which takes a while.
set(_model "${YOLO26_DATA_DIR}/yolo26n.onnx")
set(_coco "${YOLO26_DATA_DIR}/images/000000039769.jpg")
set(_coco_expected "${CMAKE_CURRENT_BINARY_DIR}/coco_expected.csv")
file(WRITE "${_coco_expected}"
  "# Ultralytics 8.4 with ONNX Runtime: cat 0.912, cat 0.899, remote 0.857\n"
  "000000039769.jpg,15,343.96,24.52,640.00,374.83\n"
  "000000039769.jpg,15,6.91,55.12,317.26,466.10\n"
  "000000039769.jpg,65,40.41,73.99,175.90,118.56\n")
function(yolo26_real_test name output)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=0 "-DEXPECT_OUTPUT=${output}"
      "-DSKIP_UNLESS_EXISTS=${_model}|${_coco}" -P "${_run_and_match}" --
      $<TARGET_FILE:yolo26_detection> --model "${_model}"
      --labels "${YOLO26_DATA_DIR}/yolo26n.labels.txt" --source files --images "${_coco}"
      --expect "${_coco_expected}" --expect-iou 0.9 ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 RUN_SERIAL TRUE)
endfunction()
set(_cats "cat \\(15\\) 0\\.9[0-9]+ .*cat \\(15\\) 0\\.[89][0-9]+ .*remote \\(65\\) 0\\.8[0-9]+ ")
yolo26_real_test(yolo26_real_coco "${_cats}.*expected: matched=3/3\n.*verdict=PASS")
yolo26_real_test(yolo26_real_coco_bgr8 "${_cats}.*expected: matched=3/3\n.*verdict=PASS"
  --encoding bgr8)
yolo26_real_test(yolo26_real_coco_fp16 "expected: matched=3/3\n.*verdict=PASS" --fp16 true)
yolo26_real_test(yolo26_real_coco_loop "frames=50/50 .*expected: matched=150/150\n.*verdict=PASS"
  --loop true --count 50 --period-ms 5)

get_property(_tests DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS _tests)
  get_test_property(${_test} TIMEOUT _timeout)
  if(NOT _timeout)
    set_tests_properties(${_test} PROPERTIES TIMEOUT 300)
  endif()
endforeach()
set_tests_properties(${_tests} PROPERTIES LABELS "yolo26_detection;gpu")
set_tests_properties(yolo26_core PROPERTIES LABELS "yolo26_detection")
