# SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs a command and passes only if it exits with EXPECT_EXIT (default 0) and
# its combined output matches EXPECT_OUTPUT. CTest's own PASS_REGULAR_EXPRESSION
# would ignore the exit status.
#
#   cmake -DEXPECT_EXIT=2 -DEXPECT_OUTPUT=regex [-DMAKE_DIRECTORY=dir]
#         -P run_and_match.cmake -- command [args...]
#
# MAKE_DIRECTORY is created first, for a test that needs one in its way.

if(NOT DEFINED EXPECT_EXIT)
  set(EXPECT_EXIT 0)
endif()
if(NOT DEFINED EXPECT_OUTPUT)
  message(FATAL_ERROR "run_and_match: EXPECT_OUTPUT is required")
endif()

set(_command)
set(_after_separator FALSE)
math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_index RANGE ${_last})
  if(_after_separator)
    list(APPEND _command "${CMAKE_ARGV${_index}}")
  elseif(CMAKE_ARGV${_index} STREQUAL "--")
    set(_after_separator TRUE)
  endif()
endforeach()
if(NOT _command)
  message(FATAL_ERROR "run_and_match: no command after --")
endif()

if(DEFINED MAKE_DIRECTORY)
  file(MAKE_DIRECTORY "${MAKE_DIRECTORY}")
endif()

execute_process(COMMAND ${_command}
  RESULT_VARIABLE _status OUTPUT_VARIABLE _output ERROR_VARIABLE _output)
message("${_output}")
if(NOT _status STREQUAL "${EXPECT_EXIT}")
  message(FATAL_ERROR "exit status ${_status}, expected ${EXPECT_EXIT}")
endif()
if(NOT _output MATCHES "${EXPECT_OUTPUT}")
  message(FATAL_ERROR "output does not match: ${EXPECT_OUTPUT}")
endif()
