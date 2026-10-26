// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// A minimal check-and-count harness: the SDK image has no unit-test framework,
// and these tests need none beyond named checks and a failing exit status.

#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace depth_to_point_cloud::testing {

inline int& failures() {
  static int count = 0;
  return count;
}

inline void check(bool condition, const std::string& what, const char* file, int line) {
  if (!condition) {
    ++failures();
    std::cerr << file << ':' << line << ": check failed: " << what << '\n';
  }
}

struct TestCase {
  std::string name;
  std::function<void()> body;
};

// Runs every case, reporting each, and returns the process exit status.
inline int run(const std::vector<TestCase>& cases) {
  for (const TestCase& test : cases) {
    const int before = failures();
    try {
      test.body();
    } catch (const std::exception& error) {
      ++failures();
      std::cerr << test.name << ": unexpected exception: " << error.what() << '\n';
    }
    std::cout << (failures() == before ? "[ PASS ] " : "[ FAIL ] ") << test.name << '\n';
  }
  std::cout << (failures() == 0 ? "all passed" : std::to_string(failures()) + " check(s) failed")
            << '\n';
  return failures() == 0 ? 0 : 1;
}

template <typename Exception, typename Body>
bool throws(Body&& body) {
  try {
    std::forward<Body>(body)();
  } catch (const Exception&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

}  // namespace depth_to_point_cloud::testing

#define VERIFY(condition) \
  ::depth_to_point_cloud::testing::check((condition), #condition, __FILE__, __LINE__)
#define VERIFY_MSG(condition, message) \
  ::depth_to_point_cloud::testing::check((condition), (message), __FILE__, __LINE__)
