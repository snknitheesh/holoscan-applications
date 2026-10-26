// SPDX-FileCopyrightText: Copyright (c) 2026 Nitheesh Kumar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Preloaded into the application by a test: starts a thread before main(), as
// profilers, sanitizers and other preloaded libraries do.

#include <chrono>
#include <thread>

namespace {

struct StartThread {
  StartThread() {
    std::thread([] { std::this_thread::sleep_for(std::chrono::hours(1)); }).detach();
  }
} start_thread;

}  // namespace
