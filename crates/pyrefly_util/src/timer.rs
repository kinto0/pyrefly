/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A wall-clock timer for internal profiling counters that can be globally
//! disabled.
//!
//! `Instant::now()` reads `CLOCK_MONOTONIC` via the vDSO, so it is nearly free
//! at runtime. Under Valgrind (which CodSpeed's instrumented instrument uses),
//! the vDSO is bypassed and every reading becomes a real `clock_gettime`
//! syscall. A single type-check makes dozens of them, and CodSpeed cannot
//! instrument syscalls, so they pollute the measurement. Benchmarks call
//! [`set_timing_enabled(false)`] to make every [`Timer`] a no-op; production
//! leaves timing enabled so telemetry counters stay populated.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use web_time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable all [`Timer`]s process-wide. Enabled by default; a disabled
/// timer never calls `Instant::now()` and reports zero elapsed time.
pub fn set_timing_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// A profiling timer that captures a start instant only when timing is enabled.
/// When disabled, `elapsed*` return zero and no `clock_gettime` syscall is made.
#[derive(Debug, Clone, Copy)]
pub struct Timer(Option<Instant>);

impl Timer {
    /// Start a timer, reading the clock only if timing is enabled.
    pub fn start() -> Self {
        Self(ENABLED.load(Ordering::Relaxed).then(Instant::now))
    }

    pub fn elapsed(&self) -> Duration {
        self.0.map_or(Duration::ZERO, |start| start.elapsed())
    }

    pub fn elapsed_nanos(&self) -> u64 {
        self.0.map_or(0, |start| start.elapsed().as_nanos() as u64)
    }
}
