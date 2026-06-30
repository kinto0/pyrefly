/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Microbenchmarks for Pyrefly type checking, run with divan via CodSpeed's
//! drop-in `codspeed-divan-compat` harness. Locally (and in internal buck
//! builds) this behaves like a normal divan walltime harness; in OSS CI under
//! `cargo codspeed` it emits instrumented measurements to CodSpeed.
//!
//! Each benchmark uses Pyrefly's in-memory checking API (`State` / `Transaction`)
//! with `SHARED_STATE` to pre-initialize stdlib, so only the checking of the
//! benchmark snippet is measured.
//!
//! Run with cargo: `cargo bench -p pyrefly --bench micro`
//! Run with buck: `buck run @fbcode//mode/opt fbcode//pyrefly/pyrefly:micro_bench -- --bench`

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use dupe::Dupe;
use pyrefly::state::load::FileContents;
use pyrefly::state::require::Require;
use pyrefly::state::state::State;
use pyrefly_build::handle::Handle;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::thread_pool::ThreadCount;

const BENCH_FILE: &str = "bench.py";

/// Single-threaded state with stdlib pre-initialized.
static SHARED_STATE: LazyLock<State> = LazyLock::new(|| {
    let sys_info = SysInfo::new(PythonVersion::default(), PythonPlatform::default());
    let config = {
        let mut c = ConfigFile::default();
        c.python_environment.python_version = Some(PythonVersion::default());
        c.python_environment.python_platform = Some(PythonPlatform::default());
        c.configure();
        ArcId::new(c)
    };
    let finder = ConfigFinder::new_constant(config);
    let state = State::new(finder, ThreadCount::NumThreads(NonZeroUsize::MIN));
    // Force stdlib init by running an empty module.
    let h = Handle::new(
        ModuleName::from_str("_bench_init"),
        ModulePath::memory(PathBuf::from("_bench_init.py")),
        sys_info,
    );
    let mut t = state.new_committable_transaction(Require::Errors, None);
    t.as_mut().set_memory(vec![(
        PathBuf::from("_bench_init.py"),
        Some(Arc::new(FileContents::from_source(String::new()))),
    )]);
    t.as_mut().run(&[h], Require::Errors, None);
    state.commit_transaction(t, None);
    state
});

/// Run a type check on the given Python code and return the error count.
fn check_code(code: Arc<FileContents>) -> usize {
    let sys_info = SysInfo::new(PythonVersion::default(), PythonPlatform::default());
    let h = Handle::new(
        ModuleName::from_str("bench"),
        ModulePath::memory(PathBuf::from(BENCH_FILE)),
        sys_info,
    );
    let mut t = SHARED_STATE.transaction();
    t.set_memory(vec![(PathBuf::from(BENCH_FILE), Some(code))]);
    t.run(&[h.dupe()], Require::Errors, None);
    let errors = t.get_errors([&h]);
    errors.collect_errors().ordinary.len()
}

fn main() {
    divan::main();
}

/// Smoke benchmark validating the harness end-to-end. Real benchmarks are added
/// on top of this scaffolding.
#[divan::bench]
fn smoke(bencher: divan::Bencher) {
    let code = Arc::new(FileContents::from_source("x: int = 1".to_owned()));
    assert_eq!(check_code(code.dupe()), 0);
    bencher.bench(|| check_code(code.dupe()));
}
