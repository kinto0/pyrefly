/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! LSP benchmarks against a large real-world codebase (PyTorch), run under divan
//! via CodSpeed's `codspeed-divan-compat` harness like `micro.rs`.
//!
//! Two benchmarks:
//!
//! - `cold_start_go_to_definition` — time to first index. From a *cold* server,
//!   measures the whole path a developer waits through when opening a project:
//!   initialize, open a file deep in the dependency graph, and answer the first
//!   cross-file go-to-definition (on `from torch.nn import Parameter`). The
//!   response can only be produced once the file and its import closure have been
//!   analyzed, so this captures the initial index/check latency.
//!
//! - `error_propagation` — incremental recheck latency. With the server already
//!   warm, rebinds the `Parameter` export in `torch/nn/__init__.py` to a non-type
//!   value and measures how long the resulting error takes to surface in
//!   `torch/distributed/pipelining/_backward.py`. We rebind rather than delete the
//!   export so the trigger is independent of filesystem case-sensitivity: deleting
//!   `Parameter as Parameter` lets a case-insensitive filesystem resolve
//!   `Parameter` to the `parameter.py` submodule, masking the error. The warm-up
//!   happens in `with_inputs`, *outside* the measured region.
//!
//! PyTorch is vendored as a pinned git submodule at `benches/pytorch`. When that
//! submodule isn't checked out the benchmarks skip, so `cargo bench` works in a
//! bare checkout; CI initializes the submodule before benching.
//!
//! Run with cargo: `cargo bench -p pyrefly --bench pytorch`

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use lsp_types::GotoDefinitionResponse;
use lsp_types::Url;
use pyrefly::commands::lsp::IndexingMode;
use pyrefly::commands::lsp::LspArgs;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_util::fs_anyhow::read_to_string;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;
use serde_json::json;

/// File whose `Parameter` re-export we remove to trigger propagation.
const NN_INIT: &str = "torch/nn/__init__.py";
/// File deep in the import graph where the error must eventually surface.
const BACKWARD: &str = "torch/distributed/pipelining/_backward.py";
/// Position of `Parameter` in `_backward.py`'s `from torch.nn import Parameter`
/// (0-indexed), the cross-file symbol the cold-start benchmark navigates to.
const PARAM_LINE: u32 = 9;
const PARAM_COL: u32 = 21;

/// Standard LSP args used by both benchmarks. `LazyBlocking` indexing plus a
/// workspace folder mirrors a real IDE session opening the project.
fn lsp_args() -> LspArgs {
    LspArgs {
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_indexing_limit: 50,
        build_system_blocking: false,
    }
}

/// Path to the pinned PyTorch submodule, or `None` if it isn't checked out.
fn pytorch_root() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch");
    root.join(NN_INIT).exists().then_some(root)
}

/// Restores a file's original contents on drop, so the in-place edit the
/// benchmark makes never leaves the submodule working tree dirty.
struct RestoreOnDrop {
    path: PathBuf,
    original: String,
}

impl Drop for RestoreOnDrop {
    fn drop(&mut self) {
        let _ = fs::write(&self.path, &self.original);
    }
}

/// A server with both files open and first diagnostics already received — the
/// state right before the edit whose propagation we measure.
struct Prepared {
    interaction: LspInteraction,
    backward_path: PathBuf,
    modified_content: String,
    restore: RestoreOnDrop,
}

/// Cold start (unmeasured): launch the server over all cores, open both files,
/// and wait for the initial full check to settle.
fn prepare(root: &Path) -> Prepared {
    let mut interaction =
        LspInteraction::new_with_args(lsp_args(), NoTelemetry, Some(ThreadCount::AllThreads), None);
    // Valgrind instrumentation makes the initial check far slower than native, so
    // the server can stay silent well past the interactive-test defaults.
    interaction
        .client
        .set_timeouts(Duration::from_secs(120), Duration::from_secs(3600));
    interaction.set_root(root.to_path_buf());

    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            workspace_folders: Some(vec![(
                "pytorch".to_owned(),
                Url::from_file_path(root).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();

    let nn_init_path = root.join(NN_INIT);
    let backward_path = root.join(BACKWARD);

    interaction.client.did_open(NN_INIT);
    interaction.client.did_open(BACKWARD);

    // Wait for the server to finish the initial check of both files. We only
    // require that diagnostics arrive (not a specific count, which varies with
    // the pyrefly version and platform); `_backward.py`'s diagnostics arrive once
    // its import closure — reaching torch.nn — has resolved, marking the end of
    // cold start.
    interaction
        .client
        .expect_publish_diagnostics_for_file(nn_init_path.clone())
        .unwrap();
    interaction
        .client
        .expect_publish_diagnostics_for_file(backward_path.clone())
        .unwrap();

    let original = read_to_string(&nn_init_path).unwrap();
    // Rebind `Parameter` to an `int`; dependents that use it as a type then error.
    let modified_content =
        format!("{original}\nParameter = 42  # pyrefly bench: force a propagated error\n");

    Prepared {
        interaction,
        backward_path,
        modified_content,
        restore: RestoreOnDrop {
            path: nn_init_path,
            original,
        },
    }
}

/// Measured region: apply the edit and wait for the resulting error to reach the
/// far file. Dependencies only recheck off saved files, so we write to disk and
/// notify via the file watcher rather than relying on the in-memory change alone.
fn measure(p: &mut Prepared) {
    let client = &p.interaction.client;
    client.did_change(NN_INIT, &p.modified_content);
    fs::write(&p.restore.path, &p.modified_content).unwrap();
    client.file_modified(NN_INIT);
    client.did_save(NN_INIT);

    // `Iterator[Parameter]` is now `Iterator[int-instance]`, which is not a type.
    client
        .expect_publish_diagnostics_eventual_message_contains(
            p.backward_path.clone(),
            "Expected a type form, got instance of `int`",
        )
        .unwrap();
}

/// Full cold start: launch a fresh server, initialize, open a deep file, and
/// resolve the first cross-file go-to-definition. Returns the interaction so it
/// is dropped (and the server torn down) outside the measured region.
fn cold_start_definition(root: &Path) -> LspInteraction {
    let mut interaction =
        LspInteraction::new_with_args(lsp_args(), NoTelemetry, Some(ThreadCount::AllThreads), None);
    interaction
        .client
        .set_timeouts(Duration::from_secs(120), Duration::from_secs(3600));
    interaction.set_root(root.to_path_buf());

    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            workspace_folders: Some(vec![(
                "pytorch".to_owned(),
                Url::from_file_path(root).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open(BACKWARD);
    interaction
        .client
        .definition(BACKWARD, PARAM_LINE, PARAM_COL)
        .expect_response_with(|resp: Option<GotoDefinitionResponse>| match resp {
            Some(GotoDefinitionResponse::Scalar(_)) => true,
            Some(GotoDefinitionResponse::Array(locs)) => !locs.is_empty(),
            Some(GotoDefinitionResponse::Link(links)) => !links.is_empty(),
            None => false,
        })
        .unwrap();

    interaction
}

fn main() {
    divan::main();
}

/// Time to first index: the cold path from server launch to the first answered
/// cross-file navigation. A fresh server per sample (sample_count = 1) makes each
/// run a genuine cold start; the check dominates, so one sample suffices.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn cold_start_go_to_definition(bencher: divan::Bencher) {
    let Some(root) = pytorch_root() else {
        eprintln!(
            "Skipping pytorch benchmark: submodule not checked out. Run \
             `git submodule update --init pyrefly/benches/pytorch`."
        );
        return;
    };
    bencher.bench_local(|| cold_start_definition(&root));
}

/// One sample only: each run is a full project recheck, and a fresh server is
/// generated per sample (the edit is not idempotent), so repeating it would only
/// multiply cost without improving the instruction-count measurement.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn error_propagation(bencher: divan::Bencher) {
    let Some(root) = pytorch_root() else {
        eprintln!(
            "Skipping pytorch benchmark: submodule not checked out. Run \
             `git submodule update --init pyrefly/benches/pytorch`."
        );
        return;
    };
    bencher
        .with_inputs(|| prepare(&root))
        .bench_local_refs(measure);
}
