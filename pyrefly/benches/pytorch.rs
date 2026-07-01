/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Benchmark: type-error propagation latency across a large real-world codebase.
//!
//! Opens PyTorch through a full LSP server, rebinds the `Parameter` export in
//! `torch/nn/__init__.py` to a non-type value, and measures how long the
//! resulting error takes to surface in
//! `torch/distributed/pipelining/_backward.py` — a file deep in the dependency
//! graph that imports `Parameter` and uses it as a type (`Iterator[Parameter]`).
//! This is the incremental recheck latency a developer feels after an edit
//! ripples through the project.
//!
//! We rebind rather than delete the export so the trigger is independent of
//! filesystem case-sensitivity: simply removing `Parameter as Parameter` lets a
//! case-insensitive filesystem resolve `Parameter` to the `parameter.py`
//! submodule, masking the error.
//!
//! Like `micro.rs`, this runs under divan via CodSpeed's `codspeed-divan-compat`
//! harness. The expensive cold-start (server init + first full check) happens in
//! `with_inputs`, *outside* the measured region, so only the propagation is timed.
//!
//! PyTorch is vendored as a pinned git submodule at `benches/pytorch`. When that
//! submodule isn't checked out the benchmark skips, so `cargo bench` works in a
//! bare checkout; CI initializes the submodule before benching.
//!
//! Run with cargo: `cargo bench -p pyrefly --bench pytorch`

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

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
    let args = LspArgs {
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_indexing_limit: 50,
        build_system_blocking: false,
    };
    let mut interaction =
        LspInteraction::new_with_args(args, NoTelemetry, Some(ThreadCount::AllThreads), None);
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

fn main() {
    divan::main();
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
