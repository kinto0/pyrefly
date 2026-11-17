/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use lsp_server::ErrorCode;
use lsp_types::Range;
use lsp_types::SymbolInformation;
use lsp_types::Uri;
use pyrefly_lsp_test::object_model::ExternalProvider;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_lsp_test::object_model::TestTelemetry;
use pyrefly_util::telemetry::SubTaskTelemetry;
use pyrefly_util::telemetry::TelemetryEventKind;
use tempfile::TempDir;

/// Hold one search so later requests remain queued until cancellation is processed.
struct BlockingReferences {
    calls: Sender<String>,
    release: Receiver<()>,
}

impl ExternalProvider for BlockingReferences {
    fn find_references(
        &self,
        qualified_name: &str,
        _source_uri: &Uri,
        timeout: Duration,
        _telemetry: Option<SubTaskTelemetry>,
    ) -> anyhow::Result<Vec<(Uri, Vec<Range>)>> {
        self.calls.send(qualified_name.to_owned()).unwrap();
        if qualified_name == "sample.first" {
            self.release.recv_timeout(timeout).unwrap();
        }
        Ok(Vec::new())
    }

    fn workspace_symbols(
        &self,
        _query: &str,
        _workspace_uri: &Uri,
        _timeout: Duration,
        _telemetry: Option<SubTaskTelemetry>,
    ) -> anyhow::Result<Vec<SymbolInformation>> {
        Ok(Vec::new())
    }
}

/// A canceled queued request must not call either reference search provider.
#[test]
fn queued_reference_cancellation_skips_search() {
    let root = TempDir::new().unwrap();
    fs::write(
        root.path().join("sample.py"),
        "def first(): pass\ndef canceled(): pass\ndef live(): pass\n",
    )
    .unwrap();
    let (calls_tx, calls) = crossbeam_channel::unbounded();
    let (release, release_rx) = crossbeam_channel::bounded(1);
    let telemetry = TestTelemetry::new();
    let events = telemetry.subscribe();
    let mut interaction = LspInteraction::new_with_external_references(
        LspInteractionArgs {
            telemetry: Box::new(telemetry),
            ..Default::default()
        },
        Arc::new(BlockingReferences {
            calls: calls_tx,
            release: release_rx,
        }),
    );
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(Default::default()).unwrap();
    interaction.client.did_open("sample.py");
    let first = interaction.client.references("sample.py", 0, 5, false);
    let timeout = Duration::from_secs(10);
    assert_eq!(calls.recv_timeout(timeout).unwrap(), "sample.first");
    let wait_for_dispatch = |id: String| {
        let deadline = Instant::now() + timeout;
        loop {
            let event = events.recv_deadline(deadline).unwrap();
            if event.event.request_id.as_ref() == Some(&id)
                && matches!(&event.event.kind, TelemetryEventKind::LspEvent(name) if name.starts_with("LspRequest("))
            {
                break;
            }
        }
    };
    let canceled = interaction.client.references("sample.py", 1, 5, false);
    wait_for_dispatch(canceled.id().to_string());
    let live = interaction.client.references("sample.py", 2, 5, false);
    wait_for_dispatch(live.id().to_string());
    interaction.client.cancel_request(canceled.id());
    let deadline = Instant::now() + timeout;
    loop {
        let event = events.recv_deadline(deadline).unwrap();
        if matches!(&event.event.kind, TelemetryEventKind::LspEvent(name) if name == "CancelRequest")
        {
            break;
        }
    }
    release.send(()).unwrap();
    first
        .expect_response_with(|locations| {
            assert!(locations.is_some());
            true
        })
        .unwrap();
    canceled
        .expect_response_error_code(ErrorCode::RequestCanceled as i32)
        .unwrap();
    live.expect_response_with(|locations| {
        assert!(locations.is_some());
        true
    })
    .unwrap();
    assert_eq!(calls.recv_timeout(timeout).unwrap(), "sample.live");
    assert!(calls.is_empty());
    interaction.shutdown().unwrap();
}
