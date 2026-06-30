/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A reusable, in-process harness for driving Pyrefly's language server from
//! tests and benchmarks. [`object_model::LspInteraction`] spawns the server on a
//! thread connected via in-memory channels and exposes a `TestClient` for sending
//! requests/notifications and asserting on responses and diagnostics.
//!
//! This lives in its own crate (rather than inside the `pyrefly` test module) so
//! that both the `pyrefly_lsp_interaction_tests` integration tests and the
//! `pyrefly` benchmarks — separate compilation units — can share it.

pub mod init;
pub mod object_model;
