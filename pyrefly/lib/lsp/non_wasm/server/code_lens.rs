/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Server integration for runnable CodeLens commands.

mod runnable;

use lsp_types::CodeLens;
use lsp_types::CodeLensParams;
use lsp_types::CodeLensRequest;
use lsp_types::Request as _;

use super::Server;
use crate::state::state::Transaction;

impl Server {
    /// Return the enabled Run/Test commands for one document.
    pub(super) fn code_lens(
        &self,
        transaction: &Transaction<'_>,
        params: CodeLensParams,
    ) -> Option<Vec<CodeLens>> {
        let uri = &params.text_document.uri;
        let handle = self
            .make_handle_if_enabled(uri, Some(CodeLensRequest::METHOD.as_str()))
            .ok()?;
        runnable::code_lenses(self, transaction, &handle, uri).ok()
    }
}
