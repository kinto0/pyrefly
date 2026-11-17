/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_types::function::Deprecation;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprStringLiteral;

use crate::export::special::SpecialExport;

fn extract_message(call: &ExprCall) -> Option<String> {
    if let Some(arg) = call.arguments.args.first()
        && let Expr::StringLiteral(ExprStringLiteral { value, .. }) = arg
    {
        Some(value.to_string())
    } else {
        None
    }
}

/// Parse a decorator and return its deprecation metadata if it represents `@deprecated`.
pub fn parse_deprecation(e: &Expr) -> Option<Deprecation> {
    let call = e.as_call_expr()?;
    if SpecialExport::from_syntax(&call.func) != Some(SpecialExport::Deprecated) {
        return None;
    }
    Some(Deprecation::new(extract_message(call)))
}
