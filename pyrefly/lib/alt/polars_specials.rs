/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Special handling for Polars `DataFrame` construction.
//!
//! Polars' stubs type a `DataFrame` as one opaque blob, so we synthesize a
//! `Type::DataFrame` carrying an inferred column schema when a DataFrame is built
//! from a dict literal. This is the entry point for column-aware checking.

use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprDict;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Number;
use ruff_python_ast::name::Name;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::types::class::Class;

pub fn is_polars_dataframe(cls: &Class) -> bool {
    cls.has_toplevel_qname("polars.dataframe.frame", "DataFrame")
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Infer a column schema from a `pl.DataFrame({...})` dict literal, or `None` to
    /// fall back to plain construction.
    ///
    /// Extraction is purely syntactic and never infers the element expressions.
    /// Duplicate keys yield `None`: Python keeps only the last value for a repeated
    /// key, so one column per syntactic entry would misdescribe the runtime schema.
    pub fn infer_polars_schema(&self, dict: &ExprDict) -> Option<Vec<(Name, Type)>> {
        if dict.items.is_empty() {
            return None;
        }
        let mut columns = Vec::with_capacity(dict.items.len());
        let mut seen = SmallSet::new();
        for item in &dict.items {
            let Some(Expr::StringLiteral(key)) = &item.key else {
                return None;
            };
            let name = Name::new(key.value.to_str());
            if !seen.insert(name.clone()) {
                return None;
            }
            let Expr::List(ExprList { elts, .. }) = &item.value else {
                return None;
            };
            columns.push((name, self.polars_list_element_type(elts)?));
        }
        Some(columns)
    }

    /// The modeled element type of a list literal: `int` if all elements are integer
    /// literals, `str` if all are string literals, else `None`.
    fn polars_list_element_type(&self, elts: &[Expr]) -> Option<Type> {
        if elts.is_empty() {
            return None;
        }
        if elts.iter().all(|e| {
            matches!(
                e,
                Expr::NumberLiteral(ExprNumberLiteral {
                    value: Number::Int(_),
                    ..
                })
            )
        }) {
            Some(self.heap.mk_class_type(self.stdlib.int().clone()))
        } else if elts.iter().all(|e| matches!(e, Expr::StringLiteral(_))) {
            Some(self.heap.mk_class_type(self.stdlib.str().clone()))
        } else {
            None
        }
    }
}
