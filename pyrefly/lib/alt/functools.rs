/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type-checks `functools.partial(...)` bound arguments and synthesizes residual signatures.

use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::unwrap::HintRef;
use crate::error::collector::ErrorCollector;
use crate::types::callable::Callable;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Handle a `functools.partial(func, ...)` call, synthesizing the residual signature instead of
    /// deferring to the typeshed stub.
    pub fn call_functools_partial(
        &self,
        partial_ty: &Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let Type::ClassDef(_) = partial_ty else {
            unreachable!("call_functools_partial dispatched on a non-class callee");
        };
        let Some(CallArg::Arg(target)) = args.first() else {
            // No inspectable callable target (e.g. a `*args` splat); defer to the stub.
            return self.freeform_call_infer(
                partial_ty.clone(),
                args,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            );
        };
        let target_ty = target.infer(self, errors);
        // Fall back to the stub, reusing the already-inferred target so it isn't inferred twice.
        let fallback = |me: &Self| {
            let mut args_with_ty = args.to_vec();
            args_with_ty[0] = CallArg::ty(&target_ty, target.range());
            me.freeform_call_infer(
                partial_ty.clone(),
                &args_with_ty,
                kws,
                callee_range,
                arg_range,
                hint,
                errors,
            )
        };
        // Only a directly-typed function/callable with an ordinary parameter list is handled; others
        // defer.
        let sig = match &target_ty {
            Type::Callable(c) => (**c).clone(),
            Type::Function(f) => f.signature.clone(),
            _ => return fallback(self),
        };
        if !matches!(sig.params, Params::List(_)) {
            return fallback(self);
        }
        // Always type-check the bound arguments against the target's real parameters: making every
        // parameter optional means binding only a prefix doesn't report the rest missing.
        let mut callee = target_ty.clone();
        callee.transform_toplevel_callable(&mut |c: &mut Callable| make_params_optional(c));
        let ret = self.freeform_call_infer(
            callee,
            &args[1..],
            kws,
            target.range(),
            arg_range,
            None,
            errors,
        );
        let residual = match partial_residual(&sig, &args[1..], kws) {
            Some(residual) => residual,
            None => return fallback(self),
        };
        self.heap.mk_callable_from(Callable::partial(residual, ret))
    }
}

/// Make every parameter of a callable optional, so a `functools.partial` construction can bind a
/// prefix of the arguments without the remaining parameters being reported as missing.
fn make_params_optional(callable: &mut Callable) {
    if let Params::List(params) = &mut callable.params {
        for param in params.items_mut() {
            match param {
                Param::PosOnly(_, _, r) | Param::Pos(_, _, r) | Param::KwOnly(_, _, r) => {
                    *r = Required::Optional(None)
                }
                Param::Varargs(..) | Param::Kwargs(..) => {}
            }
        }
    }
}

/// Residual parameters after binding arguments to `callable`. Returns `None` when the
/// target/arguments can't be reduced.
fn partial_residual(
    callable: &Callable,
    bound_args: &[CallArg],
    keywords: &[CallKeyword],
) -> Option<ParamList> {
    let Params::List(params) = &callable.params else {
        return None;
    };
    let mut remaining = params.items().to_vec();
    for arg in bound_args {
        let CallArg::Arg(_) = arg else {
            return None;
        };
        let idx = remaining
            .iter()
            .position(|p| matches!(p, Param::PosOnly(..) | Param::Pos(..) | Param::Varargs(..)))?;
        if !matches!(remaining[idx], Param::Varargs(..)) {
            remaining.remove(idx);
        }
    }
    for kw in keywords {
        let name = &kw.arg?.id;
        let idx = remaining.iter().position(|p| {
            matches!(p, Param::Pos(n, ..) | Param::KwOnly(n, ..) if n == name)
                || matches!(p, Param::Kwargs(..))
        })?;
        if matches!(remaining[idx], Param::Kwargs(..)) {
            continue;
        }
        // A positional can't follow a bound keyword, so demote later positionals to keyword-only.
        if matches!(remaining[idx], Param::Pos(..)) {
            for later in remaining.iter_mut().skip(idx + 1) {
                if let Param::Pos(n, t, r) = later {
                    *later = Param::KwOnly(n.clone(), t.clone(), r.clone());
                }
            }
        }
        let (n, t) = match &remaining[idx] {
            Param::Pos(n, t, _) | Param::KwOnly(n, t, _) => (n.clone(), t.clone()),
            _ => unreachable!("matched a positional or keyword-only parameter"),
        };
        remaining[idx] = Param::KwOnly(n, t, Required::Optional(None));
    }
    Some(ParamList::new(remaining))
}
