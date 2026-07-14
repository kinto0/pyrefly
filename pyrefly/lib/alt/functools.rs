/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type-checks `functools.partial(...)` bound arguments and synthesizes residual signatures.

use std::sync::Arc;

use pyrefly_types::heap::TypeHeap;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Var;
use pyrefly_util::visit::Visit;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

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
use crate::types::types::Forallable;
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
        // We handle a directly-typed function/callable and a generic (`Forall`-wrapped) function.
        // A generic target keeps its `tparams`: type variables the bound args don't pin stay symbolic
        // in the residual and are re-scoped into a `Forall` below, so a partial over a generic
        // function (including decorator use) preserves its genericity instead of leaking a residual
        // through the stub. Class objects, bound methods, overloads with bound args, and unions defer.
        let (tparams, sig) = match &target_ty {
            Type::Callable(c) => (None, (**c).clone()),
            Type::Function(f) => (None, f.signature.clone()),
            Type::Forall(forall) => match &forall.body {
                Forallable::Function(f) => (Some(forall.tparams.clone()), f.signature.clone()),
                Forallable::Callable(c) => (Some(forall.tparams.clone()), c.clone()),
                Forallable::TypeAlias(_) => return fallback(self),
            },
            _ => return fallback(self),
        };
        // Only plain type variables are re-scoped correctly; a `ParamSpec` or `TypeVarTuple` target
        // needs structural residual handling we don't do, so defer it to the stub.
        if let Some(tparams) = &tparams
            && !tparams.iter().all(|q| q.kind() == QuantifiedKind::TypeVar)
        {
            return fallback(self);
        }
        if !matches!(sig.params, Params::List(_)) {
            return fallback(self);
        }
        // Nominal `partial[ret]` fallback for when no residual can be built. For a generic target
        // erase the target's own type vars so they don't leak out of scope; otherwise defer to the stub.
        let nominal_partial = |me: &Self, ret: Type| -> Type {
            match &tparams {
                None => fallback(me),
                Some(tparams) => {
                    let Type::ClassDef(cls) = partial_ty else {
                        unreachable!("call_functools_partial dispatched on a non-class callee");
                    };
                    let mut ret = ret;
                    ret.subst_mut_fn(&mut |q| {
                        tparams
                            .iter()
                            .any(|tp| tp == q)
                            .then(|| q.as_gradual_type())
                    });
                    me.specialize(cls, vec![ret], callee_range, errors)
                }
            }
        };
        // Type-check the bound arguments; making every parameter optional means binding only a prefix
        // doesn't report the remaining parameters as missing. For a generic target we instantiate the
        // type parameters as fresh vars and check against those, so a bound argument can *solve* a
        // typevar (e.g. pin it to an enclosing-scope typevar); the residual is then built from the
        // solved signature, and only the typevars the bound args left unsolved are restored to their
        // original quantified so they can be re-scoped into a `Forall` below.
        let sig = match &tparams {
            None => {
                let mut callee = target_ty.clone();
                callee.transform_toplevel_callable(&mut |c: &mut Callable| make_params_optional(c));
                self.freeform_call_infer(
                    callee,
                    &args[1..],
                    kws,
                    target.range(),
                    arg_range,
                    None,
                    errors,
                );
                sig
            }
            Some(tparams) => {
                let (qs, inst) = self.instantiate_fresh_callable(tparams, sig);
                let var_to_q: SmallMap<Var, Quantified> = qs
                    .vars()
                    .iter()
                    .copied()
                    .zip(tparams.iter().cloned())
                    .collect();
                let mut callee = self.heap.mk_callable_from(inst.clone());
                callee.transform_toplevel_callable(&mut |c: &mut Callable| make_params_optional(c));
                self.freeform_call_infer(
                    callee,
                    &args[1..],
                    kws,
                    target.range(),
                    arg_range,
                    None,
                    errors,
                );
                // A typevar in a *required* residual param stays symbolic so a later call arg can re-solve
                // it (as a direct call would); one only in optional params keeps its solved value (GH #3546).
                let mut regeneric_vars: SmallSet<Var> = SmallSet::new();
                if let Some(residual) = partial_residual(&inst, &args[1..], kws) {
                    for param in residual.items() {
                        let (ty, required) = match param {
                            Param::PosOnly(_, t, r)
                            | Param::Pos(_, t, r)
                            | Param::KwOnly(_, t, r) => (t, r),
                            // `*args`/`**kwargs` carry no `Required` flag, but a later call can always
                            // pass more arguments through them, so their typevars must stay symbolic.
                            Param::Varargs(_, t) | Param::Kwargs(_, t) => (t, &Required::Required),
                        };
                        if !matches!(required, Required::Required) {
                            continue;
                        }
                        for v in ty.collect_all_vars() {
                            if var_to_q.contains_key(&v) {
                                regeneric_vars.insert(v);
                            }
                        }
                    }
                }
                // Restore concretely-pinned residual-parameter typevars to their quantified *before*
                // expanding, so the solution isn't baked into a residual parameter (freezing it too
                // narrowly). `expand_with_bounds` then resolves the remaining vars: a bound argument
                // that pins one substitutes it, while a typevar left unconstrained stays a `Var` and
                // is restored below. Using `expand_with_bounds` (not `finish_quantified`) preserves
                // that distinction; finishing would erase the unconstrained vars to `Any`.
                let mut solved = self.heap.mk_callable_from(inst);
                solved = solved.transform(&mut |t: &mut Type| {
                    if let Type::Var(v) = t
                        && regeneric_vars.contains(v)
                        && let Some(q) = var_to_q.get(v)
                    {
                        *t = self.heap.mk_quantified(q.clone());
                    }
                });
                self.solver().expand_with_bounds(&mut solved);
                let solved = solved.transform(&mut |t: &mut Type| {
                    if let Type::Var(v) = t
                        && let Some(q) = var_to_q.get(v)
                    {
                        *t = self.heap.mk_quantified(q.clone());
                    }
                });
                // Finalize the vars `instantiate_fresh_callable` registered. Their substitution was
                // already applied manually above, and any specialization error was reported by the
                // bound-argument check via `freeform_call_infer`, so dropping the result is safe.
                let _ = self.finish_quantified(qs, false);
                match solved {
                    Type::Callable(c) => *c,
                    _ => unreachable!("built by mk_callable_from"),
                }
            }
        };
        // The arguments can't be reduced to a residual (e.g. too many bound positionals); hand
        // back the nominal `partial[ret]` rather than re-running the stub over a `Forall`.
        let Some(residual) = partial_residual(&sig, &args[1..], kws) else {
            return nominal_partial(self, sig.ret);
        };
        let callable = Callable::partial(residual, sig.ret);
        match tparams {
            None => self.heap.mk_callable_from(callable),
            Some(tparams) => restore_partial_generics(self.heap, callable, &tparams),
        }
    }
}

/// Re-scope a partial residual over the target's still-used type parameters. After binding a prefix
/// of a generic function's arguments, the type variables the bound args didn't pin remain in the
/// residual signature; wrap the result in a `Forall` over exactly those, so calling the residual
/// (e.g. when it is applied as a decorator) instantiates them afresh. Mirrors the decorator path's
/// `restore_decoratee_generics`.
fn restore_partial_generics(heap: &TypeHeap, callable: Callable, tparams: &TParams) -> Type {
    let mut used: SmallSet<Quantified> = SmallSet::new();
    callable.visit(&mut |ty: &Type| {
        if let Type::Quantified(q) = ty {
            used.insert((**q).clone());
        }
    });
    let surviving: Vec<Quantified> = tparams
        .iter()
        .filter(|q| used.contains(*q))
        .cloned()
        .collect();
    if surviving.is_empty() {
        return heap.mk_callable_from(callable);
    }
    Forallable::Callable(callable).forall(Arc::new(TParams::new(surviving)))
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
