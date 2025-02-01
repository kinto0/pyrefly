/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use ruff_python_ast::name::Name;
use ruff_python_ast::Arguments;
use ruff_python_ast::BoolOp;
use ruff_python_ast::Comprehension;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprNoneLiteral;
use ruff_python_ast::ExprSlice;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::UnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers::UNKNOWN;
use crate::alt::callable::CallArg;
use crate::ast::Ast;
use crate::binding::binding::Key;
use crate::dunder;
use crate::module::short_identifier::ShortIdentifier;
use crate::types::callable::Callable;
use crate::types::callable::CallableKind;
use crate::types::callable::DataclassKeywords;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::ClassKind;
use crate::types::class::ClassType;
use crate::types::literal::Lit;
use crate::types::param_spec::ParamSpec;
use crate::types::special_form::SpecialForm;
use crate::types::tuple::Tuple;
use crate::types::type_var::Restriction;
use crate::types::type_var::TypeVar;
use crate::types::type_var::TypeVarArgs;
use crate::types::type_var::Variance;
use crate::types::type_var_tuple::TypeVarTuple;
use crate::types::types::AnyStyle;
use crate::types::types::CalleeKind;
use crate::types::types::Decoration;
use crate::types::types::Type;
use crate::types::types::Var;
use crate::util::prelude::SliceExt;

enum CallStyle<'a> {
    Method(&'a Name),
    BinaryOp(Operator),
    FreeForm,
}

/// A thing that can be called (see as_call_target and call_infer).
/// Note that a single "call" may invoke multiple functions under the hood,
/// e.g., `__new__` followed by `__init__` for Class.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum CallTarget {
    /// A thing whose type is a Callable, usually a function.
    Callable(Callable),
    /// The dataclasses.dataclass function.
    Dataclass(Callable),
    /// Method of a class. The `Type` is the self/cls argument.
    BoundMethod(Type, Callable),
    /// A class object.
    Class(ClassType),
}

impl CallTarget {
    pub fn any(style: AnyStyle) -> Self {
        Self::Callable(Callable {
            params: Params::Ellipsis,
            ret: style.propagate(),
        })
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    // Helper method for inferring the type of a boolean operation over a sequence of values.
    fn boolop(&self, values: &[Expr], op: BoolOp) -> Type {
        let target = match op {
            BoolOp::And => false,
            BoolOp::Or => true,
        };
        let should_shortcircuit = |t: &Type| t.as_bool() == Some(target);
        let should_discard = |t: &Type| t.as_bool() == Some(!target);

        let mut types = Vec::new();
        let last_index = values.len() - 1;
        for (i, value) in values.iter().enumerate() {
            let t = self.expr_infer(value);
            if should_shortcircuit(&t) {
                types.push(t);
                break;
            }
            // If we reach the last value, we should always keep it.
            if i != last_index && should_discard(&t) {
                continue;
            }
            match t {
                Type::Union(options) => {
                    for option in options {
                        if !should_discard(&option) {
                            types.push(option);
                        }
                    }
                }
                _ => types.push(t),
            }
        }
        self.unions(&types)
    }

    fn error_call_target(&self, range: TextRange, msg: String) -> (Vec<Var>, CallTarget) {
        self.errors().add(self.module_info(), range, msg);
        (Vec::new(), CallTarget::any(AnyStyle::Error))
    }

    /// Return a pair of the quantified variables I had to instantiate, and the resulting call target.
    fn as_call_target(&self, ty: Type) -> Option<(Vec<Var>, CallTarget)> {
        match ty {
            Type::Callable(c, CallableKind::Dataclass(_)) => {
                Some((Vec::new(), CallTarget::Dataclass(*c)))
            }
            Type::Callable(c, _) => Some((Vec::new(), CallTarget::Callable(*c))),
            Type::BoundMethod(obj, func) => match self.as_call_target(*func.clone()) {
                Some((gs, CallTarget::Callable(c))) => Some((gs, CallTarget::BoundMethod(*obj, c))),
                _ => None,
            },
            Type::ClassDef(cls) => self.as_call_target(self.instantiate_fresh(&cls)),
            Type::Type(box Type::ClassType(cls)) => Some((Vec::new(), CallTarget::Class(cls))),
            Type::Forall(params, t) => {
                let (mut qs, t) = self.solver().fresh_quantified(
                    params.quantified().collect::<Vec<_>>().as_slice(),
                    *t,
                    self.uniques,
                );
                self.as_call_target(t).map(|(qs2, x)| {
                    qs.extend(qs2);
                    (qs, x)
                })
            }
            Type::Var(v) if let Some(_guard) = self.recurser.recurse(v) => {
                self.as_call_target(self.solver().force_var(v))
            }
            Type::Union(xs) => {
                let res = xs
                    .into_iter()
                    .map(|x| self.as_call_target(x))
                    .collect::<Option<SmallSet<_>>>()?;
                if res.len() == 1 {
                    Some(res.into_iter().next().unwrap())
                } else {
                    None
                }
            }
            Type::Any(style) => Some((Vec::new(), CallTarget::any(style))),
            Type::TypeAlias(ta) => self.as_call_target(ta.as_value(self.stdlib)),
            Type::ClassType(cls) => self
                .get_instance_attribute(&cls, &dunder::CALL)
                .and_then(|attr| self.resolve_as_instance_method(attr))
                .and_then(|ty| self.as_call_target(ty)),
            Type::Type(box Type::TypedDict(typed_dict)) => {
                Some((Vec::new(), CallTarget::Callable(typed_dict.as_callable())))
            }
            _ => None,
        }
    }

    fn as_call_target_or_error(
        &self,
        ty: Type,
        call_style: CallStyle,
        range: TextRange,
    ) -> (Vec<Var>, CallTarget) {
        match self.as_call_target(ty.clone()) {
            Some(target) => target,
            None => {
                let expect_message = match call_style {
                    CallStyle::Method(method) => {
                        format!("Expected `{}` to be a callable", method)
                    }
                    CallStyle::BinaryOp(op) => {
                        format!("Expected `{}` to be a callable", op.dunder())
                    }
                    CallStyle::FreeForm => "Expected a callable".to_owned(),
                };
                self.error_call_target(
                    range,
                    format!("{}, got {}", expect_message, ty.deterministic_printing()),
                )
            }
        }
    }

    /// Calls a method. If no attribute exists with the given method name, returns None without attempting the call.
    pub fn call_method(
        &self,
        ty: &Type,
        method_name: &Name,
        range: TextRange,
        args: &[CallArg],
        keywords: &[Keyword],
    ) -> Option<Type> {
        let callee_ty =
            self.type_of_attr_get_if_found(ty.clone(), method_name, range, "Expr::call_method")?;
        let call_target =
            self.as_call_target_or_error(callee_ty, CallStyle::Method(method_name), range);
        Some(self.call_infer(call_target, args, keywords, range))
    }

    /// Calls a method. If no attribute exists with the given method name, logs an error and calls the method with
    /// an assumed type of Callable[..., Any].
    pub fn call_method_or_error(
        &self,
        ty: &Type,
        method_name: &Name,
        range: TextRange,
        args: &[CallArg],
        keywords: &[Keyword],
    ) -> Type {
        if let Some(ret) = self.call_method(ty, method_name, range, args, keywords) {
            ret
        } else {
            self.call_infer(
                self.error_call_target(range, format!("`{ty}` has no attribute `{method_name}`")),
                args,
                keywords,
                range,
            )
        }
    }

    /// If the metaclass defines a custom `__call__`, call it. If the `__call__` comes from `type`, ignore
    /// it because `type.__call__` behavior is baked into our constructor logic.
    pub fn call_metaclass(
        &self,
        cls: &ClassType,
        range: TextRange,
        args: &[CallArg],
        keywords: &[Keyword],
    ) -> Option<Type> {
        let dunder_call = match self.get_metaclass_dunder_call(cls)? {
            Type::BoundMethod(_, func) => {
                // This method was bound to a general instance of the metaclass, but we have more
                // information about the particular instance that it should be bound to.
                Type::BoundMethod(
                    Box::new(Type::type_form(Type::ClassType(cls.clone()))),
                    func,
                )
            }
            dunder_call => dunder_call,
        };
        Some(self.call_infer(
            self.as_call_target_or_error(dunder_call, CallStyle::Method(&dunder::CALL), range),
            args,
            keywords,
            range,
        ))
    }

    pub fn expr(&self, x: &Expr, check: Option<&Type>) -> Type {
        match check {
            Some(want) if !want.is_any() => {
                let got = self.expr_infer_with_hint(x, Some(want));
                self.check_type(want, &got, x.range())
            }
            _ => self.expr_infer(x),
        }
    }

    /// Infers types for `if` clauses in the given comprehensions.
    /// This is for error detection only; the types are not used.
    fn ifs_infer(&self, comps: &[Comprehension]) {
        for comp in comps.iter() {
            for if_clause in comp.ifs.iter() {
                self.expr_infer(if_clause);
            }
        }
    }

    fn construct(
        &self,
        cls: ClassType,
        args: &[CallArg],
        keywords: &[Keyword],
        range: TextRange,
    ) -> Type {
        // Based on https://typing.readthedocs.io/en/latest/spec/constructors.html.
        let instance_ty = Type::ClassType(cls.clone());
        if let Some(ret) = self.call_metaclass(&cls, range, args, keywords)
            && !self
                .solver()
                .is_subset_eq(&ret, &instance_ty, self.type_order())
        {
            // Got something other than an instance of the class under construction.
            return ret;
        }
        let overrides_new = if let Some(new_method) = self.get_dunder_new(&cls) {
            let cls_ty = Type::type_form(instance_ty.clone());
            let mut full_args = vec![CallArg::Type(&cls_ty, range)];
            full_args.extend_from_slice(args);
            let ret = self.call_infer(
                self.as_call_target_or_error(new_method, CallStyle::Method(&dunder::NEW), range),
                &full_args,
                keywords,
                range,
            );
            if !self
                .solver()
                .is_subset_eq(&ret, &instance_ty, self.type_order())
            {
                // Got something other than an instance of the class under construction.
                return ret;
            }
            true
        } else {
            false
        };
        if let Some(init_method) = self.get_dunder_init(&cls, overrides_new) {
            self.call_infer(
                self.as_call_target_or_error(init_method, CallStyle::Method(&dunder::INIT), range),
                args,
                keywords,
                range,
            );
        }
        cls.self_type()
    }

    fn call_infer(
        &self,
        call_target: (Vec<Var>, CallTarget),
        args: &[CallArg],
        keywords: &[Keyword],
        range: TextRange,
    ) -> Type {
        let is_dataclass = matches!(call_target.1, CallTarget::Dataclass(_));
        let res = match call_target.1 {
            CallTarget::Class(cls) => self.construct(cls, args, keywords, range),
            CallTarget::BoundMethod(obj, c) => {
                let first_arg = CallArg::Type(&obj, range);
                self.callable_infer(c, Some(first_arg), args, keywords, range)
            }
            CallTarget::Callable(callable) | CallTarget::Dataclass(callable) => {
                self.callable_infer(callable, None, args, keywords, range)
            }
        };
        self.solver().finish_quantified(&call_target.0);
        if is_dataclass && let Type::Callable(c, _) = res {
            let mut kws = DataclassKeywords::default();
            for kw in keywords {
                kws.set_keyword(kw.arg.as_ref(), self.expr_infer(&kw.value));
            }
            Type::Callable(c, CallableKind::Dataclass(kws))
        } else {
            res
        }
    }

    pub fn attr_infer(&self, obj: &Type, attr_name: &Name, range: TextRange) -> Type {
        self.distribute_over_union(obj, |obj| {
            self.type_of_attr_get(obj.clone(), attr_name, range, "Expr::attr_infer")
        })
    }

    fn binop_infer(&self, x: &ExprBinOp) -> Type {
        let binop_call = |op: Operator, lhs: &Type, rhs: Type, range: TextRange| -> Type {
            // TODO(yangdanny): handle reflected dunder methods
            let method_type = self.attr_infer(lhs, &Name::new(op.dunder()), range);
            let callable =
                self.as_call_target_or_error(method_type, CallStyle::BinaryOp(op), range);
            self.call_infer(callable, &[CallArg::Type(&rhs, range)], &[], range)
        };
        let lhs = self.expr_infer(&x.left);
        let rhs = self.expr_infer(&x.right);
        if let Type::Any(style) = &lhs {
            return style.propagate();
        } else if x.op == Operator::BitOr
            && let Some(l) = self.untype_opt(lhs.clone(), x.left.range())
            && let Some(r) = self.untype_opt(rhs.clone(), x.right.range())
        {
            return Type::type_form(self.union(&l, &r));
        }
        self.distribute_over_union(&lhs, |lhs| binop_call(x.op, lhs, rhs.clone(), x.range))
    }

    /// When interpreted as static types (as opposed to when accounting for runtime
    /// behavior when used as values), `Type::ClassDef(cls)` is equivalent to
    /// `Type::Type(box Type::ClassType(cls, default_targs(cls)))` where `default_targs(cls)`
    /// is the result of looking up the class `tparams` and synthesizing default `targs` that
    /// are gradual if needed (e.g. `list` is treated as `list[Any]` when used as an annotation).
    ///
    /// This function canonicalizes to `Type::ClassType` or `Type::TypedDict`
    pub fn canonicalize_all_class_types(&self, ty: Type, range: TextRange) -> Type {
        ty.transform(|ty| match ty {
            Type::ClassDef(cls) => {
                *ty = Type::type_form(self.promote(cls, range));
            }
            _ => {}
        })
    }

    fn literal_bool_infer(&self, x: &Expr) -> bool {
        let ty = self.expr_infer(x);
        match ty {
            Type::Literal(Lit::Bool(b)) => b,
            _ => {
                self.error(
                    x.range(),
                    format!("Expected literal True or False, got {ty}"),
                );
                false
            }
        }
    }

    fn tyvar_from_arguments(&self, arguments: &Arguments) -> TypeVar {
        let args = TypeVarArgs::from_arguments(arguments);

        let name = match args.name {
            Some(Expr::StringLiteral(x)) => Identifier::new(Name::new(x.value.to_str()), x.range),
            _ => {
                let msg = if args.name.is_none() {
                    "Missing `name` argument to TypeVar"
                } else {
                    "Expected first argument of TypeVar to be a string literal"
                };
                self.error(arguments.range, msg.to_owned());
                // FIXME: This isn't ideal - we are creating a fake Identifier, which is not good.
                Identifier::new(UNKNOWN, arguments.range)
            }
        };
        let constraints = args.constraints.map(|x| self.expr_untype(x));
        let bound = args.bound.map(|x| self.expr_untype(x));
        let default = args.default.map(|x| self.expr_untype(x));
        let covariant = args.covariant.is_some_and(|x| self.literal_bool_infer(x));
        let contravariant = args
            .contravariant
            .is_some_and(|x| self.literal_bool_infer(x));
        let infer_variance = args
            .infer_variance
            .is_some_and(|x| self.literal_bool_infer(x));

        for kw in args.unknown {
            self.error(
                kw.range,
                match &kw.arg {
                    Some(id) => format!("Unexpected keyword argument `{}` to TypeVar", id.id),
                    None => "Cannot pass unpacked keyword arguments to TypeVar".to_owned(),
                },
            );
        }
        let restriction = if let Some(bound) = bound {
            if !constraints.is_empty() {
                self.error(
                    arguments.range,
                    "TypeVar cannot have both constraints and bound".to_owned(),
                );
            }
            Restriction::Bound(bound)
        } else if !constraints.is_empty() {
            Restriction::Constraints(constraints)
        } else {
            Restriction::Unrestricted
        };
        if [covariant, contravariant, infer_variance]
            .iter()
            .filter(|x| **x)
            .count()
            > 1
        {
            self.error(
                arguments.range,
                "Contradictory variance specifications".to_owned(),
            );
        }
        let variance = if covariant {
            Some(Variance::Covariant)
        } else if contravariant {
            Some(Variance::Contravariant)
        } else if infer_variance {
            None
        } else {
            Some(Variance::Invariant)
        };
        TypeVar::new(
            name,
            self.module_info().dupe(),
            restriction,
            default,
            variance,
        )
    }

    pub fn expr_infer(&self, x: &Expr) -> Type {
        self.expr_infer_with_hint(x, None)
    }

    fn yield_expr(&self, x: Expr) -> Expr {
        match x {
            Expr::Yield(x) => match x.value {
                Some(x) => *x,
                None => Expr::NoneLiteral(ExprNoneLiteral { range: x.range() }),
            },
            // This case should be unreachable.
            _ => unreachable!("yield or yield from expression expected"),
        }
    }

    /// Apply a decorator. This effectively synthesizes a function call.
    pub fn apply_decorator(&self, decorator: &Decorator, decoratee: Type) -> Type {
        if matches!(&decoratee, Type::ClassDef(cls) if cls.has_qname("typing", "TypeVar")) {
            // Avoid recursion in TypeVar, which is decorated with `@final`, whose type signature
            // itself depends on a TypeVar.
            return decoratee;
        }
        let ty_decorator = self.expr(&decorator.expression, None);
        match ty_decorator.callee_kind() {
            Some(CalleeKind::Class(ClassKind::StaticMethod)) => {
                return Type::Decoration(Decoration::StaticMethod(Box::new(decoratee)));
            }
            Some(CalleeKind::Class(ClassKind::ClassMethod)) => {
                return Type::Decoration(Decoration::ClassMethod(Box::new(decoratee)));
            }
            Some(CalleeKind::Class(ClassKind::Property)) => {
                return Type::Decoration(Decoration::Property(Box::new((decoratee, None))));
            }
            _ => {}
        }
        match ty_decorator {
            Type::Decoration(Decoration::PropertySetterDecorator(getter)) => {
                return Type::Decoration(Decoration::Property(Box::new((
                    *getter,
                    Some(decoratee),
                ))));
            }
            _ => {}
        }
        if matches!(&decoratee, Type::ClassDef(_)) {
            // TODO: don't blanket ignore class decorators.
            return decoratee;
        }
        let call_target =
            { self.as_call_target_or_error(ty_decorator, CallStyle::FreeForm, decorator.range) };
        let arg = CallArg::Type(&decoratee, decorator.range);
        self.call_infer(call_target, &[arg], &[], decorator.range)
    }

    /// Helper function hide details of call synthesis from the attribute resolution code.
    pub fn call_property_getter(&self, getter_method: Type, range: TextRange) -> Type {
        let call_target = self.as_call_target_or_error(getter_method, CallStyle::FreeForm, range);
        self.call_infer(call_target, &[], &[], range)
    }

    /// Helper function hide details of call synthesis from the attribute resolution code.
    pub fn call_property_setter(
        &self,
        setter_method: Type,
        got: CallArg,
        range: TextRange,
    ) -> Type {
        let call_target = self.as_call_target_or_error(setter_method, CallStyle::FreeForm, range);
        self.call_infer(call_target, &[got], &[], range)
    }

    fn expr_infer_with_hint(&self, x: &Expr, hint: Option<&Type>) -> Type {
        match x {
            Expr::BoolOp(x) => self.boolop(&x.values, x.op),
            Expr::Named(x) => self.expr_infer_with_hint(&x.value, hint),
            Expr::BinOp(x) => self.binop_infer(x),
            Expr::UnaryOp(x) => {
                let t = self.expr_infer(&x.operand);
                let unop = |t: &Type, f: &dyn Fn(&Lit) -> Type, method: &Name| match t {
                    Type::Literal(lit) => f(lit),
                    Type::ClassType(_) => self.call_method_or_error(t, method, x.range, &[], &[]),
                    _ => self.error(
                        x.range,
                        format!("Unary {} is not supported on {}", x.op.as_str(), t),
                    ),
                };
                self.distribute_over_union(&t, |t| match x.op {
                    UnaryOp::USub => {
                        let f = |lit: &Lit| {
                            lit.negate(self.stdlib, self.module_info(), x.range, self.errors())
                        };
                        unop(t, &f, &dunder::NEG)
                    }
                    UnaryOp::UAdd => {
                        let f = |lit: &Lit| lit.clone().to_type();
                        unop(t, &f, &dunder::POS)
                    }
                    UnaryOp::Not => match t.as_bool() {
                        None => self.stdlib.bool().to_type(),
                        Some(b) => Type::Literal(Lit::Bool(!b)),
                    },
                    UnaryOp::Invert => {
                        let f = |lit: &Lit| lit.invert(self.module_info(), x.range, self.errors());
                        unop(t, &f, &dunder::INVERT)
                    }
                })
            }
            Expr::Lambda(lambda) => {
                let hint_callable = hint.and_then(|ty| {
                    if let Some((_, CallTarget::Callable(c))) = self.as_call_target(ty.clone()) {
                        Some(c)
                    } else {
                        None
                    }
                });
                let parameters: Vec<Param> = if let Some(parameters) = &lambda.parameters {
                    // TODO: use callable hint parameters to check body
                    (**parameters)
                        .iter()
                        .map(|p| {
                            let ty = self
                                .get(&Key::Definition(ShortIdentifier::new(p.name())))
                                .arc_clone();
                            Param::Pos(p.name().clone().id, ty, Required::Required)
                        })
                        .collect()
                } else {
                    vec![]
                };
                if let Some(callable) = hint_callable {
                    let inferred_callable = Type::Callable(
                        Box::new(Callable {
                            params: Params::List(ParamList::new(parameters)),
                            ret: self.expr(&lambda.body, Some(&callable.ret)),
                        }),
                        CallableKind::Anon,
                    );
                    let wanted_callable = Type::Callable(Box::new(callable), CallableKind::Anon);
                    self.check_type(&wanted_callable, &inferred_callable, x.range());
                    wanted_callable
                } else {
                    Type::Callable(
                        Box::new(Callable {
                            params: Params::List(ParamList::new(parameters)),
                            ret: self.expr_infer(&lambda.body),
                        }),
                        CallableKind::Anon,
                    )
                }
            }
            Expr::If(x) => {
                // TODO: Support type refinement
                let condition_type = self.expr_infer(&x.test);
                if let Some(hint_ty) = hint {
                    let mut body_hint = hint;
                    let mut orelse_hint = hint;
                    match condition_type.as_bool() {
                        Some(true) => orelse_hint = None,
                        Some(false) => body_hint = None,
                        None => {}
                    }
                    self.expr(&x.body, body_hint);
                    self.expr(&x.orelse, orelse_hint);
                    hint_ty.clone()
                } else {
                    let body_type = self.expr_infer(&x.body);
                    let orelse_type = self.expr_infer(&x.orelse);
                    match condition_type.as_bool() {
                        Some(true) => body_type,
                        Some(false) => orelse_type,
                        None => self.union(&body_type, &orelse_type),
                    }
                }
            }
            Expr::Tuple(x) => {
                let ts = match hint {
                    Some(Type::Tuple(Tuple::Concrete(elts))) if elts.len() == x.elts.len() => elts,
                    Some(ty) => match self.decompose_tuple(ty) {
                        Some(elem_ty) => &vec![elem_ty; x.elts.len()],
                        None => &Vec::new(),
                    },
                    None => &Vec::new(),
                };
                Type::tuple(
                    x.elts
                        .iter()
                        .enumerate()
                        .map(|(i, x)| self.expr(x, ts.get(i)))
                        .collect(),
                )
            }
            Expr::List(x) => {
                let hint = hint.and_then(|ty| self.decompose_list(ty));
                if let Some(hint) = hint {
                    x.elts.iter().for_each(|x| {
                        self.expr(x, Some(&hint));
                    });
                    self.stdlib.list(hint).to_type()
                } else if x.is_empty() {
                    let elem_ty = self.solver().fresh_contained(self.uniques).to_type();
                    self.stdlib.list(elem_ty).to_type()
                } else {
                    let tys = x
                        .elts
                        .map(|x| self.expr_infer(x).promote_literals(self.stdlib));
                    self.stdlib.list(self.unions(&tys)).to_type()
                }
            }
            Expr::Dict(x) => {
                let unwrapped_hint = hint.and_then(|ty| self.decompose_dict(ty));
                let flattened_items = Ast::flatten_dict_items(&x.items);
                if let Some(hint @ Type::TypedDict(typed_dict)) = hint {
                    let fields = typed_dict.fields();
                    let mut has_expansion = false;
                    let mut keys: SmallSet<Name> = SmallSet::new();
                    flattened_items.iter().for_each(|x| match &x.key {
                        Some(key) => {
                            let key_type = self.expr(key, None);
                            if let Type::Literal(Lit::String(name)) = key_type {
                                let key_name = Name::new(name.clone());
                                if let Some(field) = fields.get(&key_name) {
                                    self.expr(&x.value, Some(&field.ty));
                                } else {
                                    self.error(
                                        key.range(),
                                        format!(
                                            "Key `{}` is not defined in TypedDict `{}`",
                                            name,
                                            typed_dict.name()
                                        ),
                                    );
                                }
                                keys.insert(key_name);
                            } else {
                                self.error(
                                    key.range(),
                                    format!("Expected string literal key, got `{}`", key_type),
                                );
                            }
                        }
                        None => {
                            has_expansion = true;
                            self.expr(&x.value, Some(hint));
                        }
                    });
                    if !has_expansion {
                        fields.iter().for_each(|(key, field)| {
                            if field.required && !keys.contains(key) {
                                self.error(
                                    x.range,
                                    format!(
                                        "Missing required key `{}` for TypedDict `{}`",
                                        key,
                                        typed_dict.name()
                                    ),
                                );
                            }
                        });
                    }
                    hint.clone()
                } else if let Some(hint) = unwrapped_hint {
                    flattened_items.iter().for_each(|x| match &x.key {
                        Some(key) => {
                            self.expr(key, Some(&hint.key));
                            self.expr(&x.value, Some(&hint.value));
                        }
                        None => {
                            self.expr(
                                &x.value,
                                Some(
                                    &self
                                        .stdlib
                                        .dict(hint.key.clone(), hint.value.clone())
                                        .to_type(),
                                ),
                            );
                        }
                    });
                    hint.to_type(self.stdlib)
                } else if x.is_empty() {
                    let key_ty = self.solver().fresh_contained(self.uniques).to_type();
                    let value_ty = self.solver().fresh_contained(self.uniques).to_type();
                    self.stdlib.dict(key_ty, value_ty).to_type()
                } else {
                    let mut key_tys = Vec::new();
                    let mut value_tys = Vec::new();
                    flattened_items.iter().for_each(|x| match &x.key {
                        Some(key) => {
                            let key_t = self.expr_infer(key).promote_literals(self.stdlib);
                            let value_t = self.expr_infer(&x.value).promote_literals(self.stdlib);
                            key_tys.push(key_t);
                            value_tys.push(value_t);
                        }
                        None => {
                            let value_t = self.expr(&x.value, None);
                            if let Some(unwrapped) = self.decompose_dict(&value_t) {
                                key_tys.push(unwrapped.key);
                                value_tys.push(unwrapped.value);
                            } else {
                                self.error(
                                    x.value.range(),
                                    format!("Expected a dict, got {}", value_t),
                                );
                            }
                        }
                    });
                    let key_ty = self.unions(&key_tys);
                    let value_ty = self.unions(&value_tys);
                    self.stdlib.dict(key_ty, value_ty).to_type()
                }
            }
            Expr::Set(x) => {
                let hint = hint.and_then(|ty| self.decompose_set(ty));
                if let Some(hint) = hint {
                    x.elts.iter().for_each(|x| {
                        self.expr(x, Some(&hint));
                    });
                    self.stdlib.set(hint).to_type()
                } else if x.is_empty() {
                    let elem_ty = self.solver().fresh_contained(self.uniques).to_type();
                    self.stdlib.set(elem_ty).to_type()
                } else {
                    let tys = x
                        .elts
                        .map(|x| self.expr_infer(x).promote_literals(self.stdlib));
                    self.stdlib.set(self.unions(&tys)).to_type()
                }
            }
            Expr::ListComp(x) => {
                let hint = hint.and_then(|ty| self.decompose_list(ty));
                self.ifs_infer(&x.generators);
                if let Some(hint) = hint {
                    self.expr(&x.elt, Some(&hint));
                    self.stdlib.list(hint).to_type()
                } else {
                    let elem_ty = self.expr_infer(&x.elt).promote_literals(self.stdlib);
                    self.stdlib.list(elem_ty).to_type()
                }
            }
            Expr::SetComp(x) => {
                let hint = hint.and_then(|ty| self.decompose_set(ty));
                self.ifs_infer(&x.generators);
                if let Some(hint) = hint {
                    self.expr(&x.elt, Some(&hint));
                    self.stdlib.set(hint).to_type()
                } else {
                    let elem_ty = self.expr_infer(&x.elt).promote_literals(self.stdlib);
                    self.stdlib.set(elem_ty).to_type()
                }
            }
            Expr::DictComp(x) => {
                let hint = hint.and_then(|ty| self.decompose_dict(ty));
                self.ifs_infer(&x.generators);
                if let Some(hint) = hint {
                    self.expr(&x.key, Some(&hint.key));
                    self.expr(&x.value, Some(&hint.value));
                    hint.to_type(self.stdlib)
                } else {
                    let key_ty = self.expr_infer(&x.key).promote_literals(self.stdlib);
                    let value_ty = self.expr_infer(&x.value).promote_literals(self.stdlib);
                    self.stdlib.dict(key_ty, value_ty).to_type()
                }
            }
            Expr::Generator(x) => {
                self.ifs_infer(&x.generators);
                let yield_ty = self.expr_infer(&x.elt);
                self.stdlib
                    .generator(yield_ty, Type::None, Type::None)
                    .to_type()
            }
            Expr::Await(x) => {
                let awaiting_ty = self.expr_infer(&x.value);
                match self.unwrap_awaitable(&awaiting_ty) {
                    Some(ty) => ty,
                    None => self.error(x.range, "Expression is not awaitable".to_owned()),
                }
            }
            Expr::Yield(_) => {
                let yield_expr_type = &self
                    .get(&Key::YieldTypeOfYieldAnnotation(x.clone().range()))
                    .arc_clone();
                let yield_value = self.yield_expr(x.clone());
                let inferred_expr_type = &self.expr_infer(&yield_value);

                self.check_type(yield_expr_type, inferred_expr_type, x.clone().range());

                self.get(&Key::SendTypeOfYieldAnnotation(x.range()))
                    .arc_clone()
            }
            Expr::YieldFrom(y) => {
                let inferred_expr_type = &self.expr_infer(&y.value);

                let final_generator_type = &self
                    .get(&Key::TypeOfYieldAnnotation(x.clone().range()))
                    .arc_clone();

                self.check_type(final_generator_type, inferred_expr_type, x.clone().range());

                self.get(&Key::ReturnTypeOfYieldAnnotation(x.range()))
                    .arc_clone()
            }
            Expr::Compare(x) => {
                let _ty = self.expr_infer(&x.left);
                let _tys = x.comparators.map(|x| self.expr_infer(x));
                // We don't actually check that comparing these types is sensible, which matches Pyright
                self.stdlib.bool().to_type()
            }
            Expr::Call(x) if is_special_name(&x.func, "assert_type") => {
                if x.arguments.args.len() == 2 {
                    let expr_a = &x.arguments.args[0];
                    let expr_b = &x.arguments.args[1];
                    let a = self.expr_infer(expr_a);
                    let b = self.expr_untype(expr_b);
                    let a = self.canonicalize_all_class_types(
                        self.solver().deep_force(a).explicit_any().anon_callables(),
                        expr_a.range(),
                    );
                    let b = self.canonicalize_all_class_types(
                        self.solver().deep_force(b).explicit_any().anon_callables(),
                        expr_b.range(),
                    );
                    if a != b {
                        self.error(
                            x.range,
                            format!(
                                "assert_type({}, {}) failed",
                                a.deterministic_printing(),
                                b.deterministic_printing()
                            ),
                        );
                    }
                } else {
                    self.error(
                        x.range,
                        format!(
                            "assert_type needs 2 arguments, got {:#?}",
                            x.arguments.args.len()
                        ),
                    );
                }
                Type::None
            }
            Expr::Call(x) if is_special_name(&x.func, "reveal_type") => {
                if x.arguments.args.len() == 1 {
                    let t = self
                        .solver()
                        .deep_force(self.expr_infer(&x.arguments.args[0]));
                    self.error(
                        x.range,
                        format!("revealed type: {}", t.deterministic_printing()),
                    );
                } else {
                    self.error(
                        x.range,
                        format!(
                            "reveal_type needs 1 argument, got {}",
                            x.arguments.args.len()
                        ),
                    );
                }
                Type::None
            }
            Expr::Call(x) => {
                let ty_fun = self.expr_infer(&x.func);
                if TypeVar::is_ctor(&ty_fun) {
                    Type::type_form(self.tyvar_from_arguments(&x.arguments).to_type())
                } else if TypeVarTuple::is_ctor(&ty_fun)
                    && let Some(name) = arguments_one_string(&x.arguments)
                {
                    Type::type_form(TypeVarTuple::new(name, self.module_info().dupe()).to_type())
                } else if ParamSpec::is_ctor(&ty_fun)
                    && let Some(name) = arguments_one_string(&x.arguments)
                {
                    Type::type_form(ParamSpec::new(name, self.module_info().dupe()).to_type())
                } else {
                    let func_range = x.func.range();
                    let args = x.arguments.args.map(|arg| match arg {
                        Expr::Starred(x) => CallArg::Star(&x.value, x.range),
                        _ => CallArg::Expr(arg),
                    });
                    self.distribute_over_union(&ty_fun, |ty| {
                        let callable = self.as_call_target_or_error(
                            ty.clone(),
                            CallStyle::FreeForm,
                            func_range,
                        );
                        self.call_infer(callable, &args, &x.arguments.keywords, func_range)
                    })
                }
            }
            Expr::FString(x) => match Lit::from_fstring(x) {
                Some(lit) => lit.to_type(),
                _ => self.stdlib.str().to_type(),
            },
            Expr::StringLiteral(x) => Lit::from_string_literal(x).to_type(),
            Expr::BytesLiteral(x) => Lit::from_bytes_literal(x).to_type(),
            Expr::NumberLiteral(x) => match &x.value {
                Number::Int(x) => match Lit::from_int(x) {
                    Some(lit) => lit.to_type(),
                    None => self.stdlib.int().to_type(),
                },
                Number::Float(_) => self.stdlib.float().to_type(),
                Number::Complex { .. } => self.stdlib.complex().to_type(),
            },
            Expr::BooleanLiteral(x) => Lit::from_boolean_literal(x).to_type(),
            Expr::NoneLiteral(_) => Type::None,
            Expr::EllipsisLiteral(_) => Type::Ellipsis,
            Expr::Attribute(x) => {
                let obj = self.expr_infer(&x.value);
                match (&obj, x.attr.id.as_str()) {
                    (Type::Literal(Lit::Enum(box (_, member))), "_name_") => {
                        Type::Literal(Lit::String(member.as_str().into()))
                    }
                    _ => self.attr_infer(&obj, &x.attr.id, x.range),
                }
            }
            Expr::Subscript(x) => {
                let xs = Ast::unpack_slice(&x.slice);
                // FIXME: We don't deal properly with hint here, we should.
                let mut fun = self.expr_infer(&x.value);
                if let Type::Var(v) = fun {
                    fun = self.solver().force_var(v);
                }
                if matches!(&fun, Type::ClassDef(t) if t.name() == "tuple") {
                    fun = Type::type_form(Type::SpecialForm(SpecialForm::Tuple));
                }
                match fun {
                    Type::Forall(params, ty) => {
                        let param_map = params
                            .quantified()
                            .zip(xs.map(|x| self.expr_untype(x)))
                            .collect::<SmallMap<_, _>>();
                        ty.subst(&param_map)
                    }
                    // Note that we have to check for `builtins.type` by name here because this code runs
                    // when we're bootstrapping the stdlib and don't have access to class objects yet.
                    Type::ClassDef(cls) if cls.has_qname("builtins", "type") => {
                        let targ = match xs.len() {
                            // This causes us to treat `type[list]` as equivalent to `type[list[Any]]`,
                            // which may or may not be what we want.
                            1 => self.expr_untype(&xs[0]),
                            _ => {
                                self.error(
                                    x.range,
                                    format!(
                                        "Expected 1 type argument for class `type`, got {}",
                                        xs.len()
                                    ),
                                );
                                Type::any_error()
                            }
                        };
                        // TODO: Validate that `targ` refers to a "valid in-scope class or TypeVar"
                        // (https://typing.readthedocs.io/en/latest/spec/annotations.html#type-and-annotation-expressions)
                        Type::type_form(Type::type_form(targ))
                    }
                    Type::ClassDef(cls) => Type::type_form(self.specialize(
                        &cls,
                        xs.map(|x| self.expr_untype(x)),
                        x.range,
                    )),
                    Type::Type(box Type::SpecialForm(special)) => {
                        self.apply_special_form(special, xs, x.range)
                    }
                    Type::Tuple(Tuple::Concrete(elts)) if xs.len() == 1 => match &xs[0] {
                        Expr::Slice(ExprSlice {
                            lower: lower_expr,
                            upper: upper_expr,
                            step: None,
                            ..
                        }) => {
                            let lower_literal = match lower_expr {
                                Some(box expr) => {
                                    let lower_type = self.expr_infer(expr);
                                    match &lower_type {
                                        Type::Literal(Lit::Int(idx)) => Some(*idx),
                                        _ => None,
                                    }
                                }
                                None => Some(0),
                            };
                            let upper_literal = match upper_expr {
                                Some(box expr) => {
                                    let upper_type = self.expr_infer(expr);
                                    match &upper_type {
                                        Type::Literal(Lit::Int(idx)) => Some(*idx),
                                        _ => None,
                                    }
                                }
                                None => Some(elts.len() as i64),
                            };
                            match (lower_literal, upper_literal) {
                                (Some(lower), Some(upper))
                                    if lower <= upper
                                        && lower >= 0
                                        && upper >= 0
                                        && upper <= elts.len() as i64 =>
                                {
                                    Type::Tuple(Tuple::concrete(
                                        elts[lower as usize..upper as usize].to_vec(),
                                    ))
                                }
                                _ => self.todo("tuple slice", x),
                            }
                        }
                        _ => {
                            let idx_type = self.expr_infer(&xs[0]);
                            match &idx_type {
                                Type::Literal(Lit::Int(idx)) => {
                                    let elt_idx = if *idx >= 0 {
                                        *idx
                                    } else {
                                        elts.len() as i64 + *idx
                                    } as usize;
                                    if let Some(elt) = elts.get(elt_idx) {
                                        elt.clone()
                                    } else {
                                        self.error(
                                        x.range,
                                        format!(
                                            "Index {idx} out of range for tuple with {} elements",
                                            elts.len()
                                        ),
                                    )
                                    }
                                }
                                _ => self.call_method_or_error(
                                    &Type::Tuple(Tuple::Concrete(elts)),
                                    &dunder::GETITEM,
                                    x.range,
                                    &[CallArg::Expr(&x.slice)],
                                    &[],
                                ),
                            }
                        }
                    },
                    Type::Tuple(Tuple::Unbounded(elt)) if xs.len() == 1 => self
                        .call_method_or_error(
                            &Type::Tuple(Tuple::Unbounded(elt)),
                            &dunder::GETITEM,
                            x.range,
                            &[CallArg::Expr(&x.slice)],
                            &[],
                        ),
                    Type::Any(style) => style.propagate(),
                    Type::ClassType(_) => self.call_method_or_error(
                        &fun,
                        &dunder::GETITEM,
                        x.range,
                        &[CallArg::Expr(&x.slice)],
                        &[],
                    ),
                    Type::TypedDict(typed_dict) => match self.expr_infer(&x.slice) {
                        Type::Literal(Lit::String(field_name)) => {
                            if let Some(field) =
                                typed_dict.fields().get(&Name::new(field_name.clone()))
                            {
                                field.ty.clone()
                            } else {
                                self.error(
                                    x.slice.range(),
                                    format!(
                                        "TypedDict `{}` does not have key `{}`",
                                        typed_dict.name(),
                                        field_name
                                    ),
                                )
                            }
                        }
                        t => self.error(
                            x.slice.range(),
                            format!(
                                "Invalid key for TypedDict `{}`, got `{}`",
                                typed_dict.name(),
                                t.deterministic_printing()
                            ),
                        ),
                    },
                    t => self.error(
                        x.range,
                        format!(
                            "Can't apply arguments to non-class, got {}",
                            t.deterministic_printing()
                        ),
                    ),
                }
            }
            Expr::Starred(_) => self.todo("Answers::expr_infer", x),
            Expr::Name(x) => match x.id.as_str() {
                "Any" => Type::type_form(Type::any_explicit()),
                _ => self
                    .get(&Key::Usage(ShortIdentifier::expr_name(x)))
                    .arc_clone(),
            },
            Expr::Slice(_) => {
                // TODO(stroxler, yangdanny): slices are generic, we should not hard code to int.
                let int = self.stdlib.int().to_type();
                self.stdlib.slice(int.clone(), int.clone(), int).to_type()
            }
            Expr::IpyEscapeCommand(x) => {
                self.error(x.range, "IPython escapes are not supported".to_owned())
            }
        }
    }
}

fn is_special_name(x: &Expr, name: &str) -> bool {
    match x {
        Expr::Name(x) => x.id.as_str() == name,
        _ => false,
    }
}

fn arguments_one_string(x: &Arguments) -> Option<Identifier> {
    if x.args.len() == 1 && x.keywords.is_empty() {
        match &x.args[0] {
            Expr::StringLiteral(x) => Some(Identifier {
                id: Name::new(x.value.to_str()),
                range: x.range,
            }),
            _ => None,
        }
    } else {
        None
    }
}
