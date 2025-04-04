/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;

use itertools::izip;
use itertools::EitherOrBoth;
use itertools::Itertools;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use starlark_map::Hashed;

use crate::alt::answers::LookupAnswer;
use crate::dunder;
use crate::solver::solver::Subset;
use crate::types::callable::Callable;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::ClassType;
use crate::types::class::TArgs;
use crate::types::quantified::QuantifiedKind;
use crate::types::simplify::unions;
use crate::types::tuple::Tuple;
use crate::types::type_var::Restriction;
use crate::types::type_var::Variance;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::TParams;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> Subset<'a, Ans> {
    /// Can a function with l_args be called as a function with u_args?
    pub fn is_subset_param_list(&mut self, l_args: &[Param], u_args: &[Param]) -> bool {
        let mut l_args_iter = l_args.iter();
        let mut u_args_iter = u_args.iter();
        let mut l_arg = l_args_iter.next();
        let mut u_arg = u_args_iter.next();
        // Handle positional args
        loop {
            match (l_arg, u_arg) {
                (None, None) => return true,
                (
                    Some(Param::PosOnly(l, l_req) | Param::Pos(_, l, l_req)),
                    Some(Param::PosOnly(u, u_req)),
                ) if (*u_req == Required::Required || *l_req == Required::Optional) => {
                    if self.is_subset_eq(u, l) {
                        l_arg = l_args_iter.next();
                        u_arg = u_args_iter.next();
                    } else {
                        return false;
                    }
                }
                (Some(Param::Pos(l_name, l, l_req)), Some(Param::Pos(u_name, u, u_req)))
                    if l_name == u_name
                        && (*u_req == Required::Required || *l_req == Required::Optional) =>
                {
                    if self.is_subset_eq(u, l) {
                        l_arg = l_args_iter.next();
                        u_arg = u_args_iter.next();
                    } else {
                        return false;
                    }
                }
                (Some(Param::VarArg(_, Type::Unpack(l))), None) => {
                    if self.is_subset_eq(&Type::tuple(Vec::new()), l) {
                        l_arg = l_args.iter().next();
                    } else {
                        return false;
                    }
                }
                (
                    Some(
                        Param::PosOnly(_, Required::Optional)
                        | Param::Pos(_, _, Required::Optional)
                        | Param::KwOnly(_, _, Required::Optional)
                        | Param::VarArg(_, _)
                        | Param::Kwargs(_),
                    ),
                    None,
                ) => return true,
                (
                    Some(Param::VarArg(_, Type::Unpack(box l))),
                    Some(Param::PosOnly(_, Required::Required)),
                ) => {
                    let mut u_types = Vec::new();
                    loop {
                        if let Some(Param::PosOnly(u, Required::Required)) = u_arg {
                            u_types.push(u.clone());
                            u_arg = u_args_iter.next();
                        } else if let Some(Param::VarArg(_, Type::Unpack(box u))) = u_arg {
                            if self.is_subset_eq(
                                &Type::Tuple(Tuple::unpacked(u_types, u.clone(), Vec::new())),
                                l,
                            ) {
                                l_arg = l_args_iter.next();
                                u_arg = u_args_iter.next();
                                break;
                            } else {
                                return false;
                            }
                        } else if let Some(Param::VarArg(_, u)) = u_arg {
                            if self.is_subset_eq(
                                &Type::Tuple(Tuple::unpacked(
                                    u_types,
                                    Type::Tuple(Tuple::unbounded(u.clone())),
                                    Vec::new(),
                                )),
                                l,
                            ) {
                                l_arg = l_args_iter.next();
                                u_arg = u_args_iter.next();
                                break;
                            } else {
                                return false;
                            }
                        } else if self.is_subset_eq(&Type::tuple(u_types), l) {
                            l_arg = l_args_iter.next();
                            break;
                        } else {
                            return false;
                        }
                    }
                }
                (
                    Some(Param::PosOnly(_, _) | Param::Pos(_, _, _)),
                    Some(Param::VarArg(_, Type::Unpack(box u))),
                ) => {
                    let mut l_types = Vec::new();
                    loop {
                        if let Some(Param::PosOnly(l, _) | Param::Pos(_, l, _)) = l_arg {
                            l_types.push(l.clone());
                            l_arg = l_args_iter.next();
                        } else if let Some(Param::VarArg(_, Type::Unpack(box l))) = l_arg {
                            if self.is_subset_eq(
                                u,
                                &Type::Tuple(Tuple::unpacked(l_types, l.clone(), Vec::new())),
                            ) {
                                l_arg = l_args_iter.next();
                                u_arg = u_args_iter.next();
                                break;
                            } else {
                                return false;
                            }
                        } else if let Some(Param::VarArg(_, l)) = l_arg {
                            if self.is_subset_eq(
                                u,
                                &Type::Tuple(Tuple::unpacked(
                                    l_types,
                                    Type::Tuple(Tuple::unbounded(l.clone())),
                                    Vec::new(),
                                )),
                            ) {
                                l_arg = l_args_iter.next();
                                u_arg = u_args_iter.next();
                                break;
                            } else {
                                return false;
                            }
                        } else if self.is_subset_eq(u, &Type::tuple(l_types)) {
                            u_arg = u_args_iter.next();
                            break;
                        } else {
                            return false;
                        }
                    }
                }
                (Some(Param::VarArg(_, l)), Some(Param::PosOnly(u, Required::Required))) => {
                    if self.is_subset_eq(u, l) {
                        u_arg = u_args_iter.next();
                    } else {
                        return false;
                    }
                }
                (
                    Some(Param::VarArg(_, Type::Unpack(box l))),
                    Some(Param::VarArg(_, Type::Unpack(box u))),
                ) => {
                    if self.is_subset_eq(u, l) {
                        l_arg = l_args_iter.next();
                        u_arg = u_args_iter.next();
                    } else {
                        return false;
                    }
                }
                (Some(Param::VarArg(_, l)), Some(Param::VarArg(_, Type::Unpack(box u)))) => {
                    if self.is_subset_eq(u, &Type::Tuple(Tuple::unbounded(l.clone()))) {
                        l_arg = l_args_iter.next();
                        u_arg = u_args_iter.next();
                    } else {
                        return false;
                    }
                }
                (Some(Param::VarArg(_, Type::Unpack(box l))), Some(Param::VarArg(_, u))) => {
                    if self.is_subset_eq(&Type::Tuple(Tuple::unbounded(u.clone())), l) {
                        l_arg = l_args_iter.next();
                        u_arg = u_args_iter.next();
                    } else {
                        return false;
                    }
                }
                (Some(Param::VarArg(_, l)), Some(Param::VarArg(_, u))) => {
                    if self.is_subset_eq(u, l) {
                        l_arg = l_args_iter.next();
                        u_arg = u_args_iter.next();
                    } else {
                        return false;
                    }
                }
                (Some(_), Some(Param::KwOnly(_, _, _) | Param::Kwargs(_))) => {
                    break;
                }
                _ => return false,
            }
        }
        let mut l_keywords = SmallMap::new();
        let mut l_kwargs = None;
        for arg in Option::into_iter(l_arg).chain(l_args_iter) {
            match arg {
                Param::KwOnly(name, ty, required) | Param::Pos(name, ty, required) => {
                    l_keywords.insert(name.clone(), (ty.clone(), *required));
                }
                Param::Kwargs(ty) => l_kwargs = Some(ty.clone()),
                _ => (),
            }
        }
        let mut u_keywords = SmallMap::new();
        let mut u_kwargs = None;
        for arg in Option::into_iter(u_arg).chain(u_args_iter) {
            match arg {
                Param::KwOnly(name, ty, required) => {
                    u_keywords.insert(name.clone(), (ty.clone(), *required));
                }
                Param::Kwargs(ty) => u_kwargs = Some(ty.clone()),
                _ => (),
            }
        }
        let object_type = self
            .type_order
            .stdlib()
            .object_class_type()
            .clone()
            .to_type();
        // Expand typed dict kwargs if necessary, check regular kwargs
        let l_kwargs = match (l_kwargs, u_kwargs) {
            (
                Some(Type::Unpack(box Type::TypedDict(box l_typed_dict))),
                Some(Type::Unpack(box Type::TypedDict(box u_typed_dict))),
            ) => {
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&l_typed_dict)
                {
                    l_keywords.insert(name, (ty, required));
                }
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&u_typed_dict)
                {
                    u_keywords.insert(name, (ty, required));
                }
                Some(object_type)
            }
            (Some(Type::Unpack(box Type::TypedDict(box l_typed_dict))), _) => {
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&l_typed_dict)
                {
                    l_keywords.insert(name, (ty, required));
                }
                Some(object_type)
            }
            (l_kwargs @ Some(_), Some(Type::Unpack(box Type::TypedDict(box u_typed_dict)))) => {
                for (name, ty, required) in self.type_order.typed_dict_kw_param_info(&u_typed_dict)
                {
                    u_keywords.insert(name, (ty, required));
                }
                l_kwargs
            }
            (Some(l), Some(u)) => {
                if !self.is_subset_eq(&u, &l) {
                    return false;
                }
                Some(l)
            }
            (None, Some(_)) => {
                return false;
            }
            (l_kwargs, _) => l_kwargs,
        };
        // Handle keyword-only args
        for (name, (u_ty, u_req)) in u_keywords.iter() {
            if let Some((l_ty, l_req)) = l_keywords.shift_remove_hashed(Hashed::new(name)) {
                if !(*u_req == Required::Required || l_req == Required::Optional)
                    || !self.is_subset_eq(u_ty, &l_ty)
                {
                    return false;
                }
            } else if let Some(l_ty) = &l_kwargs {
                if !self.is_subset_eq(u_ty, l_ty) {
                    return false;
                }
            } else {
                return false;
            }
        }
        for (_, l_req) in l_keywords.values() {
            if *l_req == Required::Required {
                return false;
            }
        }
        true
    }

    fn try_lookup_attr_from_class(&mut self, cls: &Type, name: &Name) -> Option<Type> {
        self.type_order
            .try_lookup_attr_no_union(cls, name)
            .and_then(|attr| self.type_order.resolve_as_instance_method(attr))
    }

    fn convert_class_to_callable(&mut self, cls: &ClassType) -> Option<Type> {
        let class_type = cls.clone().to_type();
        if self.type_order.is_protocol(cls.class_object()) {
            // If it's a protocol, always use the __call__ method
            return self.try_lookup_attr_from_class(&class_type, &dunder::CALL);
        } else if let Some(metaclass_call_attr_ty) = self.type_order.get_metaclass_dunder_call(cls)
        {
            // If the class has a custom metaclass and the return type of the metaclass's __call__
            // is not a subclass of the current class, use that and ignore __new__ and __init__
            if metaclass_call_attr_ty
                .callable_return_type()
                .is_some_and(|ret| {
                    !self
                        .type_order
                        .is_compatible_constructor_return(&ret, cls.class_object())
                })
            {
                return Some(metaclass_call_attr_ty);
            }
        }
        // Default constructor that takes no args and returns Self.
        let default_constructor = || {
            Type::Callable(Box::new(Callable::list(
                ParamList::new(Vec::new()),
                class_type.clone(),
            )))
        };
        // Check the __new__ method and whether it comes from object or has been overridden
        let (new_attr_ty, overrides_new) = if let Some(t) = self
            .type_order
            .get_dunder_new(cls)
            .and_then(|t| t.to_unbound_callable())
        {
            if t.callable_return_type().is_some_and(|ret| {
                !self
                    .type_order
                    .is_compatible_constructor_return(&ret, cls.class_object())
            }) {
                // If the return type of __new__ is not a subclass of the current class, use that and ignore __init__
                return Some(t);
            }
            (t, true)
        } else {
            (default_constructor(), false)
        };
        // Check the __init__ method and whether it comes from object or has been overridden
        let (init_attr_ty, overrides_init) =
            if let Some(mut t) = self.type_order.get_dunder_init(cls, false) {
                // Replace the return type with Self (the current class)
                t.set_callable_return_type(class_type.clone());
                (t, true)
            } else {
                (default_constructor(), false)
            };
        if !overrides_new && overrides_init {
            // If `__init__` is overridden and `__new__` is inherited from object, use `__init__`
            Some(init_attr_ty)
        } else if overrides_new && !overrides_init {
            // If `__new__` is overridden and `__init__` is inherited from object, use `__new__`
            Some(new_attr_ty)
        } else {
            let result = unions(vec![new_attr_ty, init_attr_ty]);
            // If both are overridden, take the union
            // Only if neither are overridden, use the `__new__` and `__init__` from object
            Some(result)
        }
    }

    pub fn is_subset_protocol(&mut self, got: Type, protocol: ClassType) -> bool {
        let recursive_check = (got.clone(), Type::ClassType(protocol.clone()));
        if !self.recursive_assumptions.insert(recursive_check) {
            // Assume recursive checks are true
            return true;
        }
        let to = self.type_order;
        let protocol_members = to.get_protocol_member_names(protocol.class_object());
        for name in protocol_members {
            if name == dunder::INIT || name == dunder::NEW {
                // Protocols can't be instantiated
                continue;
            }
            if matches!(
                got,
                Type::Callable(_) | Type::Function(_) | Type::BoundMethod(_)
            ) && name == dunder::CALL
                && let Some(want) = self.convert_class_to_callable(&protocol)
            {
                if let Type::BoundMethod(box ref method) = want
                    && let Some(want_no_self) = method.to_callable()
                {
                    if !self.is_subset_eq(&got, &want_no_self) {
                        return false;
                    }
                } else if !self.is_subset_eq(&got, &want) {
                    return false;
                }
            } else if let got_attrs = to.try_lookup_attr(&got, &name)
                && !got_attrs.is_empty()
                && let Some(want) = to.try_lookup_attr_no_union(&protocol.clone().to_type(), &name)
            {
                for got in got_attrs {
                    if !to
                        .is_attr_subset(&got, &want, &mut |got, want| self.is_subset_eq(got, want))
                    {
                        return false;
                    }
                }
            } else {
                return false;
            }
        }
        true
    }

    pub fn is_subset_tuple(&mut self, got: &Tuple, want: &Tuple) -> bool {
        match (got, want) {
            (Tuple::Concrete(lelts), Tuple::Concrete(uelts)) => {
                if lelts.len() == uelts.len() {
                    lelts
                        .iter()
                        .zip(uelts)
                        .all(|(l, u)| self.is_subset_eq(l, u))
                } else {
                    false
                }
            }
            (Tuple::Unbounded(box Type::Any(_)), _) | (_, Tuple::Unbounded(box Type::Any(_))) => {
                true
            }
            (Tuple::Concrete(lelts), Tuple::Unbounded(box u)) => {
                lelts.iter().all(|l| self.is_subset_eq(l, u))
            }
            (Tuple::Unbounded(box l), Tuple::Unbounded(box u)) => self.is_subset_eq(l, u),
            (Tuple::Concrete(lelts), Tuple::Unpacked(box (u_prefix, u_middle, u_suffix))) => {
                if lelts.len() < u_prefix.len() + u_suffix.len() {
                    false
                } else {
                    let mut l_middle = Vec::new();
                    lelts.iter().enumerate().all(|(idx, l)| {
                        if idx < u_prefix.len() {
                            self.is_subset_eq(l, &u_prefix[idx])
                        } else if idx >= lelts.len() - u_suffix.len() {
                            self.is_subset_eq(l, &u_suffix[idx + u_suffix.len() - lelts.len()])
                        } else {
                            l_middle.push(l.clone());
                            true
                        }
                    }) && self.is_subset_eq(&Type::Tuple(Tuple::Concrete(l_middle)), u_middle)
                }
            }
            (Tuple::Unbounded(_), Tuple::Unpacked(box (u_prefix, u_middle, u_suffix))) => {
                u_prefix.is_empty()
                    && u_suffix.is_empty()
                    && self.is_subset_eq(&Type::Tuple(got.clone()), u_middle)
            }
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Unbounded(box u)) => {
                l_prefix.iter().all(|l| self.is_subset_eq(l, u))
                    && l_suffix.iter().all(|l| self.is_subset_eq(l, u))
                    && self.is_subset_eq(l_middle, &Type::Tuple(want.clone()))
            }
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Concrete(uelts)) => {
                if uelts.len() < l_prefix.len() + l_suffix.len() {
                    false
                } else {
                    let mut u_middle = Vec::new();
                    uelts.iter().enumerate().all(|(idx, u)| {
                        if idx < l_prefix.len() {
                            self.is_subset_eq(&l_prefix[idx], u)
                        } else if idx >= uelts.len() - l_suffix.len() {
                            self.is_subset_eq(&l_suffix[idx + l_suffix.len() - uelts.len()], u)
                        } else {
                            u_middle.push(u.clone());
                            true
                        }
                    }) && self.is_subset_eq(l_middle, &Type::Tuple(Tuple::Concrete(u_middle)))
                }
            }
            (
                Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)),
                Tuple::Unpacked(box (u_prefix, u_middle, u_suffix)),
            ) => {
                // Invariant: 0-2 of these are non-empty
                // l_before and u_before cannot both be non-empty
                // l_after and u_after cannot both be non-empty
                let mut l_before = Vec::new();
                let mut l_after = Vec::new();
                let mut u_before = Vec::new();
                let mut u_after = Vec::new();
                if !(l_prefix
                    .iter()
                    .zip_longest(u_prefix.iter())
                    .all(|pair| match pair {
                        EitherOrBoth::Both(l, u) => self.is_subset_eq(l, u),
                        EitherOrBoth::Left(l) => {
                            l_before.push(l.clone());
                            true
                        }
                        EitherOrBoth::Right(u) => {
                            u_before.push(u.clone());
                            true
                        }
                    })
                    && l_suffix
                        .iter()
                        .rev()
                        .zip_longest(u_suffix.iter().rev())
                        .all(|pair| match pair {
                            EitherOrBoth::Both(l, u) => self.is_subset_eq(l, u),
                            EitherOrBoth::Left(l) => {
                                l_after.push(l.clone());
                                true
                            }
                            EitherOrBoth::Right(u) => {
                                u_after.push(u.clone());
                                true
                            }
                        }))
                {
                    return false;
                }
                l_after.reverse();
                u_after.reverse();
                self.is_subset_eq(
                    &Type::Tuple(Tuple::unpacked(l_before, l_middle.clone(), l_after)),
                    u_middle,
                ) && self.is_subset_eq(
                    l_middle,
                    &Type::Tuple(Tuple::unpacked(u_before, u_middle.clone(), u_after)),
                )
            }
            _ => false,
        }
    }

    pub fn is_paramlist_subset_of_paramspec(
        &mut self,
        got: &ParamList,
        want_ts: &[Type],
        want_pspec: &Type,
    ) -> bool {
        let args = ParamList::new_types(want_ts);
        if got.len() < args.len() {
            return false;
        }
        let (pre, post) = got.items().split_at(args.len());
        if !self.is_subset_param_list(pre, args.items()) {
            return false;
        }
        self.is_subset_eq(
            &Type::ParamSpecValue(ParamList::new(post.to_vec())),
            want_pspec,
        )
    }

    pub fn is_paramspec_subset_of_paramlist(
        &mut self,
        got_ts: &[Type],
        got_pspec: &Type,
        want: &ParamList,
    ) -> bool {
        let args = ParamList::new_types(got_ts);
        if want.len() < args.len() {
            return false;
        }
        let (pre, post) = want.items().split_at(args.len());
        if !self.is_subset_param_list(args.items(), pre) {
            return false;
        }
        self.is_subset_eq(
            got_pspec,
            &Type::ParamSpecValue(ParamList::new(post.to_vec())),
        )
    }

    pub fn is_paramspec_subset_of_paramspec(
        &mut self,
        got_ts: &[Type],
        got_pspec: &Type,
        want_ts: &[Type],
        want_pspec: &Type,
    ) -> bool {
        match got_ts.len().cmp(&want_ts.len()) {
            Ordering::Greater => {
                let (pre, post) = got_ts.split_at(want_ts.len());
                for (l, u) in pre.iter().zip(want_ts.iter()) {
                    if !self.is_subset_eq(u, l) {
                        return false;
                    }
                }
                let post = post.to_vec().into_boxed_slice();
                self.is_subset_eq(
                    &Type::Concatenate(post, Box::new(want_pspec.clone())),
                    got_pspec,
                )
            }
            Ordering::Less => {
                let (pre, post) = want_ts.split_at(got_ts.len());
                for (l, u) in got_ts.iter().zip(pre.iter()) {
                    if !self.is_subset_eq(u, l) {
                        return false;
                    }
                }
                let post = post.to_vec().into_boxed_slice();
                self.is_subset_eq(
                    want_pspec,
                    &Type::Concatenate(post, Box::new(got_pspec.clone())),
                )
            }
            Ordering::Equal => {
                for (l, u) in got_ts.iter().zip(want_ts.iter()) {
                    if !self.is_subset_eq(u, l) {
                        return false;
                    }
                }
                self.is_subset_eq(want_pspec, got_pspec)
            }
        }
    }

    /// Implementation of subset equality for Type, other than Var.
    pub fn is_subset_eq_impl(&mut self, got: &Type, want: &Type) -> bool {
        match (got, want) {
            (Type::Any(_), _) => {
                // Special case in Python, because we want `x: int = Any` to be valid,
                // as Any is more the lack of information, rather than the actual union it is modelled to be.
                true
            }
            (_, Type::Any(_)) => true,
            (Type::Never(_), _) => true,
            (_, Type::ClassType(want))
                if *want == *self.type_order.stdlib().object_class_type() =>
            {
                true // everything is an instance of `object`
            }
            (Type::Union(ls), u) => ls.iter().all(|l| self.is_subset_eq(l, u)),
            (l, Type::Intersect(us)) => us.iter().all(|u| self.is_subset_eq(l, u)),
            (l, Type::Overload(overload)) => overload
                .signatures
                .iter()
                .all(|u| self.is_subset_eq(l, &u.as_type())),
            (l, Type::Union(us)) => us.iter().any(|u| self.is_subset_eq(l, u)),
            (Type::Intersect(ls), u) => ls.iter().any(|l| self.is_subset_eq(l, u)),
            (Type::Module(_), Type::ClassType(cls))
                if cls.class_object().has_qname("types", "ModuleType") =>
            {
                true
            }
            (
                Type::Function(_)
                | Type::Overload(_)
                | Type::Forall(box Forall {
                    body: Forallable::Function(_),
                    ..
                }),
                Type::ClassType(cls),
            ) if cls.class_object().has_qname("types", "FunctionType") => true,
            (Type::BoundMethod(_), Type::ClassType(cls))
                if cls.class_object().has_qname("types", "MethodType") =>
            {
                true
            }
            (Type::Overload(overload), u) => overload
                .signatures
                .iter()
                .any(|l| self.is_subset_eq(&l.as_type(), u)),
            (Type::BoundMethod(box method), Type::Callable(_) | Type::Function(_))
                if let Some(l_no_self) = method.to_callable() =>
            {
                self.is_subset_eq_impl(&l_no_self, want)
            }
            (Type::Callable(_) | Type::Function(_), Type::BoundMethod(box method))
                if let Some(u_no_self) = method.to_callable() =>
            {
                self.is_subset_eq_impl(got, &u_no_self)
            }
            (Type::BoundMethod(box l), Type::BoundMethod(box u))
                if let Some(l_no_self) = l.to_callable()
                    && let Some(u_no_self) = u.to_callable() =>
            {
                self.is_subset_eq_impl(&l_no_self, &u_no_self)
            }
            (
                Type::Callable(box l)
                | Type::Function(box Function {
                    signature: l,
                    metadata: _,
                }),
                Type::Callable(box u)
                | Type::Function(box Function {
                    signature: u,
                    metadata: _,
                }),
            ) => {
                self.is_subset_eq(&l.ret, &u.ret)
                    && match (&l.params, &u.params) {
                        (Params::Ellipsis, Params::ParamSpec(_, pspec)) => {
                            self.is_subset_eq(&Type::Ellipsis, pspec)
                        }
                        (Params::ParamSpec(_, pspec), Params::Ellipsis) => {
                            self.is_subset_eq(pspec, &Type::Ellipsis)
                        }
                        (Params::Ellipsis, _) | (_, Params::Ellipsis) => true,
                        (Params::List(l_args), Params::List(u_args)) => {
                            self.is_subset_param_list(l_args.items(), u_args.items())
                        }
                        (Params::List(ls), Params::ParamSpec(args, pspec)) => {
                            self.is_paramlist_subset_of_paramspec(ls, args, pspec)
                        }
                        (Params::ParamSpec(args, pspec), Params::List(ls)) => {
                            self.is_paramspec_subset_of_paramlist(args, pspec, ls)
                        }
                        (Params::ParamSpec(box ls, p1), Params::ParamSpec(box us, p2)) => {
                            self.is_paramspec_subset_of_paramspec(ls, p1, us, p2)
                        }
                    }
            }
            (Type::TypedDict(got), Type::TypedDict(want)) => {
                // For each key in `want`, `got` has the corresponding key
                // and the corresponding value type in `got` is consistent with the value type in `want`.
                // For each required key in `got`, the corresponding key is required in `want`.
                // For each non-required key in `got`, the corresponding key is not required in `want`.
                let got_fields = self.type_order.typed_dict_fields(got);
                let want_fields = self.type_order.typed_dict_fields(want);

                want_fields.iter().all(|(k, want_v)| {
                    got_fields
                        .get(k)
                        .is_some_and(|got_v| self.is_subset_eq(&got_v.ty, &want_v.ty))
                }) && got_fields.iter().all(|(k, got_v)| {
                    want_fields
                        .get(k)
                        .is_none_or(|want_v| got_v.required == want_v.required)
                })
            }
            (Type::TypedDict(_), _) => {
                let stdlib = self.type_order.stdlib();
                self.is_subset_eq(
                    &stdlib
                        .mapping(
                            stdlib.str().to_type(),
                            stdlib.object_class_type().clone().to_type(),
                        )
                        .to_type(),
                    want,
                )
            }
            (Type::ClassType(ty), _) | (_, Type::ClassType(ty))
                if self.type_order.extends_any(ty.class_object()) =>
            {
                true
            }
            (Type::ClassType(got), Type::ClassType(want))
                if *want == self.type_order.stdlib().float()
                    && *got == self.type_order.stdlib().int() =>
            {
                true
            }
            (Type::ClassType(got), Type::ClassType(want))
                if *want == self.type_order.stdlib().complex()
                    && (*got == self.type_order.stdlib().int()
                        || *got == self.type_order.stdlib().float()) =>
            {
                true
            }
            (Type::ClassType(got), Type::ClassType(want)) => {
                let got_is_protocol = self.type_order.is_protocol(got.class_object());
                let want_is_protocol = self.type_order.is_protocol(want.class_object());
                if got_is_protocol && !want_is_protocol {
                    // Protocols are never assignable to concrete types
                    return false;
                }
                match self.type_order.as_superclass(got, want.class_object()) {
                    Some(got) => self.check_targs(got.targs(), want.targs(), want.tparams()),
                    // Structural checking for assigning to protocols
                    None if want_is_protocol => {
                        self.is_subset_protocol(got.clone().to_type(), want.clone())
                    }
                    _ => false,
                }
            }
            (_, Type::ClassType(want)) if self.type_order.is_protocol(want.class_object()) => {
                self.is_subset_protocol(got.clone(), want.clone())
            }
            // Callable protocols
            (
                Type::ClassType(got),
                Type::BoundMethod(_) | Type::Callable(_) | Type::Function(_),
            ) if self.type_order.is_protocol(got.class_object())
                && let Some(call_ty) = self.convert_class_to_callable(got) =>
            {
                self.is_subset_eq(&call_ty, want)
            }
            // Constructors as callables
            (
                Type::Type(box Type::ClassType(got)),
                Type::BoundMethod(_) | Type::Callable(_) | Type::Function(_),
            ) if let Some(call_ty) = self.convert_class_to_callable(got) => {
                self.is_subset_eq(&call_ty, want)
            }
            (Type::ClassDef(got), Type::BoundMethod(_) | Type::Callable(_) | Type::Function(_)) => {
                self.is_subset_eq(
                    &Type::type_form(self.type_order.promote_silently(got)),
                    want,
                )
            }
            (Type::ClassType(got), Type::Tuple(_))
                if *got.class_object() == self.type_order.stdlib().tuple_class_object()
                    && got.targs().as_slice().len() == 1 =>
            {
                let mut tuple_targ = got.targs().as_slice()[0].clone();
                if let Type::Var(var) = tuple_targ {
                    tuple_targ = self.force_var(var);
                }
                self.is_subset_eq(&Type::Tuple(Tuple::Unbounded(Box::new(tuple_targ))), want)
            }
            (Type::ClassDef(got), Type::ClassDef(want)) => {
                self.type_order.has_superclass(got, want)
            }
            (Type::ClassDef(got), Type::Type(want)) => {
                self.is_subset_eq(&self.type_order.promote_silently(got), want)
            }
            (Type::Type(box Type::ClassType(got)), Type::ClassDef(want)) => {
                self.type_order.has_superclass(got.class_object(), want)
            }
            (Type::ClassDef(got), Type::ClassType(want)) => {
                self.type_order.has_metaclass(got, want)
            }
            (Type::Type(box Type::ClassType(got)), Type::ClassType(want)) => {
                self.type_order.has_metaclass(got.class_object(), want)
            }
            (Type::Type(box Type::Any(_)), Type::ClassDef(_)) => true,
            (Type::ClassType(cls), want @ Type::Tuple(_))
                if let Some(elts) = self.type_order.named_tuple_element_types(cls) =>
            {
                self.is_subset_eq(&Type::Tuple(Tuple::Concrete(elts)), want)
            }
            (Type::ClassType(got), Type::SelfType(want)) => self
                .type_order
                .has_superclass(got.class_object(), want.class_object()),
            (Type::Type(box Type::ClassType(got)), Type::SelfType(want)) => {
                self.type_order.has_metaclass(got.class_object(), want)
            }
            (Type::SelfType(_), Type::SelfType(_)) => true,
            (Type::SelfType(got), _) => self.is_subset_eq(&Type::ClassType(got.clone()), want),
            (Type::Tuple(l), Type::Tuple(u)) => self.is_subset_tuple(l, u),
            (Type::Tuple(Tuple::Concrete(left_elts)), _) => {
                let tuple_type = self
                    .type_order
                    .stdlib()
                    .tuple(unions(left_elts.clone()))
                    .to_type();
                self.is_subset_eq(&tuple_type, want)
            }
            (Type::Tuple(Tuple::Unbounded(box left_elt)), _) => {
                let tuple_type = self.type_order.stdlib().tuple(left_elt.clone()).to_type();
                self.is_subset_eq(&tuple_type, want)
            }
            (
                Type::Tuple(Tuple::Unpacked(box (
                    prefix,
                    Type::Tuple(Tuple::Unbounded(box middle)),
                    suffix,
                ))),
                _,
            ) => {
                let mut elts = prefix.clone();
                elts.push(middle.clone());
                elts.extend(suffix.clone());
                let tuple_type = self.type_order.stdlib().tuple(unions(elts)).to_type();
                self.is_subset_eq(&tuple_type, want)
            }
            (Type::Tuple(Tuple::Unpacked(box (prefix, middle, suffix))), _) => {
                let mut elts = prefix.clone();
                elts.extend(suffix.clone());
                let tuple_type = self.type_order.stdlib().tuple(unions(elts)).to_type();
                self.is_subset_eq(&tuple_type, want) && self.is_subset_eq(middle, want)
            }
            (Type::Literal(lit), Type::LiteralString) => lit.is_string(),
            (Type::Literal(lit), t @ Type::ClassType(_)) => self.is_subset_eq(
                &lit.general_class_type(self.type_order.stdlib()).to_type(),
                t,
            ),
            (Type::Literal(l_lit), Type::Literal(u_lit)) => l_lit == u_lit,
            (Type::LiteralString, _) => {
                self.is_subset_eq(&self.type_order.stdlib().str().to_type(), want)
            }
            (Type::Type(l), Type::Type(u)) => self.is_subset_eq(l, u),
            (Type::Type(_), _) => {
                self.is_subset_eq(&self.type_order.stdlib().builtins_type().to_type(), want)
            }
            (Type::TypeGuard(l), Type::TypeGuard(u)) => {
                // TypeGuard is covariant
                self.is_subset_eq(l, u)
            }
            (Type::TypeGuard(_) | Type::TypeIs(_), _) => {
                self.is_subset_eq(&self.type_order.stdlib().bool().to_type(), want)
            }
            (Type::Ellipsis, Type::ParamSpecValue(_) | Type::Concatenate(_, _))
            | (Type::ParamSpecValue(_) | Type::Concatenate(_, _), Type::Ellipsis) => true,
            (Type::ParamSpecValue(ls), Type::ParamSpecValue(us)) => {
                self.is_subset_param_list(ls.items(), us.items())
            }
            (Type::ParamSpecValue(ls), Type::Concatenate(box us, box u_pspec)) => {
                self.is_paramlist_subset_of_paramspec(ls, us, u_pspec)
            }
            (Type::Concatenate(box ls, box l_pspec), Type::ParamSpecValue(us)) => {
                self.is_paramspec_subset_of_paramlist(ls, l_pspec, us)
            }
            (Type::Concatenate(box ls, box l_pspec), Type::Concatenate(box us, box u_pspec)) => {
                self.is_paramspec_subset_of_paramspec(ls, l_pspec, us, u_pspec)
            }
            (Type::Ellipsis, _) => {
                // Bit of a weird case - pretty sure we should be modelling these slightly differently
                // - probably not as a dedicated Type alternative.
                self.is_subset_eq(&self.type_order.stdlib().ellipsis_type().to_type(), want)
            }
            (Type::None, _) => {
                self.is_subset_eq(&self.type_order.stdlib().none_type().to_type(), want)
            }
            (_, Type::None) => {
                self.is_subset_eq(got, &self.type_order.stdlib().none_type().to_type())
            }
            (Type::Forall(_), _) => {
                // TODO: Probably need to do some kind of substitution here
                false
            }
            (Type::TypeAlias(ta), _) => {
                self.is_subset_eq_impl(&ta.as_value(self.type_order.stdlib()), want)
            }
            (Type::Quantified(q), t2) => match q.restriction() {
                Restriction::Bound(bound) => self.is_subset_eq(bound, t2),
                Restriction::Constraints(constraints) => constraints
                    .iter()
                    .all(|constraint| self.is_subset_eq(constraint, t2)),
                Restriction::Unrestricted => self.is_subset_eq_impl(
                    &self
                        .type_order
                        .stdlib()
                        .object_class_type()
                        .clone()
                        .to_type(),
                    want,
                ),
            },
            _ => false,
        }
    }

    fn check_targs(&mut self, got: &TArgs, want: &TArgs, params: &TParams) -> bool {
        let got = got.as_slice();
        let want = want.as_slice();
        assert_eq!(got.len(), want.len());
        assert_eq!(want.len(), params.len());
        for (got_arg, want_arg, param) in izip!(got, want, params.iter()) {
            let result = if param.quantified.kind() == QuantifiedKind::TypeVarTuple {
                self.is_equal(got_arg, want_arg)
            } else {
                match param.variance {
                    Variance::Covariant => self.is_subset_eq(got_arg, want_arg),
                    Variance::Contravariant => self.is_subset_eq(want_arg, got_arg),
                    Variance::Invariant => self.is_equal(got_arg, want_arg),
                }
            };
            if !result {
                return false;
            }
        }
        true
    }
}
