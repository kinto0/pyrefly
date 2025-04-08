/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;

use crate::alt::solve::TypeFormContext;
use crate::error::kind::ErrorKind;
use crate::types::class::ClassType;
use crate::types::stdlib::Stdlib;
use crate::types::type_var::Restriction;
use crate::types::types::Type;
use crate::util::uniques::Unique;
use crate::util::uniques::UniqueFactory;

#[derive(
    Debug, Clone, PartialEq, Eq, TypeEq, Hash, Visit, Ord, PartialOrd, VisitMut
)]
pub struct QuantifiedInfo {
    pub name: Name,
    pub kind: QuantifiedKind,
    pub default: Option<Type>,
    pub restriction: Restriction,
}

impl QuantifiedInfo {
    fn as_gradual_type_helper(&self, default: Option<&Type>) -> Type {
        default.map_or_else(
            || self.kind.empty_value(),
            |default| {
                default.clone().transform(&mut |default| match default {
                    Type::TypeVar(t) => {
                        *default = Self::type_var(
                            t.qname().id().clone(),
                            t.default().cloned(),
                            t.restriction().clone(),
                        )
                        .as_gradual_type();
                    }
                    Type::TypeVarTuple(t) => {
                        *default =
                            Self::type_var_tuple(t.qname().id().clone(), t.default().cloned())
                                .as_gradual_type();
                    }
                    Type::ParamSpec(p) => {
                        *default = Self::param_spec(p.qname().id().clone(), p.default().cloned())
                            .as_gradual_type();
                    }
                    Type::Quantified(q) => {
                        *default = q.as_gradual_type();
                    }
                    _ => {}
                })
            },
        )
    }

    pub fn as_gradual_type(&self) -> Type {
        self.as_gradual_type_helper(self.default.as_ref())
    }

    fn type_var(name: Name, default: Option<Type>, restriction: Restriction) -> Self {
        Self {
            name,
            kind: QuantifiedKind::TypeVar,
            restriction,
            default,
        }
    }

    fn param_spec(name: Name, default: Option<Type>) -> Self {
        Self {
            name,
            kind: QuantifiedKind::ParamSpec,
            restriction: Restriction::Unrestricted,
            default,
        }
    }

    fn type_var_tuple(name: Name, default: Option<Type>) -> Self {
        Self {
            name,
            kind: QuantifiedKind::TypeVarTuple,
            restriction: Restriction::Unrestricted,
            default,
        }
    }
}

#[derive(
    Debug, Clone, Visit, VisitMut, TypeEq, PartialEq, Eq, Ord, PartialOrd, Hash
)]
pub struct Quantified {
    /// Unique identifier
    unique: Unique,
    info: Box<QuantifiedInfo>,
}

#[derive(
    Debug, Clone, Copy, Visit, VisitMut, TypeEq, PartialEq, Eq, PartialOrd, Ord, Hash, Display
)]
pub enum QuantifiedKind {
    TypeVar,
    ParamSpec,
    TypeVarTuple,
}

impl QuantifiedKind {
    fn empty_value(self) -> Type {
        match self {
            QuantifiedKind::TypeVar => Type::any_implicit(),
            QuantifiedKind::ParamSpec => Type::Ellipsis,
            QuantifiedKind::TypeVarTuple => Type::any_tuple(),
        }
    }

    pub fn error_kind(self) -> ErrorKind {
        match self {
            QuantifiedKind::TypeVar => ErrorKind::InvalidTypeVar,
            QuantifiedKind::ParamSpec => ErrorKind::InvalidParamSpec,
            QuantifiedKind::TypeVarTuple => ErrorKind::InvalidTypeVarTuple,
        }
    }

    pub fn type_form_context_for_default(self) -> TypeFormContext {
        match self {
            QuantifiedKind::TypeVar => TypeFormContext::TypeVarDefault,
            QuantifiedKind::ParamSpec => TypeFormContext::ParamSpecDefault,
            QuantifiedKind::TypeVarTuple => TypeFormContext::TypeVarTupleDefault,
        }
    }
}

impl Display for Quantified {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.info.kind {
            QuantifiedKind::TypeVar => write!(f, "TypeVar[{}]", self.info.name),
            QuantifiedKind::ParamSpec => write!(f, "ParamSpec[{}]", self.info.name),
            QuantifiedKind::TypeVarTuple => write!(f, "TypeVarTuple[{}]", self.info.name),
        }
    }
}

impl Quantified {
    pub fn new(unique: Unique, info: QuantifiedInfo) -> Self {
        Quantified {
            unique,
            info: Box::new(info),
        }
    }

    pub fn type_var(
        name: Name,
        uniques: &UniqueFactory,
        default: Option<Type>,
        restriction: Restriction,
    ) -> Self {
        Self::new(
            uniques.fresh(),
            QuantifiedInfo::type_var(name, default, restriction),
        )
    }

    pub fn param_spec(name: Name, uniques: &UniqueFactory, default: Option<Type>) -> Self {
        Self::new(uniques.fresh(), QuantifiedInfo::param_spec(name, default))
    }

    pub fn type_var_tuple(name: Name, uniques: &UniqueFactory, default: Option<Type>) -> Self {
        Self::new(
            uniques.fresh(),
            QuantifiedInfo::type_var_tuple(name, default),
        )
    }

    pub fn to_type(self) -> Type {
        Type::Quantified(self)
    }

    pub fn as_value<'a>(&self, stdlib: &'a Stdlib) -> &'a ClassType {
        match self.info.kind {
            QuantifiedKind::TypeVar => stdlib.type_var(),
            QuantifiedKind::ParamSpec => stdlib.param_spec(),
            QuantifiedKind::TypeVarTuple => stdlib.type_var_tuple(),
        }
    }

    pub fn name(&self) -> &Name {
        &self.info.name
    }

    pub fn kind(&self) -> QuantifiedKind {
        self.info.kind
    }

    pub fn default(&self) -> Option<&Type> {
        self.info.default.as_ref()
    }

    pub fn restriction(&self) -> &Restriction {
        &self.info.restriction
    }

    pub fn is_type_var(&self) -> bool {
        matches!(self.info.kind, QuantifiedKind::TypeVar)
    }

    pub fn is_param_spec(&self) -> bool {
        matches!(self.info.kind, QuantifiedKind::ParamSpec)
    }

    pub fn is_type_var_tuple(&self) -> bool {
        matches!(self.info.kind, QuantifiedKind::TypeVarTuple)
    }

    pub fn as_gradual_type(&self) -> Type {
        self.info.as_gradual_type()
    }
}
