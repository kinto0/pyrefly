/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::TypeOrd;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::qname::QName;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::display::Fmt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;

use crate::class::ClassDefIndex;
use crate::class::ClassType;
use crate::dimension::gradual_size;
use crate::env::TypeEnv;
use crate::function::FuncDefIndex;
use crate::heap::TypeHeap;
use crate::stdlib::Stdlib;
use crate::type_alias::TypeAliasIndex;
use crate::type_ord::TypeOrd;
use crate::type_var::PreInferenceVariance;
use crate::type_var::Restriction;
use crate::types::Type;

/// Discriminator for the origin of a `Quantified`, making collisions structurally impossible
/// between quantifieds that share the same anchor but have different origins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantifiedOrigin {
    /// Legacy TypeVar scoped to a function/class/alias owner.
    ScopedLegacy,
    /// Legacy TypeVar that solving finds in a type, such as the value of a type alias or a
    /// callable annotation, rather than one that binding finds in its owner's signature or bases.
    InferredLegacy,
    /// PEP 695 type parameter.
    Pep695,
    /// Synthetic quantified created by pyrefly rather than written in source.
    Synthetic {
        /// Is this a Self quantified synthesized for `__new__` on a class?
        is_self: bool,
    },
    /// An inferred receiver shared by an instance method and a classmethod wrapper.
    MethodReceiver,
    /// A dimension or variadic shape declared by a shape declaration on a class or function.
    ShapeDeclaration,
    /// Synthetic binder for the parameter of an `IntTuples` mapper.
    MapIntTuplesParameter,
    /// De Bruijn sentinel used only while comparing `MapIntTuples` lambdas.
    NormalizedMapIntTuplesParameter,
}

impl QuantifiedOrigin {
    pub fn synthetic() -> Self {
        Self::Synthetic { is_self: false }
    }
}

/// The definition that a quantified belongs to, named by its index in the module rather than by
/// a source position, so that an edit that only moves code keeps every identity. Indices follow
/// source order, so inserting a definition renumbers the definitions of the same kind after it,
/// which also changes their `Class` or function identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantifiedAnchor {
    /// A quantified that no definition owns: a synthetic binder whose index alone tells it apart
    /// from the others with the same origin, or a legacy type variable that solving promotes in
    /// a closed `Forall`.
    None,
    Class(ClassDefIndex),
    Function(FuncDefIndex),
    TypeAlias(TypeAliasIndex),
    /// The parameter of a type-level lambda, by its `LambdaParamId`.
    LambdaParameter(u32),
}

/// A definition plus an index that disambiguates the quantifieds of the same origin that it owns,
/// such as the position of a type parameter in its definition's list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnchorIndex {
    pub anchor: QuantifiedAnchor,
    pub index: u32,
}

impl AnchorIndex {
    pub fn new(anchor: QuantifiedAnchor, index: u32) -> Self {
        Self { anchor, index }
    }

    pub fn first(anchor: QuantifiedAnchor) -> Self {
        Self { anchor, index: 0 }
    }
}

/// Deterministic identity for a `Quantified`, derived from the definition that owns it. Two
/// quantifieds are the same iff their identity is the same.
///
/// Globally unique because `module` distinguishes cross-module collisions, `anchor` pins the
/// owning definition and index, and `origin` prevents collisions between quantifieds of different
/// kinds at the same anchor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantifiedIdentity {
    pub module: ModuleName,
    pub anchor: AnchorIndex,
    pub origin: QuantifiedOrigin,
}

impl PartialOrd for QuantifiedIdentity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QuantifiedIdentity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.module
            .cmp(&other.module)
            .then_with(|| self.anchor.cmp(&other.anchor))
            .then_with(|| self.origin.cmp(&other.origin))
    }
}

impl QuantifiedIdentity {
    pub fn new(module: ModuleName, anchor: AnchorIndex, origin: QuantifiedOrigin) -> Self {
        Self {
            module,
            anchor,
            origin,
        }
    }
}

// None of these types contain Type values; they are visit leaves.
impl<To> Visit<To> for QuantifiedOrigin {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}
impl<To> VisitMut<To> for QuantifiedOrigin {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}
impl<To> Visit<To> for AnchorIndex {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}
impl<To> VisitMut<To> for AnchorIndex {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}
impl<To> Visit<To> for QuantifiedIdentity {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}
impl<To> VisitMut<To> for QuantifiedIdentity {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}

#[derive(Debug, Clone, Eq)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Quantified {
    /// Deterministic identity based on source location.
    identity: QuantifiedIdentity,
    pub name: Name,
    pub kind: QuantifiedKind,
    pub default: Option<Type>,
    pub restriction: Restriction,
    /// The *declared* variance of this type parameter, as specified by the user.
    /// For function type parameters, variance has no meaning.
    /// We store it here for convenience of our variance inference and checking
    /// infrastructure so it can directly read it from the type.
    variance: PreInferenceVariance,
    /// Qualified owner, e.g. `"mod.func"`, set for function type params to enable
    /// disambiguation in display (e.g. `T@mod.func`).
    pub owner: Option<Name>,
    /// Set when this must either be declared by an enclosing callable or replaced with its
    /// gradual fallback at a type boundary.
    pub needs_finalization: bool,
}

impl Quantified {
    pub fn with_restriction(self, restriction: Restriction) -> Self {
        Self {
            restriction,
            ..self
        }
    }

    pub fn with_needs_finalization(self) -> Self {
        Self {
            needs_finalization: true,
            ..self
        }
    }

    /// Return this parameter without its declared default.
    pub fn without_default(mut self) -> Self {
        self.default = None;
        self
    }
}

impl PartialEq for Quantified {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Hash for Quantified {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

impl Ord for Quantified {
    fn cmp(&self, other: &Self) -> Ordering {
        // Identity is fully deterministic, so sort on it directly.
        // This is used to order types in a union (and deduplication relies on Ord).
        self.identity.cmp(&other.identity)
    }
}

/// A quantified's identity holds no class, so `TypeOrd` is `Ord`.
impl TypeOrd for Quantified {}

impl PartialOrd for Quantified {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[derive(Visit, VisitMut, TypeEq, TypeOrd)]
pub enum QuantifiedKind {
    TypeVar,
    IntVar,
    ParamSpec,
    TypeVarTuple,
}

impl QuantifiedKind {
    fn empty_value(self) -> Type {
        match self {
            QuantifiedKind::TypeVar => Type::any_implicit(),
            QuantifiedKind::IntVar => gradual_size(),
            QuantifiedKind::ParamSpec => Type::Ellipsis,
            QuantifiedKind::TypeVarTuple => Type::any_tuple(),
        }
    }

    fn class_type(self, stdlib: &Stdlib) -> &ClassType {
        match self {
            QuantifiedKind::TypeVar | QuantifiedKind::IntVar => stdlib.type_var(),
            QuantifiedKind::ParamSpec => stdlib.param_spec(),
            QuantifiedKind::TypeVarTuple => stdlib.type_var_tuple(),
        }
    }
}

impl Display for QuantifiedIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@{:?}[{}]:{:?}",
            self.module, self.anchor.anchor, self.anchor.index, self.origin,
        )
    }
}

/// A legacy type variable, declared with `TypeVar(...)`, `ParamSpec(...)`, or `TypeVarTuple(...)`.
pub trait TypeVarLike {
    fn qname(&self) -> &QName;
    fn kind(&self) -> QuantifiedKind;
    fn default(&self) -> Option<&Type>;
    fn restriction(&self) -> &Restriction;
    fn variance(&self) -> PreInferenceVariance;
}

impl Quantified {
    pub fn new(
        identity: QuantifiedIdentity,
        name: Name,
        kind: QuantifiedKind,
        default: Option<Type>,
        restriction: Restriction,
        variance: PreInferenceVariance,
    ) -> Self {
        Quantified {
            identity,
            name,
            kind,
            default,
            restriction,
            variance,
            owner: None,
            needs_finalization: false,
        }
    }

    pub fn with_owner(mut self, owner: Name) -> Self {
        self.owner = Some(owner);
        self
    }

    pub fn type_var(
        name: Name,
        identity: QuantifiedIdentity,
        default: Option<Type>,
        restriction: Restriction,
        variance: PreInferenceVariance,
    ) -> Self {
        Self::new(
            identity,
            name,
            QuantifiedKind::TypeVar,
            default,
            restriction,
            variance,
        )
    }

    pub fn from_type_var_like(x: &dyn TypeVarLike, identity: QuantifiedIdentity) -> Self {
        Self::new(
            identity,
            x.qname().id().clone(),
            x.kind(),
            x.default().cloned(),
            x.restriction().clone(),
            x.variance(),
        )
    }

    pub fn to_type(self, heap: &TypeHeap) -> Type {
        heap.mk_quantified(self)
    }

    pub fn to_value(self) -> Type {
        Type::QuantifiedValue(Box::new(self))
    }

    pub fn class_type<'a>(&self, stdlib: &'a Stdlib) -> &'a ClassType {
        self.kind.class_type(stdlib)
    }

    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn kind(&self) -> QuantifiedKind {
        self.kind
    }

    pub fn default(&self) -> Option<&Type> {
        self.default.as_ref()
    }

    pub fn restriction(&self) -> &Restriction {
        &self.restriction
    }

    /// Display this type parameter name with the `*`/`**` kind prefix
    /// (e.g. `*Ts` for TypeVarTuple, `**P` for ParamSpec).
    pub fn display_name_with_prefix(&self) -> impl Display + '_ {
        Fmt(move |f| {
            if self.is_param_spec() {
                write!(f, "**")?;
            } else if self.is_type_var_tuple() {
                write!(f, "*")?;
            }
            write!(f, "{}", self.name)
        })
    }

    /// Display this type parameter with its bounds/constraints and default,
    /// in the format used for type parameter lists (e.g. `T: int = str`).
    pub fn display_with_bounds<'a>(&'a self, env: &'a dyn TypeEnv) -> impl Display + 'a {
        Fmt(move |f| {
            write!(f, "{}", self.display_name_with_prefix())?;
            match self.restriction() {
                Restriction::Bound(t) => write!(f, ": {}", t.display_with(env))?,
                Restriction::Constraints(ts) => {
                    write!(f, ": (")?;
                    for (i, t) in ts.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", t.display_with(env))?;
                    }
                    write!(f, ")")?;
                }
                Restriction::ShapeExtension(extension) => write!(f, ": {extension}")?,
                Restriction::Unrestricted => {}
            }
            if let Some(default) = self.default() {
                write!(f, " = {}", default.display_with(env))?;
            }
            Ok(())
        })
    }

    pub fn variance(&self) -> PreInferenceVariance {
        self.variance
    }

    pub fn name_in<'a, E: TypeEnv + ?Sized>(&'a self, _env: &'a E) -> &'a Name {
        &self.name
    }

    pub fn restriction_in<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a Restriction {
        env.type_var_restriction(self)
    }

    pub fn default_in<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> Option<&'a Type> {
        env.type_var_default(self)
    }

    pub fn variance_in<E: TypeEnv + ?Sized>(&self, _env: &E) -> PreInferenceVariance {
        self.variance
    }

    /// The restriction stored in this `Quantified`, which `TypeEnv::type_var_restriction`
    /// returns by default.
    pub(crate) fn inline_restriction(&self) -> &Restriction {
        &self.restriction
    }

    /// The default stored in this `Quantified`, which `TypeEnv::type_var_default` returns by
    /// default.
    pub(crate) fn inline_default(&self) -> Option<&Type> {
        self.default.as_ref()
    }

    pub fn is_type_var(&self) -> bool {
        matches!(self.kind, QuantifiedKind::TypeVar | QuantifiedKind::IntVar)
    }

    pub fn is_param_spec(&self) -> bool {
        matches!(self.kind, QuantifiedKind::ParamSpec)
    }

    pub fn is_type_var_tuple(&self) -> bool {
        matches!(self.kind, QuantifiedKind::TypeVarTuple)
    }

    pub fn identity(&self) -> &QuantifiedIdentity {
        &self.identity
    }

    pub(crate) fn normalized_map_int_tuples_parameter_depth(&self) -> Option<u32> {
        (self.identity.origin == QuantifiedOrigin::NormalizedMapIntTuplesParameter)
            .then_some(self.identity.anchor.index)
    }

    pub fn as_gradual_type_helper(kind: QuantifiedKind, default: Option<&Type>) -> Type {
        default.map_or_else(
            || kind.empty_value(),
            |default| {
                default.clone().transform(&mut |default| match default {
                    Type::TypeVar(t) => {
                        *default =
                            Self::as_gradual_type_helper(QuantifiedKind::TypeVar, t.default())
                    }
                    Type::TypeVarTuple(t) => {
                        *default =
                            Self::as_gradual_type_helper(QuantifiedKind::TypeVarTuple, t.default())
                    }
                    Type::ParamSpec(p) => {
                        *default =
                            Self::as_gradual_type_helper(QuantifiedKind::ParamSpec, p.default())
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
        Self::as_gradual_type_helper(self.kind(), self.default())
    }
}
