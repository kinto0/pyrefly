/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::LazyLock;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::qname::QName;
use ruff_python_ast::name::Name;

use crate::annotation::Qualifier;
use crate::class::Class;
use crate::class::ClassType;
use crate::heap::TypeHeap;
use crate::read_only::ReadOnlyReason;
use crate::simplify;
use crate::stdlib::Stdlib;
use crate::types::Substitution;
use crate::types::TArgs;
use crate::types::Type;

#[derive(
    Clone, Debug, Visit, VisitMut, TypeEq, PartialEq, Eq, Hash, PartialOrd, Ord
)]
pub struct TypedDictField {
    pub ty: Type,
    pub required: bool,
    /// The reason this field is read-only. `None` indicates it is read-write.
    pub read_only_reason: Option<ReadOnlyReason>,
}

impl TypedDictField {
    /// Check if this field is read-only.
    pub fn is_read_only(&self) -> bool {
        self.read_only_reason.is_some()
    }

    pub fn substitute_with(self, substitution: &Substitution) -> Self {
        Self {
            ty: substitution.substitute_into(self.ty),
            required: self.required,
            read_only_reason: self.read_only_reason,
        }
    }
}

#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub struct TypedDictInner {
    class: Class,
    args: TArgs,
}

impl TypedDictInner {
    pub fn new(class: Class, args: TArgs) -> Self {
        Self { class, args }
    }

    pub fn qname(&self) -> &QName {
        self.class.qname()
    }

    pub fn name(&self) -> &Name {
        self.class.name()
    }

    pub fn class_object(&self) -> &Class {
        &self.class
    }

    pub fn targs(&self) -> &TArgs {
        &self.args
    }

    pub fn targs_mut(&mut self) -> &mut TArgs {
        &mut self.args
    }

    /// A declared TypedDict is described by exactly the class and type arguments that make up
    /// a `ClassType`, so the two are freely interconvertible.
    pub fn as_class_type(&self) -> ClassType {
        ClassType::new(self.class.dupe(), self.args.clone())
    }

    pub fn from_class_type(class_type: &ClassType) -> Self {
        Self::new(class_type.class_object().dupe(), class_type.targs().clone())
    }

    pub fn to_type(self, heap: &TypeHeap) -> Type {
        heap.mk_typed_dict(self)
    }

    /// Display label for error messages.
    pub fn label(&self) -> String {
        format!("TypedDict `{}`", self.name())
    }
}

#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub struct AnonymousTypedDictInner {
    pub fields: Vec<(Name, TypedDictField)>,
}

impl AnonymousTypedDictInner {
    /// Compute the union of all field value types. This is derived from `fields`
    /// rather than stored, to avoid duplicating the type tree at each nesting
    /// level (which caused 2^N memory growth for nested dict literals).
    pub fn compute_value_type(&self, heap: &TypeHeap) -> Type {
        let tys: Vec<Type> = self.fields.iter().map(|(_, f)| f.ty.clone()).collect();
        simplify::unions(tys, heap)
    }

    pub fn field(&self, name: &Name) -> Option<&TypedDictField> {
        self.fields
            .iter()
            .find_map(|(n, field)| if n == name { Some(field) } else { None })
    }

    pub fn to_type(self, heap: &TypeHeap) -> Type {
        heap.mk_anonymous_typed_dict(self)
    }
}

// An anonymous typed dict has no class to borrow a name from, so we need a name that we
// can borrow instead.
// This is a lazily initialized value, Name::new normally can't be used as the RHS of a static
pub static ANONYMOUS_TYPED_DICT: LazyLock<Name> = LazyLock::new(|| Name::new("<anonymous>"));

/// How does the TypedDict handle extra items? See https://peps.python.org/pep-0728.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq, Hash)]
pub enum ExtraItems {
    /// Default behavior when neither the `closed` nor `extra_items` keyword is specified.
    Default,
    /// Closed TypedDict that does not allow extra items. This is equivalent to
    /// Extra(ExtraItem {ty: Never, ..}) but is its own variant because of how common it is.
    Closed,
    /// The TypedDict allows extra items of the specified type and read-only-ness. Note that
    /// Extra(ExtraItem {ty: Never, ..}) (a closed TypedDict) is represented with a separate Closed variant.
    Extra(ExtraItem),
}

impl ExtraItems {
    pub fn extra(ty: Type, qualifiers: &[Qualifier]) -> Self {
        if ty.is_never() {
            Self::Closed
        } else {
            Self::Extra(ExtraItem {
                ty,
                read_only: qualifiers.iter().any(|q| q == &Qualifier::ReadOnly),
            })
        }
    }

    pub fn extra_item(&self, stdlib: &Stdlib) -> ExtraItem {
        match self {
            Self::Extra(extra) => extra.clone(),
            Self::Closed => ExtraItem {
                ty: Type::never(),
                read_only: false,
            },
            Self::Default => ExtraItem {
                ty: stdlib.object().clone().to_type(),
                read_only: true,
            },
        }
    }
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq, Hash)]
pub struct ExtraItem {
    pub ty: Type,
    pub read_only: bool,
}
