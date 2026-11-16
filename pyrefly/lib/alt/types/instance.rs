/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::literal::LitStyle;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::simplify::intersect;
use pyrefly_types::simplify::unions;
use pyrefly_types::stdlib::Stdlib;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Type;

use crate::alt::attr::ClassBase;
use crate::alt::class::class_field::DescriptorBase;
use crate::types::env::TypeEnv;

#[derive(Debug)]
pub enum InstanceKind<'a> {
    ClassType(&'a ClassType),
    TypedDict,
    TypeVar(Quantified),
    SelfType,
    Protocol(Type),
    Metaclass(ClassBase),
    LiteralString,
}

/// Wrapper to hold a specialized instance of a class , unifying ClassType and TypedDict.
#[derive(Debug)]
pub struct Instance<'a> {
    pub kind: InstanceKind<'a>,
    pub class: &'a Class,
    pub targs: &'a TArgs,
}

impl<'a> Instance<'a> {
    pub fn literal_string(stdlib: &'a Stdlib) -> Self {
        Self {
            kind: InstanceKind::LiteralString,
            class: stdlib.str().class_object(),
            targs: stdlib.str().targs(),
        }
    }

    pub fn of_class(cls: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::ClassType(cls),
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    pub fn of_typed_dict(td: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::TypedDict,
            class: td.class_object(),
            targs: td.targs(),
        }
    }

    pub fn of_type_var(q: Quantified, bound: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::TypeVar(q),
            class: bound.class_object(),
            targs: bound.targs(),
        }
    }

    pub fn of_self_type(cls: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::SelfType,
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    pub fn of_protocol(cls: &'a ClassType, self_type: Type) -> Self {
        Self {
            kind: InstanceKind::Protocol(self_type),
            class: cls.class_object(),
            targs: cls.targs(),
        }
    }

    pub fn of_metaclass(cls: ClassBase, metaclass: &'a ClassType) -> Self {
        Self {
            kind: InstanceKind::Metaclass(cls),
            class: metaclass.class_object(),
            targs: metaclass.targs(),
        }
    }

    /// Instantiate a type that is relative to the class type parameters
    /// by substituting in the type arguments.
    pub fn instantiate_member(&self, raw_member: &mut Type) {
        self.targs.substitute_into_mut(raw_member)
    }

    pub fn to_type(&self, heap: &TypeHeap, env: &dyn TypeEnv) -> Type {
        match &self.kind {
            InstanceKind::ClassType(cls) => heap.mk_class_type((*cls).clone()),
            InstanceKind::TypedDict => {
                heap.mk_class_type(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::TypeVar(q) => {
                let quantified_ty = q.clone().to_type(heap);
                let ub = match q.restriction() {
                    Restriction::Unrestricted => return quantified_ty,
                    Restriction::Bound(b) => b.clone(),
                    Restriction::Constraints(cs) => unions(cs.clone(), env, heap),
                    // Shape-extension restrictions need a Stdlib to materialize their ordinary
                    // upper bound. Preserve the quantified instead of inventing a fallback.
                    Restriction::ShapeExtension(_) => return quantified_ty,
                };
                let instance_ty =
                    heap.mk_class_type(ClassType::new(self.class.dupe(), self.targs.clone()));
                if instance_ty == ub {
                    quantified_ty
                } else {
                    intersect(
                        vec![quantified_ty, instance_ty.clone()],
                        instance_ty,
                        env,
                        heap,
                    )
                }
            }
            InstanceKind::SelfType => {
                heap.mk_self_type(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::Protocol(self_type) => self_type.clone(),
            InstanceKind::Metaclass(cls) => cls.clone().to_type(heap),
            InstanceKind::LiteralString => heap.mk_literal_string(LitStyle::Implicit),
        }
    }

    /// Looking up a classmethod/staticmethod from an instance base has class-like
    /// lookup behavior. When this happens, we convert from an instance base to a class base.
    pub fn to_class_base(&self) -> ClassBase {
        match &self.kind {
            InstanceKind::ClassType(cls) => ClassBase::ClassType((*cls).clone()),
            InstanceKind::SelfType => {
                ClassBase::SelfType(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::Protocol(self_type) => ClassBase::Protocol(
                ClassType::new(self.class.dupe(), self.targs.clone()),
                self_type.clone(),
            ),
            InstanceKind::TypeVar(q) => ClassBase::Quantified(
                q.clone(),
                ClassType::new(self.class.dupe(), self.targs.clone()),
            ),
            _ => ClassBase::ClassType(ClassType::new(self.class.dupe(), self.targs.clone())),
        }
    }

    pub fn to_descriptor_base(&self) -> DescriptorBase {
        match &self.kind {
            InstanceKind::SelfType => {
                DescriptorBase::SelfInstance(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
            InstanceKind::ClassType(cls) => DescriptorBase::Instance((*cls).clone()),
            // A descriptor is never constructed for a TypedDict field, which is a dict key
            // rather than an attribute, so `TypedDict` needs no special case here.
            // TODO(rechen): a descriptor in a TypedDict should be an error at class creation time.
            InstanceKind::TypedDict
            | InstanceKind::Protocol(..)
            | InstanceKind::Metaclass(..)
            | InstanceKind::TypeVar(..)
            | InstanceKind::LiteralString => {
                DescriptorBase::Instance(ClassType::new(self.class.dupe(), self.targs.clone()))
            }
        }
    }
}
