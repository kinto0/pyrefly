/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::attr::Attribute;
use crate::alt::attr::DescriptorBase;
use crate::alt::attr::NoAccessReason;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::binding::binding::ClassFieldInitialValue;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::dunder;
use crate::error::collector::ErrorCollector;
use crate::error::kind::ErrorKind;
use crate::error::style::ErrorStyle;
use crate::types::annotation::Annotation;
use crate::types::annotation::Qualifier;
use crate::types::callable::BoolKeywords;
use crate::types::callable::DataclassKeywords;
use crate::types::callable::FuncMetadata;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::literal::Lit;
use crate::types::typed_dict::TypedDictField;
use crate::types::types::BoundMethod;
use crate::types::types::BoundMethodType;
use crate::types::types::CalleeKind;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::SuperObj;
use crate::types::types::Type;

/// Correctly analyzing which attributes are visible on class objects, as well
/// as handling method binding correctly, requires distinguishing which fields
/// are assigned values in the class body.
#[derive(Clone, Debug, TypeEq, VisitMut, PartialEq, Eq)]
pub enum ClassFieldInitialization {
    /// If this is a dataclass field, BoolKeywords stores the field's dataclass
    /// flags (which are boolean options that control how fields behave).
    Class(Option<BoolKeywords>),
    Instance,
}

impl Display for ClassFieldInitialization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Class(_) => write!(f, "initialized in body"),
            Self::Instance => write!(f, "not initialized in body"),
        }
    }
}

impl ClassFieldInitialization {
    pub fn recursive() -> Self {
        ClassFieldInitialization::Class(None)
    }
}

/// Raw information about an attribute declared somewhere in a class. We need to
/// know whether it is initialized in the class body in order to determine
/// both visibility rules and whether method binding should be performed.
#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut)]
pub struct ClassField(ClassFieldInner);

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut)]
enum ClassFieldInner {
    // TODO(stroxler): We should refactor `ClassFieldInner` into enum cases; currently
    // the semantics are encoded ad-hoc into the fields of a large product which
    // has made hacking features relatively easy, but makes the code hard to read.
    Simple {
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        readonly: bool,
        // Descriptor getter method, if there is one. `None` indicates no getter.
        descriptor_getter: Option<Type>,
        // Descriptor setter method, if there is one. `None` indicates no setter.
        descriptor_setter: Option<Type>,
        is_function_without_return_annotation: bool,
    },
}

impl Display for ClassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ClassFieldInner::Simple {
                ty, initialization, ..
            } => write!(f, "@{ty} ({initialization})"),
        }
    }
}

impl ClassField {
    fn new(
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        readonly: bool,
        descriptor_getter: Option<Type>,
        descriptor_setter: Option<Type>,
        is_function_without_return_annotation: bool,
    ) -> Self {
        Self(ClassFieldInner::Simple {
            ty,
            annotation,
            initialization,
            readonly,
            descriptor_getter,
            descriptor_setter,
            is_function_without_return_annotation,
        })
    }

    /// Get the raw type. Only suitable for use in this module, this type may
    /// not correspond to the type of any actual operations on the attribute.
    fn raw_type(&self) -> &Type {
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => ty,
        }
    }

    pub fn new_synthesized(ty: Type) -> Self {
        ClassField(ClassFieldInner::Simple {
            ty,
            annotation: None,
            initialization: ClassFieldInitialization::Class(None),
            readonly: false,
            descriptor_getter: None,
            descriptor_setter: None,
            is_function_without_return_annotation: false,
        })
    }

    pub fn recursive() -> Self {
        Self(ClassFieldInner::Simple {
            ty: Type::any_implicit(),
            annotation: None,
            initialization: ClassFieldInitialization::recursive(),
            readonly: false,
            descriptor_getter: None,
            descriptor_setter: None,
            is_function_without_return_annotation: false,
        })
    }

    fn initialization(&self) -> ClassFieldInitialization {
        match &self.0 {
            ClassFieldInner::Simple { initialization, .. } => initialization.clone(),
        }
    }

    fn instantiate_for(&self, cls: &ClassType) -> Self {
        match &self.0 {
            ClassFieldInner::Simple {
                ty,
                annotation,
                initialization,
                readonly,
                descriptor_getter,
                descriptor_setter,
                is_function_without_return_annotation,
            } => Self(ClassFieldInner::Simple {
                ty: cls.instantiate_member(ty.clone()),
                annotation: annotation.clone(),
                initialization: initialization.clone(),
                readonly: *readonly,
                descriptor_getter: descriptor_getter
                    .as_ref()
                    .map(|ty| cls.instantiate_member(ty.clone())),
                descriptor_setter: descriptor_setter
                    .as_ref()
                    .map(|ty| cls.instantiate_member(ty.clone())),
                is_function_without_return_annotation: *is_function_without_return_annotation,
            }),
        }
    }

    pub fn as_param(self, name: &Name, default: bool, kw_only: bool) -> Param {
        let ClassField(ClassFieldInner::Simple { ty, .. }) = self;
        let required = match default {
            true => Required::Optional,
            false => Required::Required,
        };
        if kw_only {
            Param::KwOnly(name.clone(), ty, required)
        } else {
            Param::Pos(name.clone(), ty, required)
        }
    }

    fn depends_on_class_type_parameter(&self, cls: &Class) -> bool {
        let tparams = cls.tparams();
        let mut qs = SmallSet::new();
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => ty.collect_quantifieds(&mut qs),
        };
        tparams.quantified().any(|q| qs.contains(&q))
    }

    fn as_raw_special_method_type(self, cls: &ClassType) -> Option<Type> {
        match self.instantiate_for(cls).0 {
            ClassFieldInner::Simple { ty, .. } => match self.initialization() {
                ClassFieldInitialization::Class(_) => Some(ty),
                ClassFieldInitialization::Instance => None,
            },
        }
    }

    fn as_special_method_type(self, cls: &ClassType) -> Option<Type> {
        self.as_raw_special_method_type(cls)
            .and_then(|ty| make_bound_method(cls, &ty))
    }

    pub fn as_named_tuple_type(&self) -> Type {
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => ty.clone(),
        }
    }

    pub fn as_named_tuple_requiredness(&self) -> Required {
        match &self.0 {
            ClassFieldInner::Simple {
                initialization: ClassFieldInitialization::Class(_),
                ..
            } => Required::Optional,
            ClassFieldInner::Simple {
                initialization: ClassFieldInitialization::Instance,
                ..
            } => Required::Required,
        }
    }

    pub fn as_typed_dict_field_info(self, required_by_default: bool) -> Option<TypedDictField> {
        match &self.0 {
            ClassFieldInner::Simple {
                annotation:
                    Some(Annotation {
                        ty: Some(ty),
                        qualifiers,
                    }),
                ..
            } => Some(TypedDictField {
                ty: ty.clone(),
                read_only: qualifiers.contains(&Qualifier::ReadOnly),
                required: if qualifiers.contains(&Qualifier::Required) {
                    true
                } else if qualifiers.contains(&Qualifier::NotRequired) {
                    false
                } else {
                    required_by_default
                },
            }),
            _ => None,
        }
    }

    pub fn as_enum_member(self, enum_cls: &Class) -> Option<Lit> {
        match self.0 {
            ClassFieldInner::Simple {
                ty: Type::Literal(lit),
                ..
            } if matches!(&lit, Lit::Enum(box (lit_cls, ..)) if lit_cls.class_object() == enum_cls) => {
                Some(lit)
            }
            _ => None,
        }
    }

    pub fn is_dataclass_kwonly_marker(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { ty, .. } => {
                matches!(ty, Type::ClassType(cls) if cls.class_object().has_qname("dataclasses", "KW_ONLY"))
            }
        }
    }

    pub fn is_class_var(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { annotation, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_class_var())
            }
        }
    }

    pub fn is_final(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { annotation, ty, .. } => {
                annotation.as_ref().is_some_and(|ann| ann.is_final()) || ty.has_final_decoration()
            }
        }
    }

    pub fn has_explicit_annotation(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple { annotation, .. } => annotation.is_some(),
        }
    }

    pub fn is_function_without_return_annotation(&self) -> bool {
        match &self.0 {
            ClassFieldInner::Simple {
                is_function_without_return_annotation,
                ..
            } => *is_function_without_return_annotation,
        }
    }

    pub fn dataclass_flags_of(&self, kw_only: bool) -> Option<BoolKeywords> {
        match &self.0 {
            ClassFieldInner::Simple {
                initialization,
                annotation,
                ..
            } => {
                if let Some(annot) = annotation
                    && annot.qualifiers.contains(&Qualifier::ClassVar)
                {
                    return None; // Class variables are not dataclass fields
                }
                let mut flags = match initialization {
                    ClassFieldInitialization::Class(Some(field_flags)) => field_flags.clone(),
                    ClassFieldInitialization::Class(None) => {
                        let mut kws = BoolKeywords::new();
                        kws.set(DataclassKeywords::DEFAULT.0, true);
                        kws
                    }
                    ClassFieldInitialization::Instance => BoolKeywords::new(),
                };
                if kw_only {
                    flags.set(DataclassKeywords::KW_ONLY.0, true);
                }
                Some(flags)
            }
        }
    }
}

pub fn bind_class_attribute(cls: &Class, attr: Type) -> Attribute {
    Attribute::read_write(make_bound_classmethod(cls, &attr).unwrap_or(attr))
}

fn make_bound_method_helper(
    obj: Type,
    attr: &Type,
    should_bind: &dyn Fn(&FuncMetadata) -> bool,
) -> Option<Type> {
    let func = match attr {
        Type::Forall(box Forall {
            tparams,
            body: Forallable::Function(func),
        }) if should_bind(&func.metadata) => Some(BoundMethodType::Forall(Forall {
            tparams: tparams.clone(),
            body: func.clone(),
        })),
        Type::Function(box func) if should_bind(&func.metadata) => {
            Some(BoundMethodType::Function(func.clone()))
        }
        Type::Overload(overload) if should_bind(&overload.metadata) => {
            Some(BoundMethodType::Overload(overload.clone()))
        }
        _ => None,
    };
    func.map(|func| Type::BoundMethod(Box::new(BoundMethod { obj, func })))
}

fn make_bound_classmethod(cls: &Class, attr: &Type) -> Option<Type> {
    let should_bind = |meta: &FuncMetadata| meta.flags.is_classmethod;
    make_bound_method_helper(Type::ClassDef(cls.dupe()), attr, &should_bind)
}

fn make_bound_method(cls: &ClassType, attr: &Type) -> Option<Type> {
    let should_bind =
        |meta: &FuncMetadata| !meta.flags.is_staticmethod && !meta.flags.is_classmethod;
    make_bound_method_helper(cls.instance_type(), attr, &should_bind)
}

fn bind_instance_attribute(
    cls: &ClassType,
    attr: Type,
    is_class_var: bool,
    readonly: bool,
) -> Attribute {
    // Decorated objects are methods, so they can't be ClassVars
    match attr {
        _ if attr.is_property_getter() => Attribute::property(
            make_bound_method(cls, &attr).unwrap_or(attr),
            None,
            cls.class_object().dupe(),
        ),
        _ if let Some(getter) = attr.is_property_setter_with_getter() => Attribute::property(
            make_bound_method(cls, &getter).unwrap_or(getter),
            Some(make_bound_method(cls, &attr).unwrap_or(attr)),
            cls.class_object().dupe(),
        ),
        attr if is_class_var || readonly => {
            Attribute::read_only(make_bound_method(cls, &attr).unwrap_or(attr))
        }
        attr => {
            Attribute::read_write(make_bound_method(cls, &attr).unwrap_or_else(|| {
                make_bound_classmethod(cls.class_object(), &attr).unwrap_or(attr)
            }))
        }
    }
}

/// Result of looking up a member of a class in the MRO, including a handle to the defining
/// class which may be some ancestor.
///
/// For example, given `class A: x: int; class B(A): pass`, the defining class
/// for attribute `x` is `A` even when `x` is looked up on `B`.
#[derive(Debug)]
pub(in crate::alt::class) struct WithDefiningClass<T> {
    pub value: T,
    pub defining_class: Class,
}

impl<T> WithDefiningClass<T> {
    pub(in crate::alt::class) fn defined_on(&self, cls: &Class) -> bool {
        self.defining_class == *cls
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn calculate_class_field(
        &self,
        name: &Name,
        value_ty: &Type,
        annotation: Option<&Annotation>,
        initial_value: &ClassFieldInitialValue,
        class: &Class,
        is_function_without_return_annotation: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> ClassField {
        let metadata = self.get_metadata_for_class(class);
        let initialization = self.get_class_field_initialization(&metadata, initial_value);

        // Ban typed dict from containing values; fields should be annotation-only.
        // TODO(stroxler): we ought to look into this more: class-level attributes make sense on a `TypedDict` class;
        // the typing spec does not explicitly define whether this is permitted.
        if metadata.is_typed_dict() && matches!(initialization, ClassFieldInitialization::Class(_))
        {
            self.error(
                errors,
                range,
                ErrorKind::BadClassDefinition,
                None,
                format!("TypedDict item `{}` may not be initialized", name),
            );
        }
        if metadata.is_typed_dict() || metadata.is_named_tuple() {
            for q in &[Qualifier::Final, Qualifier::ClassVar] {
                if annotation.is_some_and(|ann| ann.has_qualifier(q)) {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidAnnotation,
                        None,
                        format!(
                            "`{}` may not be used for TypedDict or NamedTuple members",
                            q
                        ),
                    );
                }
            }
        }
        if !metadata.is_typed_dict() {
            for q in &[
                Qualifier::Required,
                Qualifier::NotRequired,
                Qualifier::ReadOnly,
            ] {
                if annotation.is_some_and(|ann| ann.has_qualifier(q)) {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidAnnotation,
                        None,
                        format!("`{}` may only be used for TypedDict members", q),
                    );
                }
            }
        }

        // Determine whether this is an explicit `@override`.
        let is_override = value_ty.is_override();

        // Promote literals. The check on `annotation` is an optimization, it does not (currently) affect semantics.
        // TODO(stroxler): if we see a read-only `Qualifier` like `Final`, it is sound to preserve literals.
        let value_ty = if annotation.is_none_or(|a| a.ty.is_none()) && value_ty.is_literal() {
            value_ty.clone().promote_literals(self.stdlib)
        } else {
            value_ty.clone()
        };

        // Types provided in annotations shadow inferred types
        let ty = if let Some(ann) = annotation {
            match &ann.ty {
                Some(ty) => ty.clone(),
                None => value_ty.clone(),
            }
        } else {
            value_ty.clone()
        };

        let ty = match initial_value {
            ClassFieldInitialValue::Class(_) | ClassFieldInitialValue::Instance(None) => ty,
            ClassFieldInitialValue::Instance(Some(method_name)) => self
                .check_and_sanitize_method_scope_type_parameters(
                    class,
                    method_name,
                    ty,
                    name,
                    range,
                    errors,
                ),
        };

        // Enum handling:
        // - Check whether the field is a member (which depends only on its type and name)
        // - Validate that a member should not have an annotation, and should respect any explicit annotation on `_value_`
        //
        // TODO(stroxler, yangdanny): We currently operate on promoted types, which means we do not infer `Literal[...]`
        // types for the `.value` / `._value_` attributes of literals. This is permitted in the spec although not optimal
        // for most cases; we are handling it this way in part because generic enum behavior is not yet well-specified.
        let ty = if let Some(enum_) = metadata.enum_metadata()
            && self.is_valid_enum_member(name, &ty, &initialization)
        {
            if annotation.is_some() {
                self.error(
                    errors, range,ErrorKind::InvalidAnnotation, None,
                    format!("Enum member `{}` may not be annotated directly. Instead, annotate the _value_ attribute.", name),
                );
            }
            if let Some(enum_value_ty) = self.type_of_enum_value(enum_) {
                if !matches!(ty, Type::Tuple(_))
                    && !self
                        .solver()
                        .is_subset_eq(&ty, &enum_value_ty, self.type_order())
                {
                    self.error(
                        errors, range, ErrorKind::BadAssignment, None,
                        format!("The value for enum member `{}` must match the annotation of the _value_ attribute", name), 
                    );
                }
            }
            Type::Literal(Lit::Enum(Box::new((
                enum_.cls.clone(),
                name.clone(),
                ty.clone(),
            ))))
        } else {
            ty
        };

        // TODO: handle other kinds of readonlyness
        let is_namedtuple_member = metadata
            .named_tuple_metadata()
            .is_some_and(|named_tuple| named_tuple.elements.contains(name));
        let is_frozen_dataclass_field = metadata.dataclass_metadata().is_some_and(|dataclass| {
            dataclass.kws.is_set(&DataclassKeywords::FROZEN) && dataclass.fields.contains(name)
        });
        let readonly = is_namedtuple_member || is_frozen_dataclass_field;

        // Identify whether this is a descriptor
        let (mut descriptor_getter, mut descriptor_setter) = (None, None);
        match &ty {
            // TODO(stroxler): This works for simple descriptors. There three known gaps, there may be others:
            // - If the field is instance-only, descriptor dispatching won't occur, an instance-only attribute
            //   that happens to be a descriptor just behaves like a normal instance-only attribute.
            // - Gracefully handle instance-only `__get__`/`__set__`. Descriptors only seem to be detected
            //   when the descriptor attribute is initialized on the class body of the descriptor.
            // - Do we care about distributing descriptor behavior over unions? If so, what about the case when
            //   the raw class field is a union of a descriptor and a non-descriptor? Do we want to allow this?
            Type::ClassType(c) => {
                if c.class_object().contains(&dunder::GET) {
                    descriptor_getter =
                        Some(self.attr_infer(&ty, &dunder::GET, range, errors, None));
                }
                if c.class_object().contains(&dunder::SET) {
                    descriptor_setter =
                        Some(self.attr_infer(&ty, &dunder::SET, range, errors, None));
                }
            }
            _ => {}
        };

        // Create the resulting field and check for override inconsistencies before returning
        let class_field = ClassField::new(
            ty,
            annotation.cloned(),
            initialization,
            readonly,
            descriptor_getter,
            descriptor_setter,
            is_function_without_return_annotation,
        );
        self.check_class_field_for_override_mismatch(
            name,
            &class_field,
            class,
            is_override,
            range,
            errors,
        );
        class_field
    }

    fn get_class_field_initialization(
        &self,
        metadata: &ClassMetadata,
        initial_value: &ClassFieldInitialValue,
    ) -> ClassFieldInitialization {
        match initial_value {
            ClassFieldInitialValue::Instance(_) => ClassFieldInitialization::Instance,
            ClassFieldInitialValue::Class(None) => ClassFieldInitialization::Class(None),
            ClassFieldInitialValue::Class(Some(e)) => {
                // If this field was created via a call to a dataclass field specifier, extract field flags from the call.
                if metadata.dataclass_metadata().is_some()
                    && let Expr::Call(ExprCall {
                        range: _,
                        func,
                        arguments: Arguments { keywords, .. },
                    }) = e
                {
                    // We already type-checked this expression as part of computing the type for the ClassField,
                    // so we can ignore any errors encountered here.
                    let ignore_errors =
                        ErrorCollector::new(self.module_info().dupe(), ErrorStyle::Never);
                    let func_ty = self.expr_infer(func, &ignore_errors);
                    if matches!(
                        func_ty.callee_kind(),
                        Some(CalleeKind::Function(FunctionKind::DataclassField))
                    ) {
                        let mut flags = BoolKeywords::new();
                        for kw in keywords {
                            if let Some(id) = &kw.arg
                                && (id.id == DataclassKeywords::DEFAULT.0
                                    || id.id == "default_factory")
                            {
                                flags.set(DataclassKeywords::DEFAULT.0, true);
                            } else {
                                let val = self.expr_infer(&kw.value, &ignore_errors);
                                flags.set_keyword(kw.arg.as_ref(), val);
                            }
                        }
                        ClassFieldInitialization::Class(Some(flags))
                    } else {
                        ClassFieldInitialization::Class(None)
                    }
                } else {
                    ClassFieldInitialization::Class(None)
                }
            }
        }
    }

    fn check_and_sanitize_method_scope_type_parameters(
        &self,
        class: &Class,
        method_name: &Name,
        ty: Type,
        name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let mut qs = SmallSet::new();
        ty.collect_quantifieds(&mut qs);
        if let Some(method_field) =
            self.get_non_synthesized_field_from_current_class_only(class, method_name)
        {
            match &method_field.raw_type() {
                Type::Forall(box Forall { tparams, .. }) => {
                    let gradual_fallbacks: SmallMap<_, _> = tparams
                        .iter()
                        .filter_map(|param| {
                            let q = &param.quantified;
                            if qs.contains(q) {
                                self.error(
                                    errors,
                                    range,
                                    ErrorKind::InvalidTypeVar,
                            None,
                                format!(
                                        "Cannot initialize attribute `{}` to a value that depends on method-scoped type variable `{}`",
                                        name,
                                        &param.name,
                                    ),
                                );
                                Some((*q, q.as_gradual_type()))
                            } else {
                                None
                            }
                        })
                        .collect();
                    ty.subst(&gradual_fallbacks)
                }
                _ => ty,
            }
        } else {
            ty
        }
    }

    fn as_instance_attribute(&self, field: ClassField, cls: &ClassType) -> Attribute {
        match field.instantiate_for(cls).0 {
            // TODO(stroxler): Clean up this match by making `ClassFieldInner` an
            // enum; the match is messy
            ClassFieldInner::Simple {
                ty,
                descriptor_getter,
                descriptor_setter,
                ..
            } if descriptor_getter.is_some() || descriptor_setter.is_some() => {
                Attribute::descriptor(
                    ty,
                    DescriptorBase::Instance(cls.clone()),
                    descriptor_getter,
                    descriptor_setter,
                )
            }
            ClassFieldInner::Simple {
                ty,
                readonly,
                annotation,
                ..
            } => {
                let is_class_var = annotation.is_some_and(|ann| ann.is_class_var());
                match field.initialization() {
                    ClassFieldInitialization::Class(_) => {
                        bind_instance_attribute(cls, ty, is_class_var, readonly)
                    }
                    ClassFieldInitialization::Instance if readonly || is_class_var => {
                        Attribute::read_only(ty)
                    }
                    ClassFieldInitialization::Instance => Attribute::read_write(ty),
                }
            }
        }
    }

    fn as_class_attribute(&self, field: ClassField, cls: &Class) -> Attribute {
        match &field.0 {
            ClassFieldInner::Simple {
                ty,
                descriptor_getter,
                descriptor_setter,
                ..
            } if descriptor_getter.is_some() || descriptor_setter.is_some() => {
                Attribute::descriptor(
                    ty.clone(),
                    DescriptorBase::ClassDef(cls.dupe()),
                    descriptor_getter.clone(),
                    descriptor_setter.clone(),
                )
            }
            ClassFieldInner::Simple {
                initialization: ClassFieldInitialization::Instance,
                ..
            } => Attribute::no_access(NoAccessReason::ClassUseOfInstanceAttribute(cls.dupe())),
            ClassFieldInner::Simple {
                initialization: ClassFieldInitialization::Class(_),
                ty,
                ..
            } => {
                if field.depends_on_class_type_parameter(cls) {
                    Attribute::no_access(NoAccessReason::ClassAttributeIsGeneric(cls.dupe()))
                } else {
                    bind_class_attribute(cls, ty.clone())
                }
            }
        }
    }

    fn check_class_field_for_override_mismatch(
        &self,
        name: &Name,
        class_field: &ClassField,
        class: &Class,
        is_override: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        let Type::ClassType(class_type) = class.instance_type() else {
            return;
        };
        let got_attr = self.as_instance_attribute(class_field.clone(), &class_type);
        let metadata = self.get_metadata_for_class(class);
        let parents = metadata.bases_with_metadata();
        let mut parent_attr_found = false;
        let mut parent_has_any = false;
        for (parent, parent_metadata) in parents {
            parent_has_any = parent_has_any || parent_metadata.has_base_any();
            // todo zeina: skip private properties and dunder methods for now. This will need some special casing.
            if name.starts_with('_') && name.ends_with('_') {
                continue;
            }
            if name.starts_with("__") && !name.ends_with("__") {
                continue;
            }
            let Some(want_member) = self.get_class_member(parent.class_object(), name) else {
                continue;
            };
            parent_attr_found = true;
            let want_class_field = Arc::unwrap_or_clone(want_member.value);
            if want_class_field.is_final() {
                self.error(
                    errors,
                    range,
                    ErrorKind::BadOverride,
                    None,
                    format!(
                        "`{}` is declared as final in parent class `{}`",
                        name,
                        parent.name()
                    ),
                );
                continue;
            }
            if want_class_field.has_explicit_annotation() && class_field.has_explicit_annotation() {
                let want_is_class_var = want_class_field.is_class_var();
                let got_is_class_var = class_field.is_class_var();
                if want_is_class_var && !got_is_class_var {
                    self.error(
                            errors,
                            range,
                            ErrorKind::BadOverride,
                            None,
                            format!(
                                "Instance variable `{}.{}` overrides ClassVar of the same name in parent class `{}`",
                                class.name(),
                                name,
                                parent.name()
                            ),
                        );
                    continue;
                } else if !want_is_class_var && got_is_class_var {
                    self.error(
                            errors,
                            range,
                            ErrorKind::BadOverride,
                            None,
                            format!(
                                "ClassVar `{}.{}` overrides instance variable of the same name in parent class `{}`",
                                class.name(),
                                name,
                                parent.name()
                            ),
                        );
                    continue;
                }
            }
            let want_attr = self.as_instance_attribute(want_class_field.clone(), parent);
            let attr_check = self.is_attr_subset(&got_attr, &want_attr, &mut |got, want| {
                self.solver().is_subset_eq(got, want, self.type_order())
            });
            if !attr_check {
                self.error(
                    errors,
                    range,
                    ErrorKind::BadOverride,
                    None,
                    format!(
                        "Class member `{}.{}` overrides parent class `{}` in an inconsistent manner",
                        class.name(),
                        name,
                        parent.name()
                    ),
                );
            }
        }
        if is_override && !parent_attr_found && !parent_has_any {
            self.error(
                    errors,
                    range,
                    ErrorKind::BadOverride,
                    None,
                    format!(
                        "Class member `{}.{}` is marked as an override, but no parent class has a matching attribute",
                        class.name(),
                        name,
                    ),
                );
        }
    }

    fn get_non_synthesized_field_from_current_class_only(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        if cls.contains(name) {
            let field = self.get_from_class(cls, &KeyClassField(cls.index(), name.clone()));
            Some(field)
        } else {
            None
        }
    }

    /// This function does not return fields defined in parent classes
    pub fn get_field_from_current_class_only(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<Arc<ClassField>> {
        if let Some(field) = self.get_non_synthesized_field_from_current_class_only(cls, name) {
            Some(field)
        } else {
            let synthesized_fields =
                self.get_from_class(cls, &KeyClassSynthesizedFields(cls.index()));
            let synth = synthesized_fields.get(name);
            synth.map(|f| f.inner.dupe())
        }
    }

    pub(in crate::alt::class) fn get_class_member(
        &self,
        cls: &Class,
        name: &Name,
    ) -> Option<WithDefiningClass<Arc<ClassField>>> {
        if let Some(field) = self.get_field_from_current_class_only(cls, name) {
            Some(WithDefiningClass {
                value: field,
                defining_class: cls.dupe(),
            })
        } else {
            self.get_metadata_for_class(cls)
                .ancestors(self.stdlib)
                .find_map(|ancestor| {
                    self.get_field_from_current_class_only(ancestor.class_object(), name)
                        .map(|field| WithDefiningClass {
                            value: Arc::new(field.instantiate_for(ancestor)),
                            defining_class: ancestor.class_object().dupe(),
                        })
                })
        }
    }

    pub fn get_instance_attribute(&self, cls: &ClassType, name: &Name) -> Option<Attribute> {
        self.get_class_member(cls.class_object(), name)
            .map(|member| self.as_instance_attribute(Arc::unwrap_or_clone(member.value), cls))
    }

    /// Looks up an attribute on a super instance.
    pub fn get_super_attribute(
        &self,
        lookup_cls: &ClassType,
        super_obj: &SuperObj,
        name: &Name,
    ) -> Option<Attribute> {
        let member = self.get_class_member(lookup_cls.class_object(), name);
        match super_obj {
            SuperObj::Instance(obj) => member
                .map(|member| self.as_instance_attribute(Arc::unwrap_or_clone(member.value), obj)),
            SuperObj::Class(obj) => member
                .map(|member| self.as_class_attribute(Arc::unwrap_or_clone(member.value), obj)),
        }
    }

    /// Gets an attribute from a class definition.
    ///
    /// Returns `None` if there is no such attribute, otherwise an `Attribute` object
    /// that describes whether access is allowed and the type if so.
    ///
    /// Access is disallowed for instance-only attributes and for attributes whose
    /// type contains a class-scoped type parameter - e.g., `class A[T]: x: T`.
    pub fn get_class_attribute(&self, cls: &Class, name: &Name) -> Option<Attribute> {
        self.get_class_member(cls, name)
            .map(|member| self.as_class_attribute(Arc::unwrap_or_clone(member.value), cls))
    }

    /// Get the class's `__new__` method.
    ///
    /// This lookup skips normal method binding logic (it behaves like a cross
    /// between a classmethod and a constructor; downstream code handles this
    /// using the raw callable type).
    pub fn get_dunder_new(&self, cls: &ClassType) -> Option<Type> {
        let new_member = self.get_class_member(cls.class_object(), &dunder::NEW)?;
        if new_member.defined_on(self.stdlib.object_class_type().class_object()) {
            // The default behavior of `object.__new__` is already baked into our implementation of
            // class construction; we only care about `__new__` if it is overridden.
            None
        } else {
            Arc::unwrap_or_clone(new_member.value).as_raw_special_method_type(cls)
        }
    }

    /// Get the class's `__init__` method. The second argument controls whether we return an inherited `object.__init__`.
    pub fn get_dunder_init(&self, cls: &ClassType, get_object_init: bool) -> Option<Type> {
        let init_method = self.get_class_member(cls.class_object(), &dunder::INIT)?;
        if get_object_init
            || !init_method.defined_on(self.stdlib.object_class_type().class_object())
        {
            Arc::unwrap_or_clone(init_method.value).as_special_method_type(cls)
        } else {
            None
        }
    }

    /// Get the metaclass `__call__` method
    pub fn get_metaclass_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        let metadata = self.get_metadata_for_class(cls.class_object());
        let metaclass = metadata.metaclass()?;
        let attr = self.get_class_member(metaclass.class_object(), &dunder::CALL)?;
        if attr.defined_on(self.stdlib.builtins_type().class_object()) {
            // The behavior of `type.__call__` is already baked into our implementation of constructors,
            // so we can skip analyzing it at the type level.
            None
        } else if attr.value.is_function_without_return_annotation() {
            // According to the typing spec:
            // If a custom metaclass __call__ method is present but does not have an annotated return type,
            // type checkers may assume that the method acts like type.__call__.
            // https://typing.python.org/en/latest/spec/constructors.html#converting-a-constructor-to-callable
            None
        } else {
            Arc::unwrap_or_clone(attr.value).as_special_method_type(metaclass)
        }
    }
}
