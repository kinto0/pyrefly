/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ord;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use dupe::Dupe;
use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::TypeOrd;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::docstring::DocRanges;
use pyrefly_python::module::Module;
use pyrefly_python::module_key::ModuleKey;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::qname::QName;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::env::TypeEnv;
use crate::equality::TypeEq;
use crate::equality::TypeEqCtx;
use crate::function::Deprecation;
use crate::known_class::KnownClass;
use crate::type_ord::TypeOrd;
use crate::type_ord::TypeOrdCtx;
use crate::types::Substitution;
use crate::types::TArgs;
use crate::types::TParams;
use crate::types::Type;

/// The name of a nominal type, e.g. `str`
#[derive(Debug, Clone, Dupe)]
pub struct Class(Arc<ClassDef>);

impl Hash for Class {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key_eq().hash(state)
    }
}

impl PartialEq for Class {
    fn eq(&self, other: &Self) -> bool {
        self.key_eq().eq(&other.key_eq())
    }
}

impl Eq for Class {}

/// Two classes are `TypeEq` when they are the same class (`PartialEq`, which with `Hash` stays
/// identity-only, since `Class` is a map key) with the same definition facts: the name and the
/// enclosing scopes that its printed name shows, whether it is a protocol, its deprecation, and
/// whether it declares type parameters. A reader can use these facts from the value alone, so
/// comparing an answer that holds the class catches a change to them. They are all decided at
/// binding time, so the iterations of a recursive computation always agree on them. `known` is
/// computed from the module, the name and the nesting, so it follows from those.
impl TypeEq for Class {
    fn type_eq(&self, other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        let same_name = self.0.qname.id() == other.0.qname.id()
            && match (
                self.0.qname.parent().is_toplevel(),
                other.0.qname.parent().is_toplevel(),
            ) {
                (true, true) => true,
                (false, false) => {
                    self.0.qname.name_relative_to_module()
                        == other.0.qname.name_relative_to_module()
                }
                _ => false,
            };
        self == other
            && same_name
            && self.0.deprecation == other.0.deprecation
            && self.0.is_protocol == other.0.is_protocol
            && self.0.has_tparams == other.0.has_tparams
    }
}

impl Ord for Class {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key_ord().cmp(&other.key_ord())
    }
}

impl PartialOrd for Class {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Classes are ordered by name, then by the position of their name in its module, both read
/// through the env, and then by `Ord`. `Ord` breaks the remaining ties, so this is a total order
/// even if two different classes read the same name and position.
impl TypeOrd for Class {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        // One allocation holds one definition, so both reads would return the same data.
        if Arc::ptr_eq(&self.0, &other.0) {
            return Ordering::Equal;
        }
        // `TypeEq` ignores where a class is defined, so moving a class within its file does not
        // recheck the modules that import it. A union there that holds two classes with the same
        // name keeps its old member order until it is rechecked.
        let (a, b) = (self.qname(ctx.env()), other.qname(ctx.env()));
        (a.id(), a.range().start(), a.range().end())
            .cmp(&(b.id(), b.range().start(), b.range().end()))
            .then_with(|| self.cmp(other))
    }
}

// There are types stored inside Class, in the TParams, but we don't want to visit them.
// We typically visit types to get rid of Var, and promise these are not interesting in that sense.
impl VisitMut<Type> for Class {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for Class {
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

/// `attr.ib`/`attrib` honor a `type=` argument and accept a positional default; `field` neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrsFieldSpecifierKind {
    Attrib,
    Field,
}

/// Bundling these with the specifier keeps them unrepresentable without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttrsFieldSpecifier {
    pub kind: AttrsFieldSpecifierKind,
    pub default_is_nothing: bool,
    pub default_decorator_method_range: Option<TextRange>,
    pub converter_decorator_method_range: Option<TextRange>,
}

/// Simple properties of class fields that can be attached to the class definition. Note that this
/// does not include the type of a field, which needs to be computed lazily to avoid a recursive loop.
#[derive(Debug, Clone)]
pub struct ClassFieldProperties {
    is_annotated: bool,
    // The field is initialized on the class (outside of a method)
    is_initialized_on_class: bool,
    // The field is defined in the class body (not in a method via self.x = ...)
    is_defined_in_class_body: bool,
    /// False if the field's definition alone shows that it is not an abstract method, so that
    /// checking for abstract members does not need its type.
    may_be_abstract: bool,
    attrs_field_specifier: Option<AttrsFieldSpecifier>,
    /// An explicit Django method-copy policy set in the class body.
    django_queryset_only: Option<bool>,
    range: TextRange,
    /// The documentation of this field.
    doc_ranges: DocRanges,
}

/// The index of a class within the file, used as a reference to data associated with the class.
#[derive(Debug, Clone, Dupe, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[derive(Display)]
pub struct ClassDefIndex(pub u32);

impl ClassFieldProperties {
    pub fn new(
        is_annotated: bool,
        has_default_value: bool,
        is_defined_in_class_body: bool,
        may_be_abstract: bool,
        attrs_field_specifier: Option<AttrsFieldSpecifier>,
        range: TextRange,
        doc_ranges: DocRanges,
    ) -> Self {
        Self {
            is_annotated,
            is_initialized_on_class: has_default_value,
            is_defined_in_class_body,
            may_be_abstract,
            attrs_field_specifier,
            django_queryset_only: None,
            range,
            doc_ranges,
        }
    }

    pub fn is_initialized_on_class(&self) -> bool {
        self.is_initialized_on_class
    }

    /// Attach the method-copy policy found during binding.
    pub fn with_django_queryset_only(mut self, value: Option<bool>) -> Self {
        self.django_queryset_only = value;
        self
    }
    pub fn is_defined_in_class_body(&self) -> bool {
        self.is_defined_in_class_body
    }

    /// Whether `self` and `other` declare a field the same way, wherever in the file they are.
    /// Locations are read only by the module that declares the field.
    fn same_declaration(&self, other: &Self) -> bool {
        let Self {
            is_annotated,
            is_initialized_on_class,
            is_defined_in_class_body,
            may_be_abstract,
            attrs_field_specifier,
            django_queryset_only,
            range: _,
            doc_ranges: _,
        } = self;
        *is_annotated == other.is_annotated
            && *is_initialized_on_class == other.is_initialized_on_class
            && *is_defined_in_class_body == other.is_defined_in_class_body
            && *may_be_abstract == other.may_be_abstract
            && *django_queryset_only == other.django_queryset_only
            && match (attrs_field_specifier, &other.attrs_field_specifier) {
                (None, None) => true,
                (Some(x), Some(y)) => {
                    let AttrsFieldSpecifier {
                        kind,
                        default_is_nothing,
                        default_decorator_method_range,
                        converter_decorator_method_range,
                    } = x;
                    *kind == y.kind
                        && *default_is_nothing == y.default_is_nothing
                        && default_decorator_method_range.is_some()
                            == y.default_decorator_method_range.is_some()
                        && converter_decorator_method_range.is_some()
                            == y.converter_decorator_method_range.is_some()
                }
                _ => false,
            }
    }
}

/// The set of fields declared on a class, with their properties.
#[derive(Clone, Debug, Default)]
pub struct ClassFields(SmallMap<Name, ClassFieldProperties>);

impl ClassFields {
    pub fn new(fields: SmallMap<Name, ClassFieldProperties>) -> Self {
        Self(fields)
    }

    pub fn empty() -> Self {
        Self(SmallMap::new())
    }

    /// Whether `self` and `other` declare the same fields, in the same order and the same way,
    /// wherever in the file they are declared.
    pub fn same_declarations(&self, other: &ClassFields) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|((n1, p1), (n2, p2))| n1 == n2 && p1.same_declaration(p2))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, name: &Name) -> bool {
        self.0.contains_key(name)
    }

    pub fn get_index_of(&self, name: &Name) -> Option<usize> {
        self.0.get_index_of(name)
    }

    pub fn fields(&self) -> impl ExactSizeIterator<Item = &Name> {
        self.0.keys()
    }

    /// Alias for `fields()` — iterates over field names.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &Name> {
        self.0.keys()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&Name, &ClassFieldProperties)> {
        self.0.iter()
    }

    pub fn class_body_fields(&self) -> impl Iterator<Item = &Name> {
        self.0
            .iter()
            .filter(|(_, prop)| prop.is_defined_in_class_body)
            .map(|(name, _)| name)
    }

    /// Whether the class defines `name` in a way that shows it is not an abstract method.
    pub fn defines_concrete_field(&self, name: &Name) -> bool {
        self.0.get(name).is_some_and(|prop| !prop.may_be_abstract)
    }

    pub fn is_field_annotated(&self, name: &Name) -> bool {
        self.0.get(name).is_some_and(|prop| prop.is_annotated)
    }

    /// Return the explicit Django method-copy policy, if present.
    pub fn django_queryset_only(&self, name: &Name) -> Option<bool> {
        self.0.get(name).and_then(|prop| prop.django_queryset_only)
    }

    pub fn is_attrs_field_specifier(&self, name: &Name) -> bool {
        self.0
            .get(name)
            .is_some_and(|prop| prop.attrs_field_specifier.is_some())
    }

    pub fn default_is_attrs_nothing(&self, name: &Name) -> bool {
        self.0.get(name).is_some_and(|prop| {
            prop.attrs_field_specifier
                .is_some_and(|s| s.default_is_nothing)
        })
    }

    pub fn default_is_attrs_decorator(&self, name: &Name) -> bool {
        self.attrs_default_decorator_method_range(name).is_some()
    }

    /// The name range of this field's `@<field>.default` method, if any.
    pub fn attrs_default_decorator_method_range(&self, name: &Name) -> Option<TextRange> {
        self.0
            .get(name)
            .and_then(|prop| prop.attrs_field_specifier)
            .and_then(|s| s.default_decorator_method_range)
    }

    /// The name range of this field's first `@<field>.converter` method, if any.
    pub fn attrs_converter_decorator_method_range(&self, name: &Name) -> Option<TextRange> {
        self.0
            .get(name)
            .and_then(|prop| prop.attrs_field_specifier)
            .and_then(|s| s.converter_decorator_method_range)
    }

    /// Whether the field's attrs specifier honors a `type=` argument (`attr.ib`, not `field`).
    pub fn attrs_specifier_honors_type(&self, name: &Name) -> bool {
        self.0.get(name).is_some_and(|prop| {
            prop.attrs_field_specifier
                .is_some_and(|s| s.kind == AttrsFieldSpecifierKind::Attrib)
        })
    }

    pub fn is_field_initialized_on_class(&self, name: &Name) -> bool {
        self.0
            .get(name)
            .is_some_and(|prop| prop.is_initialized_on_class)
    }

    pub fn field_decl_range(&self, name: &Name) -> Option<TextRange> {
        Some(self.0.get(name)?.range)
    }

    /// The declaration range and documentation of a field.
    pub fn field_decl(&self, name: &Name) -> Option<(TextRange, &DocRanges)> {
        let field = self.0.get(name)?;
        Some((field.range, &field.doc_ranges))
    }

    pub fn field_doc_ranges(&self, name: &Name) -> Option<&DocRanges> {
        Some(&self.0.get(name)?.doc_ranges)
    }
}

/// The data of a class definition.
///
/// Align to a cache line to prevent false sharing. `ClassDef` is stored
/// behind `Arc` and accessed concurrently from multiple threads during
/// type checking. Without alignment, multiple `Arc<ClassDef>` allocations
/// can land on the same cache line, causing cross-thread invalidation
/// traffic when any of them are accessed (even read-only, due to the Arc
/// refcount on adjacent allocations).
#[repr(align(64))]
pub struct ClassDef {
    def_index: ClassDefIndex,
    /// Only set for top-level classes.
    known: Option<KnownClass>,
    qname: QName,
    is_protocol: bool,
    deprecation: Option<Deprecation>,
    /// Whether the class declares any type parameters. This is decided from syntax alone; the
    /// type parameters themselves come from the class's `KeyTParams`.
    has_tparams: bool,
}

impl Debug for ClassDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassDef")
            .field("index", &self.def_index)
            .field("qname", &self.qname)
            .field("is_protocol", &self.is_protocol)
            .field("deprecation", &self.deprecation)
            .field("has_tparams", &self.has_tparams)
            // We don't print `fields` because it's way too long.
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq, TypeOrd)]
pub enum ClassKind {
    StaticMethod(Name),
    ClassMethod(Name),
    Property(Name),
    CachedProperty(Name),
    Class,
    EnumMember,
    EnumNonmember,
    DataclassField,
}

impl ClassKind {
    /// `name` is the class's own name, which the descriptor kinds carry.
    fn new(known: Option<KnownClass>, name: &Name) -> Self {
        match known {
            Some(KnownClass::StaticMethod | KnownClass::AbcAbstractStaticMethod) => {
                Self::StaticMethod(name.clone())
            }
            Some(KnownClass::ClassMethod | KnownClass::AbcAbstractClassMethod) => {
                Self::ClassMethod(name.clone())
            }
            Some(
                KnownClass::Property | KnownClass::AbcAbstractProperty | KnownClass::EnumProperty,
            ) => Self::Property(name.clone()),
            Some(
                KnownClass::FunctoolsCachedProperty
                | KnownClass::CachedPropertyThreadedCachedProperty
                | KnownClass::CachedProperty
                | KnownClass::CinderCachedProperty
                | KnownClass::CinderAsyncCachedProperty,
            ) => Self::CachedProperty(name.clone()),
            Some(KnownClass::EnumMember) => Self::EnumMember,
            Some(KnownClass::EnumNonmember) => Self::EnumNonmember,
            Some(KnownClass::DataclassesField) => Self::DataclassField,
            _ => Self::Class,
        }
    }
}

impl Display for ClassDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "class {}", self.qname.id())?;
        writeln!(f, ": ...")
    }
}

// A note on terminology regarding attribute-related concepts:
// - "field" refers to something defined in a class body, with a raw type as written.
// - "member" refers to a name defined on a class, including inherited members whose
//   types should be expressed in terms of the type parameters of the current class
// - "attribute" refers to a value actually accessed from an instance or class object,
//   which involves substituting type arguments for the class type parameters as
//   well as descriptor handling (including method binding).
impl Class {
    pub fn new(
        def_index: ClassDefIndex,
        name: Identifier,
        parent: NestingContext,
        module: Module,
        has_tparams: bool,
        is_protocol: bool,
        deprecation: Option<Deprecation>,
    ) -> Self {
        let known = if parent.is_toplevel() {
            KnownClass::lookup(module.name(), name.id.as_str())
        } else {
            None
        };
        Self(Arc::new(ClassDef {
            def_index,
            known,
            qname: QName::new(name, parent, module),
            is_protocol,
            deprecation,
            has_tparams,
        }))
    }

    pub fn has_tparams(&self) -> bool {
        self.0.has_tparams
    }

    pub fn index(&self) -> ClassDefIndex {
        self.0.def_index
    }

    pub fn module_name(&self) -> ModuleName {
        self.0.qname.module_name()
    }

    pub fn module_path(&self) -> &ModulePath {
        self.0.qname.module_path()
    }

    pub fn module_key(&self) -> ModuleKey {
        self.0.qname.module().key()
    }

    /// The known class this is, if it is one. Only top-level classes can be known classes.
    pub fn known(&self) -> Option<KnownClass> {
        self.0.known
    }

    pub fn is(&self, known: KnownClass) -> bool {
        self.0.known == Some(known)
    }

    /// The definition data stored in this `Class`, which `TypeEnv::class_def` returns by default.
    pub(crate) fn inline_def(&self) -> &ClassDef {
        &self.0
    }

    pub fn def<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a ClassDef {
        env.class_def(self)
    }

    pub fn name<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a Name {
        self.def(env).qname.id()
    }

    pub fn qname<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a QName {
        &self.def(env).qname
    }

    pub fn range<E: TypeEnv + ?Sized>(&self, env: &E) -> TextRange {
        self.def(env).qname.range()
    }

    /// The module that defines this class.
    pub fn module<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a Module {
        self.def(env).qname.module()
    }

    pub fn is_protocol<E: TypeEnv + ?Sized>(&self, env: &E) -> bool {
        self.def(env).is_protocol
    }

    pub fn deprecation<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> Option<&'a Deprecation> {
        self.def(env).deprecation.as_ref()
    }

    pub fn kind<E: TypeEnv + ?Sized>(&self, env: &E) -> ClassKind {
        ClassKind::new(self.known(), self.name(env))
    }

    /// The type parameters of this class, which are empty if it is not generic.
    pub fn tparams<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a TParams {
        env.class_tparams(self)
    }

    /// Key to use for equality purposes. If we have the same module and index,
    /// we must point at the same class underneath. `ModuleKey` equality treats a
    /// file open in memory and the same file on disk as one module.
    fn key_eq(&self) -> (ClassDefIndex, ModuleKey) {
        (self.0.def_index, self.module_key())
    }

    /// Key to use for comparison purposes. Main used to sort identifiers in union,
    /// and then alphabetically sorting by the name gives a predictable answer.
    fn key_ord(&self) -> (&QName, ClassDefIndex) {
        (&self.0.qname, self.0.def_index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq, TypeOrd)]
pub struct ClassType(Class, TArgs, Option<Arc<DjangoClassMetadata>>);

/// Construction-time Django specializations on nominal class instances.
///
/// The manager resolves generated members against its queryset. A specialized class uses
/// its effective base for inherited members and nominal upcasts. Type traversal and substitution
/// include this data, and instance binding preserves it when replacing `Self`. Explicit method
/// return types stand alone: neither their model nor their row is rebound to the manager's model.
/// Fields retain construction-specific accessor and override contracts.
/// A many-to-many manager records the target model's primary key type, which replaces the
/// `int` that the stubs accept in place of model instances.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(TypeEq, TypeOrd)]
pub enum DjangoClassMetadata {
    Manager {
        queryset: ClassType,
    },
    ManyRelatedManager {
        pk_type: Type,
    },
    SpecializedBase {
        base: ClassType,
    },
    Field {
        get_type: Type,
        set_type: Type,
        override_type: Type,
    },
}

impl Visit<Type> for Arc<DjangoClassMetadata> {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        match self.as_ref() {
            DjangoClassMetadata::Manager { queryset } => queryset.visit(f),
            DjangoClassMetadata::ManyRelatedManager { pk_type } => f(pk_type),
            DjangoClassMetadata::SpecializedBase { base } => base.visit(f),
            DjangoClassMetadata::Field {
                get_type,
                set_type,
                override_type,
            } => {
                f(get_type);
                f(set_type);
                f(override_type);
            }
        }
    }
}

impl VisitMut<Type> for Arc<DjangoClassMetadata> {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        match Arc::make_mut(self) {
            DjangoClassMetadata::Manager { queryset } => queryset.visit_mut(f),
            DjangoClassMetadata::ManyRelatedManager { pk_type } => f(pk_type),
            DjangoClassMetadata::SpecializedBase { base } => base.visit_mut(f),
            DjangoClassMetadata::Field {
                get_type,
                set_type,
                override_type,
            } => {
                f(get_type);
                f(set_type);
                f(override_type);
            }
        }
    }
}

impl ClassType {
    /// Create a class type.
    /// The `targs` must match the `tparams`, if this fails we will panic.
    pub fn new(class: Class, targs: TArgs) -> Self {
        Self(class, targs, None)
    }

    pub fn class_object(&self) -> &Class {
        &self.0
    }

    pub fn into_class_object(self) -> Class {
        self.0
    }

    /// The type parameters that the type arguments were created for. Code that pairs the type
    /// arguments with parameters uses these, so that the two always match. The class's current
    /// type parameters are `class_object().tparams(env)`.
    pub fn tparams(&self) -> &TParams {
        self.1.tparams()
    }

    pub fn targs(&self) -> &TArgs {
        &self.1
    }

    pub fn targs_mut(&mut self) -> &mut TArgs {
        &mut self.1
    }

    /// Rewrite type arguments of some class relative to another.
    ///
    /// This is used to propagate instantiation of base class type parameters when computing
    /// the MRO.
    pub fn substitute_with(&self, substitution: &Substitution, env: &dyn TypeEnv) -> Self {
        let mut result = self.clone();
        result.1 = self.1.substitute_with(substitution, env);
        if let Some(metadata) = &mut result.2 {
            metadata.recurse_mut(&mut |ty| substitution.substitute_into_mut(ty));
        }
        result
    }

    /// Attach a Django specialization to this nominal class instance.
    pub fn with_django_metadata(mut self, metadata: DjangoClassMetadata) -> Self {
        self.2 = Some(Arc::new(metadata));
        self
    }

    /// Return the construction-time Django specialization, if present.
    pub fn django_metadata(&self) -> Option<&DjangoClassMetadata> {
        self.2.as_deref()
    }

    pub fn substitution(&self) -> Substitution<'_> {
        self.targs().substitution()
    }

    pub fn to_type(self) -> Type {
        Type::ClassType(self)
    }

    pub fn known(&self) -> Option<KnownClass> {
        self.0.known()
    }

    pub fn is(&self, known: KnownClass) -> bool {
        self.0.is(known)
    }

    pub fn def<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a ClassDef {
        self.0.def(env)
    }

    pub fn name<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a Name {
        self.0.name(env)
    }

    pub fn qname<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a QName {
        self.0.qname(env)
    }

    pub fn range<E: TypeEnv + ?Sized>(&self, env: &E) -> TextRange {
        self.0.range(env)
    }

    pub fn module<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> &'a Module {
        self.0.module(env)
    }

    pub fn is_protocol<E: TypeEnv + ?Sized>(&self, env: &E) -> bool {
        self.0.is_protocol(env)
    }

    pub fn deprecation<'a, E: TypeEnv + ?Sized>(&'a self, env: &'a E) -> Option<&'a Deprecation> {
        self.0.deprecation(env)
    }

    pub fn kind<E: TypeEnv + ?Sized>(&self, env: &E) -> ClassKind {
        self.0.kind(env)
    }
}
