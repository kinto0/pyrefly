/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;

use dupe::Dupe;
use parse_display::Display;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::binding::binding::ClassFieldInitialValue;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyFunction;
use crate::binding::bindings::BindingTable;
use crate::dunder;
use crate::export::definitions::DefinitionStyle;
use crate::export::definitions::Definitions;
use crate::export::exports::LookupExport;
use crate::export::special::SpecialEntry;
use crate::graph::index::Idx;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::short_identifier::ShortIdentifier;
use crate::ruff::ast::Ast;
use crate::sys_info::SysInfo;
use crate::types::class::ClassIndex;

/// Many names may map to the same TextRange (e.g. from foo import *).
/// But no other static will point at the same TextRange.
#[derive(Default, Clone, Debug)]
pub struct Static(pub SmallMap<Name, StaticInfo>);

#[derive(Clone, Debug)]
pub struct StaticInfo {
    pub loc: TextRange,
    /// The location of the first annotated name for this binding, if any.
    pub annot: Option<Idx<KeyAnnotation>>,
    /// How many times this will be redefined
    pub count: usize,
    pub style: DefinitionStyle,
}

impl StaticInfo {
    pub fn as_key(&self, name: &Name) -> Key {
        if self.count == 1 {
            if matches!(self.style, DefinitionStyle::ImportModule(_)) {
                Key::Import(name.clone(), self.loc)
            } else {
                // We are constructing an identifier, but it must have been one that we saw earlier
                assert_ne!(self.loc, TextRange::default());
                Key::Definition(ShortIdentifier::new(&Identifier {
                    id: name.clone(),
                    range: self.loc,
                }))
            }
        } else {
            Key::Anywhere(name.clone(), self.loc)
        }
    }

    pub fn is_global(&self) -> bool {
        self.style == DefinitionStyle::Global
    }

    pub fn is_nonlocal(&self) -> bool {
        self.style == DefinitionStyle::Nonlocal
    }
}

impl Static {
    fn add_with_count(
        &mut self,
        name: Hashed<Name>,
        loc: TextRange,
        annot: Option<Idx<KeyAnnotation>>,
        count: usize,
    ) -> &mut StaticInfo {
        // Use whichever one we see first
        let res = self.0.entry_hashed(name).or_insert(StaticInfo {
            loc,
            annot,
            count: 0,
            style: DefinitionStyle::Local,
        });
        res.count += count;
        res
    }

    pub fn add(&mut self, name: Name, range: TextRange, annot: Option<Idx<KeyAnnotation>>) {
        self.add_with_count(Hashed::new(name), range, annot, 1);
    }

    pub fn stmts(
        &mut self,
        x: &[Stmt],
        module_info: &ModuleInfo,
        top_level: bool,
        lookup: &dyn LookupExport,
        sys_info: &SysInfo,
        mut get_annotation_idx: impl FnMut(ShortIdentifier) -> Idx<KeyAnnotation>,
    ) {
        let mut d = Definitions::new(
            x,
            module_info.name(),
            module_info.path().is_init(),
            sys_info,
        );
        if top_level && module_info.name() != ModuleName::builtins() {
            d.inject_builtins();
        }

        let mut wildcards = Vec::with_capacity(d.import_all.len());
        for (m, range) in d.import_all {
            if let Ok(exports) = lookup.get(m) {
                wildcards.push((range, exports.wildcard(lookup)));
            }
        }

        // Try and avoid rehashing while we insert, with a little bit of spare space
        let capacity_guess =
            d.definitions.len() + wildcards.iter().map(|x| x.1.len()).sum::<usize>();
        self.0.reserve(((capacity_guess * 5) / 4) + 25);

        for (name, def) in d.definitions.into_iter_hashed() {
            let annot = def.annot.map(&mut get_annotation_idx);
            let info = self.add_with_count(name, def.range, annot, def.count);
            info.style = def.style;
        }
        for (range, wildcard) in wildcards {
            for name in wildcard.iter_hashed() {
                // TODO: semantics of import * and global var with same name
                self.add_with_count(name.cloned(), range, None, 1).style =
                    DefinitionStyle::ImportModule(module_info.name());
            }
        }
    }

    pub fn expr_lvalue(&mut self, x: &Expr) {
        let mut add = |name: &ExprName| self.add(name.id.clone(), name.range, None);
        Ast::expr_lvalue(x, &mut add);
    }
}

/// Flow-sensitive information about a name.
#[derive(Default, Clone, Debug)]
pub struct Flow {
    pub info: SmallMap<Name, FlowInfo>,
    // Should this flow be merged into the next? Flow merging occurs after constructs like branches and loops.
    pub no_next: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowStyle {
    /// Not one of the styles below.
    None,
    /// Am I an assignment in a class body?
    ClassField { initial_value: Option<Expr> },
    /// Am I the result of an import (which needs merging).
    /// E.g. `import foo.bar` and `import foo.baz` need merging.
    /// The `ModuleName` will be the most recent entry.
    MergeableImport(ModuleName),
    /// Was I imported from somewhere (and if so, where)
    /// E.g. Both `from foo import bar` and
    /// `from foo import bar as baz` would get `(foo, bar)`.
    Import(ModuleName, Name),
    /// Am I an alias for a module import, `import foo.bar as baz`
    /// would get `foo.bar` here.
    ImportAs(ModuleName),
    /// Am I a function definition? Used to chain overload definitions.
    /// If so, does my return type have an explicit annotation?
    FunctionDef(Idx<KeyFunction>, bool),
    /// The name is possibly uninitialized (perhaps due to merging branches)
    PossiblyUninitialized,
    /// The name was in an annotated declaration like `x: int` but not initialized
    Uninitialized,
}

impl FlowStyle {
    /// Produce an error message for an uninitialized or unbound variable.
    pub fn uninitialized_error_message(&self, name: &Identifier) -> Option<String> {
        match self {
            Self::Uninitialized => Some(format!("`{name}` is uninitialized")),
            Self::PossiblyUninitialized => Some(format!("`{name}` may be uninitialized")),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowInfo {
    /// The key to use if you need the value of this name.
    pub key: Idx<Key>,
    /// The default value to use if you are inside a loop and need to default a Var.
    /// Only set if you are outside a loop, OR it has never been set before.
    /// Only used if you are inside a loop.
    pub default: Idx<Key>,
    /// The style of this binding.
    pub style: FlowStyle,
}

impl FlowInfo {
    pub fn as_initial_value(&self) -> ClassFieldInitialValue {
        match &self.style {
            FlowStyle::ClassField {
                initial_value: Some(e),
            } => ClassFieldInitialValue::Class(Some(e.clone())),
            FlowStyle::ClassField {
                initial_value: None,
            } => ClassFieldInitialValue::Instance(None),
            // All other styles (e.g. function def, import) indicate we do have
            // a value, but it is not coming from a simple style.
            _ => ClassFieldInitialValue::Class(None),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClassBodyInner {
    pub name: Identifier,
    index: ClassIndex,
    attributes_from_recognized_methods: SmallMap<Name, SmallMap<Name, InstanceAttribute>>,
    attributes_from_other_methods: SmallMap<Name, SmallMap<Name, InstanceAttribute>>,
}

/// The method where an attribute was defined implicitly by assignment to `self.<attr_name>`
///
/// We track whether this method is recognized as a valid attribute-defining
/// method (e.g. a constructor); if an attribute is inferred only from assignments
/// in non-recognized methods, we will infer its type but also produce a type error.
pub struct MethodThatSetsAttr {
    pub method_name: Name,
    pub recognized_attribute_defining_method: bool,
}

impl ClassBodyInner {
    pub fn new(name: Identifier, index: ClassIndex) -> Self {
        Self {
            name,
            index,
            attributes_from_recognized_methods: SmallMap::new(),
            attributes_from_other_methods: SmallMap::new(),
        }
    }
    pub fn as_class_key(&self) -> KeyClass {
        KeyClass(ShortIdentifier::new(&self.name))
    }

    pub fn as_class_metadata_key(&self) -> KeyClassMetadata {
        KeyClassMetadata(self.index)
    }

    pub fn add_attributes_defined_by_method(
        &mut self,
        method_name: Name,
        attributes: SmallMap<Name, InstanceAttribute>,
    ) {
        if is_attribute_defining_method(&method_name, &self.name.id) {
            self.attributes_from_recognized_methods
                .insert(method_name, attributes);
        } else {
            self.attributes_from_other_methods
                .insert(method_name, attributes);
        }
    }

    /// Produces triples (hashed_attr_name, MethodThatSetsAttr, attribute) for all assignments
    /// to `self.<attr_name>` in methods.
    ///
    /// We iterate recognized methods first, which - assuming that the first result is the one
    /// used in our class logic, which is the case - ensures both that we don't produce
    /// unnecessary errors about attributes implicitly defined in unrecognized methods
    /// and that the types inferred from recognized methods take precedence.
    pub fn method_defined_attributes(
        self,
    ) -> impl Iterator<Item = (Hashed<Name>, MethodThatSetsAttr, InstanceAttribute)> {
        Self::iter_attributes(self.attributes_from_recognized_methods, true).chain(
            Self::iter_attributes(self.attributes_from_other_methods, false),
        )
    }

    fn iter_attributes(
        attrs: SmallMap<Name, SmallMap<Name, InstanceAttribute>>,
        recognized_attribute_defining_method: bool,
    ) -> impl Iterator<Item = (Hashed<Name>, MethodThatSetsAttr, InstanceAttribute)> {
        {
            attrs.into_iter().flat_map(move |(method_name, attrs)| {
                attrs.into_iter_hashed().map(move |(name, attr)| {
                    (
                        name,
                        MethodThatSetsAttr {
                            method_name: method_name.clone(),
                            recognized_attribute_defining_method,
                        },
                        attr,
                    )
                })
            })
        }
    }
}

fn is_attribute_defining_method(method_name: &Name, class_name: &Name) -> bool {
    if method_name == &dunder::INIT {
        true
    } else {
        (class_name.contains("Test") || class_name.contains("test"))
            && is_test_setup_method(method_name)
    }
}

fn is_test_setup_method(method_name: &Name) -> bool {
    match method_name.as_str() {
        "asyncSetUp" | "async_setUp" | "setUp" | "_setup" | "_async_setup"
        | "async_with_context" | "with_context" | "setUpClass" => true,
        _ => false,
    }
}

#[derive(Clone, Debug)]
pub struct InstanceAttribute(
    pub ExprOrBinding,
    pub Option<Idx<KeyAnnotation>>,
    pub TextRange,
);

#[derive(Clone, Debug)]
pub struct MethodInner {
    pub name: Identifier,
    pub self_name: Option<Identifier>,
    pub instance_attributes: SmallMap<Name, InstanceAttribute>,
}

#[derive(Clone, Debug)]
pub enum ScopeKind {
    Annotation,
    ClassBody(ClassBodyInner),
    Comprehension,
    Function,
    Method(MethodInner),
    Module,
}

#[derive(Clone, Debug, Display)]
pub enum LoopExit {
    NeverRan,
    #[display("break")]
    Break,
    #[display("continue")]
    Continue,
}

/// Flow snapshots for all possible exitpoints from a loop.
#[derive(Clone, Debug)]
pub struct Loop(pub Vec<(LoopExit, Flow)>);

#[derive(Clone, Debug)]
pub struct Scope {
    pub range: TextRange,
    /// Things that are defined in this scope, statically, e.g. `x = 1` or `def f():`.
    /// Populated at the beginning before entering the scope.
    pub stat: Static,
    /// Things that are defined in this scope as they are reached.
    /// Initially starts out empty, but is populated as statements are encountered.
    /// Updated if there are multiple assignments. E.g. `x = 1; x = 2` would update the `x` binding twice.
    /// All flow bindings will have a static binding, _usually_ in this scope, but occasionally
    /// in a parent scope (e.g. for narrowing operations).
    pub flow: Flow,
    /// Are Flow types above this unreachable.
    /// Set when we enter something like a function, and can't guarantee what flow values are in scope.
    pub barrier: bool,
    pub kind: ScopeKind,
    /// Stack of for/while loops we're in. Does not include comprehensions.
    pub loops: Vec<Loop>,
}

impl Scope {
    fn new(range: TextRange, barrier: bool, kind: ScopeKind) -> Self {
        Self {
            range,
            stat: Default::default(),
            flow: Default::default(),
            barrier,
            kind,
            loops: Default::default(),
        }
    }

    pub fn annotation(range: TextRange) -> Self {
        Self::new(range, false, ScopeKind::Annotation)
    }

    pub fn class_body(range: TextRange, index: ClassIndex, name: Identifier) -> Self {
        Self::new(
            range,
            false,
            ScopeKind::ClassBody(ClassBodyInner::new(name, index)),
        )
    }

    pub fn comprehension(range: TextRange) -> Self {
        Self::new(range, false, ScopeKind::Comprehension)
    }

    pub fn function(range: TextRange) -> Self {
        Self::new(range, true, ScopeKind::Function)
    }

    pub fn method(range: TextRange, name: Identifier) -> Self {
        Self::new(
            range,
            true,
            ScopeKind::Method(MethodInner {
                name,
                self_name: None,
                instance_attributes: SmallMap::new(),
            }),
        )
    }

    fn module(range: TextRange) -> Self {
        Self::new(range, false, ScopeKind::Module)
    }
}

#[derive(Clone, Debug)]
struct ScopeTreeNode {
    scope: Scope,
    children: Vec<ScopeTreeNode>,
}

/// Determines if a range contains a position, inclusive on both ends.
fn contains_inclusive(range: TextRange, position: TextSize) -> bool {
    range.start() <= position && position <= range.end()
}

impl ScopeTreeNode {
    /// Return whether we hit a child scope with a barrier
    fn visit_available_definitions(
        &self,
        table: &BindingTable,
        position: TextSize,
        visitor: &mut impl FnMut(Idx<Key>),
    ) -> bool {
        if !contains_inclusive(self.scope.range, position) {
            return false;
        }
        let mut barrier = false;
        for node in &self.children {
            let hit_barrier = node.visit_available_definitions(table, position, visitor);
            barrier = barrier || hit_barrier
        }
        if !barrier {
            for info in self.scope.flow.info.values() {
                visitor(info.key);
            }
        }
        for (name, info) in &self.scope.stat.0 {
            if let Some(key) = table.types.0.key_to_idx(&info.as_key(name)) {
                visitor(key);
            }
        }
        barrier || self.scope.barrier
    }

    fn collect_available_definitions(
        &self,
        table: &BindingTable,
        position: TextSize,
        collector: &mut SmallSet<Idx<Key>>,
    ) {
        self.visit_available_definitions(table, position, &mut |key| {
            collector.insert(key);
        });
    }
}

/// Scopes keep track of the current stack of the scopes we are in.
#[derive(Clone, Debug)]
pub struct Scopes {
    scopes: Vec1<ScopeTreeNode>,
    /// When `keep_scope_tree` flag is on, the stack will maintain a tree of all the scopes
    /// throughout the program, even if the scope has already been popped. This is useful
    /// for autocomplete purposes.
    keep_scope_tree: bool,
}

impl Scopes {
    pub fn module(range: TextRange, keep_scope_tree: bool) -> Self {
        let module_scope = Scope::module(range);
        Self {
            scopes: Vec1::new(ScopeTreeNode {
                scope: module_scope,
                children: Vec::new(),
            }),
            keep_scope_tree,
        }
    }

    pub fn current(&self) -> &Scope {
        &self.scopes.last().scope
    }

    pub fn current_mut(&mut self) -> &mut Scope {
        &mut self.current_mut_node().scope
    }

    fn current_mut_node(&mut self) -> &mut ScopeTreeNode {
        self.scopes.last_mut()
    }

    /// There is only one scope remaining, return it.
    pub fn finish(self) -> ScopeTrace {
        let (a, b) = self.scopes.split_off_last();
        assert_eq!(a.len(), 0);
        ScopeTrace(b)
    }

    pub fn push(&mut self, scope: Scope) {
        self.scopes.push(ScopeTreeNode {
            scope,
            children: Vec::new(),
        });
    }

    pub fn pop(&mut self) -> Scope {
        let ScopeTreeNode { scope, children } = self.scopes.pop().unwrap();
        if self.keep_scope_tree {
            self.current_mut_node().children.push(ScopeTreeNode {
                scope: scope.clone(),
                children,
            });
        }
        scope
    }

    pub fn iter_rev(&self) -> impl ExactSizeIterator<Item = &Scope> {
        self.scopes.iter().map(|node| &node.scope).rev()
    }

    pub fn iter_rev_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Scope> {
        self.scopes.iter_mut().map(|node| &mut node.scope).rev()
    }

    /// Return the default to use, if inside a loop.
    pub fn update_flow_info(
        &mut self,
        loop_depth: u32,
        name: &Name,
        key: Idx<Key>,
        style: FlowStyle,
    ) -> Option<Idx<Key>> {
        self.update_flow_info_hashed(loop_depth, Hashed::new(name), key, style)
    }

    /// Return the default to use, if inside a loop.
    pub fn update_flow_info_hashed(
        &mut self,
        loop_depth: u32,
        name: Hashed<&Name>,
        key: Idx<Key>,
        style: FlowStyle,
    ) -> Option<Idx<Key>> {
        match self.current_mut().flow.info.entry_hashed(name.cloned()) {
            Entry::Vacant(e) => {
                e.insert(FlowInfo {
                    key,
                    default: key,
                    style,
                });
                None
            }
            Entry::Occupied(mut e) => {
                let default = if loop_depth != 0 {
                    Some(e.get().default)
                } else {
                    None
                };
                *e.get_mut() = FlowInfo {
                    key,
                    default: default.unwrap_or(key),
                    style,
                };
                default
            }
        }
    }

    fn get_flow_info(&self, name: &Name) -> Option<&FlowInfo> {
        let name = Hashed::new(name);
        for scope in self.iter_rev() {
            if let Some(flow) = scope.flow.info.get_hashed(name) {
                return Some(flow);
            }
        }
        None
    }

    pub fn get_flow_style(&self, name: &Name) -> &FlowStyle {
        match self.get_flow_info(name) {
            Some(flow) => &flow.style,
            None => &FlowStyle::None,
        }
    }

    pub fn get_special_entry<'a>(&'a self, name: &Name) -> Option<SpecialEntry<'a>> {
        let flow = self.get_flow_info(name)?;
        let entry = match &flow.style {
            FlowStyle::Import(m, name) => SpecialEntry::ImportName(m.dupe(), name),
            FlowStyle::MergeableImport(m) | FlowStyle::ImportAs(m) => {
                SpecialEntry::ImportModule(m.dupe())
            }
            _ => SpecialEntry::Local,
        };
        Some(entry)
    }
}

#[derive(Clone, Debug)]
pub struct ScopeTrace(ScopeTreeNode);

impl ScopeTrace {
    pub fn toplevel_scope(&self) -> &Scope {
        &self.0.scope
    }

    pub fn available_definitions(
        &self,
        table: &BindingTable,
        position: TextSize,
    ) -> SmallSet<Idx<Key>> {
        let mut collector = SmallSet::new();
        self.0
            .collect_available_definitions(table, position, &mut collector);
        collector
    }

    pub fn definition_at_position<'a>(
        &self,
        table: &'a BindingTable,
        position: TextSize,
    ) -> Option<&'a Key> {
        let mut definition = None;
        self.0
            .visit_available_definitions(table, position, &mut |idx| {
                let key = table.types.0.idx_to_key(idx);
                match key {
                    Key::Definition(short_identifier)
                        if short_identifier.range().contains_inclusive(position) =>
                    {
                        definition = Some(key);
                    }
                    _ => {}
                }
            });
        definition
    }
}
