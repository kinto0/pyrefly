/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Metadata collected during the binding phase that can be queried
//! without going through the solve/calculation code paths.
//!
//! This module stores per-class metadata (the class definition and its fields)
//! in a `Vec` indexed by `ClassDefIndex`, enabling efficient lookups.

use dupe::Dupe;
use pyrefly_python::module::Module;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_types::class::ClassDef;
use pyrefly_types::class::ClassDefIndex;
use pyrefly_types::class::ClassFields;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

/// Metadata for a single class definition, populated during binding.
#[derive(Debug, Clone)]
pub struct ClassMetadata {
    /// The definition data of the class. Each `Class` of this module carries a copy of it.
    pub def: ClassDef,
    /// The fields are all the names declared on the class that we were able to detect
    /// from an AST traversal, which includes:
    /// - any name defined in the class body (e.g. by assignment or a def statement)
    /// - attributes annotated in the class body (but not necessarily defined)
    /// - anything assigned to something we think is a `self` or `cls` argument
    ///
    /// The last case may include names that are actually declared in a parent class,
    /// because at binding time we cannot know that so we have to treat assignment
    /// as potentially defining a field that would not otherwise exist.
    pub fields: ClassFields,
}

impl ClassMetadata {
    fn placeholder(module: &Module) -> Self {
        Self {
            def: ClassDef::new(
                Identifier::new(Name::new_static("<placeholder>"), TextRange::default()),
                NestingContext::toplevel(),
                module.dupe(),
                false,
                false,
                None,
            ),
            fields: ClassFields::empty(),
        }
    }
}

/// Metadata collected during the binding phase for all classes in a module.
///
/// Stored in an `Arc` so it can be shared between `Answers` and `Solutions`
/// without copying. During access, callers hold a `Guard` or
/// borrow through the `Arc` rather than cloning it, avoiding contended
/// atomic reference count operations.
#[derive(Debug, Clone)]
pub struct BindingsMetadata {
    classes: Vec<ClassMetadata>,
    /// The row of a class that has an index but whose definition is not bound yet. The binding
    /// of every class definition replaces its row, so no row is a placeholder once the module is
    /// bound.
    placeholder: ClassMetadata,
}

impl BindingsMetadata {
    pub fn new(module: &Module) -> Self {
        Self {
            classes: Vec::new(),
            placeholder: ClassMetadata::placeholder(module),
        }
    }

    /// Allocate a new `ClassDefIndex`, with a placeholder row that the binding of the class
    /// definition replaces.
    pub fn push_class(&mut self) -> ClassDefIndex {
        let idx = ClassDefIndex(self.classes.len() as u32);
        self.classes.push(self.placeholder.clone());
        idx
    }

    pub fn get_class(&self, idx: ClassDefIndex) -> &ClassMetadata {
        &self.classes[idx.0 as usize]
    }

    /// Bounds-checked version for lookups where the `ClassDefIndex` may be
    /// stale after an incremental rebuild.
    pub fn get_class_checked(&self, idx: ClassDefIndex) -> Option<&ClassMetadata> {
        self.classes.get(idx.0 as usize)
    }

    pub fn get_class_mut(&mut self, idx: ClassDefIndex) -> &mut ClassMetadata {
        &mut self.classes[idx.0 as usize]
    }

    /// The classes whose declared fields differ between `self` and `other`, wherever in the file
    /// they are declared, including the classes that only one of them has.
    pub fn changed_class_rows<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = ClassDefIndex> + 'a {
        (0..self.classes.len().max(other.classes.len()))
            .filter(move |&i| {
                !matches!(
                    (self.classes.get(i), other.classes.get(i)),
                    (Some(x), Some(y)) if x.fields.same_declarations(&y.fields)
                )
            })
            .map(|i| ClassDefIndex(i as u32))
    }
}
