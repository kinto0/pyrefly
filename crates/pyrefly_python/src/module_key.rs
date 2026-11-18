/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use dupe::Dupe;
use static_interner::Intern;
use static_interner::Interner;

use crate::module_name::ModuleName;
use crate::module_path::ModulePath;

/// Holds one entry for each distinct module name and path that a key was created for. Entries are
/// never freed, so it grows with the number of distinct modules seen, not with edits.
static MODULE_KEY_INTERNER: Interner<ModuleIdentity> = Interner::new();

/// The interner deduplicates by exact name and path, so a file open in memory and the same file on
/// disk are separate entries.
#[derive(PartialEq, Eq)]
struct ModuleIdentity {
    name: ModuleName,
    path: ModulePath,
}

/// Hashes only what `ModuleKey` compares, so that keys that compare equal have the same hash.
impl Hash for ModuleIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.path.to_key_eq().hash(state);
    }
}

/// Identifies a module by its name and the path it was loaded from, so that it tells apart two
/// files with the same module name, such as a stub on the search path and the typeshed module it
/// shadows, and the two module names of one file. It is interned, so it is one word, `Copy`, and
/// usually compared by pointer.
///
/// Two keys are equal when their names are equal and their paths are equal after
/// `ModulePath::to_key_eq`, as for `Class`: a file open in memory and the same file on disk are
/// the same module. Keys are ordered by name and then by path, which is the same in every run.
///
/// Code that looks a module up, or caches data by module, must use `exact` instead: the in-memory
/// and on-disk versions of a file are loaded and solved separately, and can hold different
/// contents. `path` returns the path that the key was created with, which is the one to look the
/// module up by.
#[derive(Clone, Copy, Dupe)]
pub struct ModuleKey(Intern<ModuleIdentity>);

/// Identifies a module by the exact name and path it was loaded from, for maps that look modules
/// up or cache their data. Unlike `ModuleKey`, the in-memory and on-disk versions of a file are
/// different keys. Equality compares interned pointers.
#[derive(Clone, Copy, Dupe, PartialEq, Eq, Hash)]
pub struct ExactModuleKey(Intern<ModuleIdentity>);

impl ModuleKey {
    pub fn new(name: ModuleName, path: &ModulePath) -> Self {
        Self(MODULE_KEY_INTERNER.intern(ModuleIdentity {
            name,
            path: path.dupe(),
        }))
    }

    pub fn name(self) -> ModuleName {
        self.0.deref_static().name
    }

    pub fn path(self) -> &'static ModulePath {
        &self.0.deref_static().path
    }

    pub fn exact(self) -> ExactModuleKey {
        ExactModuleKey(self.0)
    }
}

impl ExactModuleKey {
    /// The `ModuleKey` of this module, whose `name` and `path` are the ones this key was created
    /// with.
    pub fn key(self) -> ModuleKey {
        ModuleKey(self.0)
    }
}

/// The interned hash, which covers only what `eq` compares (see `ModuleIdentity`).
impl Hash for ModuleKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for ModuleKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
            || (self.name() == other.name() && self.path().to_key_eq() == other.path().to_key_eq())
    }
}

impl Eq for ModuleKey {}

impl PartialOrd for ModuleKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModuleKey {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 == other.0 {
            return Ordering::Equal;
        }
        self.name()
            .cmp(&other.name())
            .then_with(|| self.path().to_key_eq().cmp(&other.path().to_key_eq()))
    }
}

impl Debug for ModuleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ModuleKey")
            .field(&self.name().as_str())
            .field(self.path())
            .finish()
    }
}

impl Debug for ExactModuleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExactModuleKey")
            .field(&ModuleKey(self.0))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_module_key_identity() {
        let name = ModuleName::from_str("foo");
        let disk = ModuleKey::new(name, &ModulePath::filesystem(PathBuf::from("a/foo.py")));
        let memory = ModuleKey::new(name, &ModulePath::memory(PathBuf::from("a/foo.py")));
        let other = ModuleKey::new(name, &ModulePath::filesystem(PathBuf::from("b/foo.py")));
        assert_eq!(
            disk,
            ModuleKey::new(name, &ModulePath::filesystem(PathBuf::from("a/foo.py")))
        );
        assert_eq!(disk, memory);
        assert_eq!(disk.cmp(&memory), Ordering::Equal);
        assert_eq!(
            memory.path(),
            &ModulePath::memory(PathBuf::from("a/foo.py"))
        );
        assert_ne!(disk, other);
        assert_eq!(disk.cmp(&other), Ordering::Less);
        let bar = ModuleKey::new(
            ModuleName::from_str("bar"),
            &ModulePath::filesystem(PathBuf::from("z/bar.py")),
        );
        assert_eq!(bar.cmp(&disk), Ordering::Less);
    }

    #[test]
    fn test_exact_module_key_keeps_memory_and_disk_apart() {
        let name = ModuleName::from_str("foo");
        let disk = ModuleKey::new(name, &ModulePath::filesystem(PathBuf::from("a/foo.py")));
        let memory = ModuleKey::new(name, &ModulePath::memory(PathBuf::from("a/foo.py")));
        // The same module, but two modules to look up.
        assert_eq!(disk, memory);
        assert_ne!(disk.exact(), memory.exact());
        assert_eq!(
            disk.exact(),
            ModuleKey::new(name, &ModulePath::filesystem(PathBuf::from("a/foo.py"))).exact()
        );
    }

    #[test]
    fn test_module_key_one_file_two_names() {
        // One file imported under two module names is two modules.
        let path = ModulePath::filesystem(PathBuf::from("src/foo/shared.py"));
        let short = ModuleKey::new(ModuleName::from_str("foo.shared"), &path);
        let long = ModuleKey::new(ModuleName::from_str("src.foo.shared"), &path);
        assert_ne!(short, long);
        assert_ne!(short.exact(), long.exact());
        assert_eq!(short.path(), long.path());
    }
}
