/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests of the `State` object.

use std::env;
use std::mem;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lock::Mutex;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use pyrefly_util::thread_pool::ThreadCount;
use ruff_python_ast::name::Name;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::Solutions;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTParams;
use crate::binding::metadata::ClassRowChange;
use crate::config::config::ConfigFile;
use crate::config::error::ErrorDisplayConfig;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::error::print_errors;
use crate::state::errors::Errors;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::subscriber::TestSubscriber;
use crate::test::util::TestEnv;
use crate::test::util::init_test;
use crate::test::util::report_unreachable_at_warn;
use crate::types::types::Type;

#[derive(Default, Clone, Dupe, Debug)]
struct IncrementalData(Arc<Mutex<SmallMap<ModuleName, Arc<String>>>>);

/// Helper for writing incrementality tests.
struct Incremental {
    data: IncrementalData,
    files: Vec<String>,
    require: Option<Require>,
    state: State,
    to_set: Vec<(String, String)>,
}

/// What happened when we ran an incremental check.
struct IncrementalResult {
    changed: Vec<String>,
    errors: Errors,
}

impl IncrementalResult {
    fn check_recompute(&self, want: &[&str]) {
        let mut want = want.map(|x| (*x).to_owned());
        want.sort();
        assert_eq!(want, self.changed);
    }

    fn check_recompute_dedup(&self, want: &[&str]) {
        let mut changed = self.changed.clone();
        changed.dedup();
        let mut want = want.map(|x| (*x).to_owned());
        want.sort();
        assert_eq!(want, changed);
    }

    fn check_errors(&self) {
        self.errors.check_against_expectations().unwrap();
    }
}

impl Incremental {
    const USER_FILES: &[&str] = &["main", "foo", "bar", "baz"];

    fn new() -> Self {
        Self::with_files(Self::USER_FILES.map(|x| (*x).to_owned()))
    }

    fn with_files(files: Vec<String>) -> Self {
        Self::with_files_and_threads(files, TEST_THREAD_COUNT)
    }

    fn with_files_and_threads(files: Vec<String>, threads: ThreadCount) -> Self {
        Self::with_site_packages(files, threads, Vec::new())
    }

    /// Like `with_files_and_threads`, but imports that no file provides also resolve against the
    /// packages in `site_package_path`.
    fn with_site_packages(
        files: Vec<String>,
        threads: ThreadCount,
        site_package_path: Vec<PathBuf>,
    ) -> Self {
        init_test();
        let data = IncrementalData::default();

        let mut config = ConfigFile::default();
        if !site_package_path.is_empty() {
            config.python_environment.site_package_path = Some(site_package_path);
        }
        config.python_environment.set_empty_to_default();
        let mut sourcedb = MapDatabase::new(config.get_sys_info());
        for file in &files {
            sourcedb.insert(
                ModuleName::from_str(file),
                ModulePath::memory(PathBuf::from(file)),
            );
        }
        config.source_db = Some(ArcId::new(Box::new(sourcedb)));
        report_unreachable_at_warn(
            config
                .root
                .errors
                .get_or_insert_with(ErrorDisplayConfig::default),
        );
        config.configure();
        let config = ArcId::new(config);

        Self {
            data: data.dupe(),
            files,
            require: None,
            state: State::new(ConfigFinder::new_constant(config), threads),
            to_set: Vec::new(),
        }
    }

    /// Change this file to these contents, expecting this number of errors.
    fn set(&mut self, file: &str, contents: &str) {
        self.to_set.push((file.to_owned(), contents.to_owned()));
    }

    fn handle(&self, x: &str) -> Handle {
        Handle::new(
            ModuleName::from_str(x),
            ModulePath::memory(PathBuf::from(x)),
            SysInfo::default(),
        )
    }

    fn unchecked(&mut self, want: &[&str]) -> IncrementalResult {
        let subscriber = TestSubscriber::new();
        // Transitive deps default to `Require::Exports` so they stay
        // lazy (matching demand-driven solving in real-world usage).
        // Tests that need a higher level for transitive deps can set
        // `i.require` to override.
        let mut transaction = self.state.new_committable_transaction(
            self.require.unwrap_or(Require::Exports),
            Some(Box::new(subscriber.dupe())),
        );
        for (file, contents) in mem::take(&mut self.to_set) {
            let contents = Arc::new(contents.to_owned());
            self.data
                .0
                .lock()
                .insert(ModuleName::from_str(&file), contents.dupe());
            transaction.as_mut().set_memory(vec![(
                PathBuf::from(file),
                Some(Arc::new(FileContents::Source(contents))),
            )]);
        }

        let handles = want.map(|x| self.handle(x));
        self.state.run_with_committing_transaction(
            transaction,
            &handles,
            self.require.unwrap_or(Require::Everything),
            None,
            None,
        );
        let loaded = self
            .files
            .iter()
            .map(|x| self.handle(x))
            .collect::<Vec<_>>();
        let errors = self.state.transaction().get_errors(&loaded);
        let project_root = PathBuf::new();
        print_errors(project_root.as_path(), &errors.collect_display_errors());

        let mut changed = Vec::new();
        for (x, (count, _)) in subscriber.finish() {
            let m = x.module();
            if self.data.0.lock().contains_key(&m) {
                for _ in 0..count {
                    changed.push(m.as_str().to_owned());
                }
            }
        }
        changed.sort();
        IncrementalResult { changed, errors }
    }

    /// Run a check. Expect to recompute things to have changed and errors from # E: <> comments.
    fn check(&mut self, want: &[&str], recompute: &[&str]) -> IncrementalResult {
        let res = self.unchecked(want);
        res.check_errors();
        res.check_recompute(recompute);
        res
    }

    /// Run a check, and expect the same errors, at the same locations, as a check of the same files
    /// from scratch.
    fn check_matches_fresh_check(&mut self, want: &[&str]) {
        let incremental = self.unchecked(want);
        let mut fresh = Incremental::with_files(self.files.clone());
        for (module, contents) in self.data.0.lock().iter() {
            fresh.set(module.as_str(), contents);
        }
        let fresh = fresh.unchecked(want);
        let show = |result: &IncrementalResult| {
            result
                .errors
                .collect_display_errors()
                .iter()
                .map(|e| format!("{} {}: {}", e.path(), e.display_range(), e.msg()))
                .collect::<Vec<_>>()
        };
        assert_eq!(show(&incremental), show(&fresh));
    }

    /// Run a check, expecting to recompute `recompute`, and check that the modules in `want` have
    /// the errors and exported answers that checking the same sources from scratch gives.
    fn check_matches_clean(&mut self, want: &[&str], recompute: &[&str]) -> IncrementalResult {
        let res = self.unchecked(want);
        res.check_recompute(recompute);
        let mut clean = Incremental::with_files(self.files.clone());
        for (m, contents) in self.data.0.lock().iter() {
            clean.set(m.as_str(), contents);
        }
        let clean_res = clean.unchecked(want);
        let errors = res.errors.collect_display_errors();
        let clean_errors = clean_res.errors.collect_display_errors();
        for m in want {
            assert_eq!(
                module_snapshot(self, &errors, m),
                module_snapshot(&clean, &clean_errors, m),
                "`{m}` differs from a clean check"
            );
        }
        res
    }

    /// Run a check. Expect to recompute things to have changed, but ignore error comments.
    fn check_ignoring_expectations(
        &mut self,
        want: &[&str],
        recompute: &[&str],
    ) -> IncrementalResult {
        let res = self.unchecked(want);
        res.check_recompute(recompute);
        res
    }
}

#[test]
fn test_functools_wrapping_mutation_invalidates_unwrap() {
    let mut i = Incremental::new();
    let module = |mutation: &str| {
        format!(
            r#"
from functools import update_wrapper, wraps
from inspect import unwrap

def source() -> str:
    return "source"

@wraps(source)
def wrapper() -> int:
    return 1

result = unwrap(wrapper)()

def unrelated() -> bool:
    return False

{mutation}
"#
        )
    };
    for mutation in ["", "update_wrapper(unrelated, source)", ""] {
        i.set("main", &module(mutation));
        let expected = if mutation.is_empty() { "str" } else { "Any" };
        i.set(
            "foo",
            &format!("from main import result\nfrom typing import Any, assert_type\nassert_type(result, {expected})\n"),
        );
        i.unchecked(&["main", "foo"]).check_errors();
    }
}

#[test]
fn test_static_jaxtyping_declaration_rename_invalidates_consumer() {
    let mut i = Incremental::with_files(vec![
        "main".to_owned(),
        "consumer".to_owned(),
        "shape_extensions".to_owned(),
        "jaxtyping".to_owned(),
        "torch".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
from typing import Callable

class IntTuple: ...
class IntVar: ...
def static_jaxtyping[F: Callable](declaration: str) -> Callable[[F], F]: ...
"#,
    );
    i.set(
        "jaxtyping",
        r#"
from typing import Annotated as Float
"#,
    );
    i.set(
        "torch",
        r#"
from shape_extensions import IntTuple

class Tensor[Shape: IntTuple]: ...
"#,
    );
    let declared = |name: &str| {
        format!(
            r#"
from jaxtyping import Float
from shape_extensions import static_jaxtyping
from torch import Tensor

@static_jaxtyping("{name}")
def identity(x: Float[Tensor, "{name}"]) -> Float[Tensor, "{name}"]: ...
"#
        )
    };
    i.set("main", &declared("n"));
    i.set(
        "consumer",
        r#"
from main import identity
from typing import reveal_type

reveal_type(identity)
"#,
    );

    let initial = i.unchecked(&["consumer"]);
    let initial_errors = initial.errors.collect_display_errors();
    assert_eq!(initial_errors.len(), 1, "{initial_errors:?}");
    assert!(
        initial_errors[0]
            .msg()
            .contains("[n](x: Tensor[[n]]) -> Tensor[[n]]"),
        "{initial_errors:?}"
    );

    i.set("main", &declared("m"));
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer", "main"]);
    let changed_errors = changed.errors.collect_display_errors();
    assert_eq!(changed_errors.len(), 1, "{changed_errors:?}");
    assert!(
        changed_errors[0]
            .msg()
            .contains("[m](x: Tensor[[m]]) -> Tensor[[m]]"),
        "{changed_errors:?}"
    );
}

#[test]
fn test_index_shape_argument_edit_invalidates_consumer() {
    let mut i = Incremental::with_files(vec![
        "main".to_owned(),
        "consumer".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
from typing import Any
class IntTuple: pass
class Index: pass
def index_shape(shape: IntTuple, index: Any) -> IntTuple: ...
"#,
    );
    i.set(
        "main",
        r#"
from typing import Literal
from shape_extensions import IntTuple, index_shape

class Array[Shape: IntTuple]: ...
def selected() -> Array[index_shape(IntTuple[10, 20], slice[Literal[1], Literal[5], None])]: ...
"#,
    );
    i.set(
        "consumer",
        r#"
from typing import assert_type
from shape_extensions import IntTuple
from main import Array, selected

assert_type(selected(), Array[IntTuple[4, 20]])
"#,
    );
    i.check(&["consumer"], &["consumer", "main", "shape_extensions"]);

    i.set(
        "main",
        r#"
from typing import Literal
from shape_extensions import IntTuple, index_shape

class Array[Shape: IntTuple]: ...
def selected() -> Array[index_shape(IntTuple[10, 20], slice[Literal[1], Literal[7], None])]: ...
"#,
    );
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer", "main"]);
    let errors = changed.errors.collect_display_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .msg()
            .contains("assert_type(Array[[6, 20]], Array[[4, 20]]) failed")
    );
}

#[test]
fn test_type_shape_dsl_body_edit_invalidates_importer() {
    let mut i = Incremental::with_files(vec![
        "foo".to_owned(),
        "main".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class Int: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn
"#,
    );
    i.set(
        "foo",
        r#"
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def identity(x: Int) -> Int:
    return x  # version 1
"#,
    );
    i.set("main", "from foo import identity as identity");
    i.check(&["main"], &["foo", "main", "shape_extensions"]);

    i.set(
        "foo",
        r#"
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def identity(x: Int) -> Int:
    return (x)  # version 2
"#,
    );
    i.check(&["main"], &["foo", "main"]);
}

/// A list literal is inverted from its elements rather than its joined `list` type, so edits
/// to the elements must invalidate and recompute the call result.
#[test]
fn test_map_int_tuples_pattern_list_literal_edits_invalidate_consumer() {
    let mut i = Incremental::with_files(vec![
        "main".to_owned(),
        "consumer".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class IntTuple: pass
class IntTuples: pass
class MapIntTuples: pass
"#,
    );
    i.set(
        "main",
        r#"
from shape_extensions import IntTuple, IntTuples, MapIntTuples

class Box[Shape: IntTuple]: ...

def shapes[Shapes: IntTuples](
    values: MapIntTuples[lambda S: Box[S], Shapes],
) -> Shapes: ...

def box2() -> Box[IntTuple[2]]: ...
def box34() -> Box[IntTuple[3, 4]]: ...
"#,
    );
    i.set(
        "consumer",
        r#"
from main import box2, shapes
from shape_extensions import IntTuple
from typing import assert_type

assert_type(shapes([box2()]), tuple[IntTuple[2]])
"#,
    );
    let initial = i.unchecked(&["consumer"]);
    assert!(initial.errors.collect_display_errors().is_empty());

    i.set(
        "consumer",
        r#"
from main import box34, shapes
from shape_extensions import IntTuple
from typing import assert_type

assert_type(shapes([box34()]), tuple[IntTuple[2]])
"#,
    );
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer"]);
    let errors = changed.errors.collect_display_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].msg().contains("assert_type("));

    i.set(
        "consumer",
        r#"
from main import box2, shapes
from shape_extensions import IntTuple
from typing import assert_type

assert_type(shapes([box2(), box2()]), tuple[IntTuple[2], IntTuple[2]])
"#,
    );
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer"]);
    let errors = changed.errors.collect_display_errors();
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn test_shape_flag_constructor_source_edit_invalidates_consumers() {
    let mut i = Incremental::with_files(vec!["consumer".to_owned(), "main".to_owned()]);
    let main = |annotation: &str| {
        format!(
            r#"
type Carrier[T] = T
from shape_extensions import Flag

class Control[T: Flag[int]]:
    def __init__(self, value: {annotation}) -> None: ...
"#,
        )
    };
    i.set("main", &main("T"));
    i.set(
        "consumer",
        r#"
from main import Control

control = Control(value=1)
"#,
    );
    let initial = i.unchecked(&["consumer"]);
    assert!(initial.errors.collect_display_errors().is_empty());

    // The alias resolves to the same parameter type, so constructor-source metadata is the only
    // exported type information that changes.
    i.set("main", &main("Carrier[T]"));
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer", "main"]);
    assert!(changed.errors.collect_display_errors().is_empty());
}

#[test]
fn test_shape_flag_constructor_source_after_paramspec_edit_invalidates_consumers() {
    let mut i = Incremental::with_files(vec!["consumer".to_owned(), "main".to_owned()]);
    let main = |annotation: &str| {
        format!(
            r#"
type Carrier[T] = T
from shape_extensions import Flag

class Control[T: Flag[int]]:
    def __init__[**P](
        self, *args: P.args, value: {annotation}, **kwargs: P.kwargs
    ) -> None: ...
"#,
        )
    };
    i.set("main", &main("T"));
    i.set(
        "consumer",
        r#"
from main import Control

control = Control(value=1)
"#,
    );
    let initial = i.unchecked(&["consumer"]);
    assert!(initial.errors.collect_display_errors().is_empty());

    // ParamSpec variadics are removed from the lowered parameter list. Match the remaining
    // parameter to its source by name so this provenance change still invalidates consumers.
    i.set("main", &main("Carrier[T]"));
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer", "main"]);
    assert!(changed.errors.collect_display_errors().is_empty());
}

#[test]
fn test_type_shape_dsl_helper_edit_invalidates_caller() {
    let mut i = Incremental::with_files(vec![
        "helpers".to_owned(),
        "main".to_owned(),
        "consumer".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class Int: pass
class IntTuple: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn
"#,
    );
    i.set(
        "helpers",
        r#"
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def leaf(first: Int, second: Int) -> Int:
    return first
"#,
    );
    i.set(
        "main",
        r#"
from helpers import leaf
from shape_extensions import Int, IntTuple, type_shape_dsl_function

class Tensor[Shape: IntTuple]: ...

@type_shape_dsl_function
def middle(first: Int, second: Int) -> Int:
    return leaf(first, second)

@type_shape_dsl_function
def root(first: Int, second: Int) -> Int:
    return middle(first, second)

def chosen() -> Tensor[[root(Int[2], Int[3])]]: ...
"#,
    );
    i.set(
        "consumer",
        r#"
from main import Tensor, chosen

result: Tensor[[2]] = chosen()
"#,
    );
    let initial = i.unchecked(&["consumer"]);
    assert!(initial.errors.collect_display_errors().is_empty());

    i.set(
        "helpers",
        r#"
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def leaf(first: Int, second: Int) -> Int:
    return second
"#,
    );
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer", "helpers", "main"]);
    let errors = changed.errors.collect_display_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].msg().contains("not assignable to `Tensor[[2]]`"));
}

#[test]
fn test_type_shape_dsl_broadcast_edit_invalidates_importer() {
    let mut i = Incremental::with_files(vec![
        "main".to_owned(),
        "consumer".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class IntTuple: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn

@type_shape_dsl_function
def broadcast(left: IntTuple, right: IntTuple) -> IntTuple:
    return left
"#,
    );
    i.set(
        "main",
        r#"
from shape_extensions import IntTuple, broadcast

class Tensor[Shape: IntTuple]: ...

def chosen() -> Tensor[broadcast(IntTuple[2], IntTuple[3])]: ...
"#,
    );
    i.set(
        "consumer",
        r#"
from main import Tensor, chosen

result: Tensor[[2]] = chosen()
"#,
    );
    let initial = i.unchecked(&["consumer"]);
    assert!(initial.errors.collect_display_errors().is_empty());

    i.set(
        "shape_extensions",
        r#"
class IntTuple: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn

@type_shape_dsl_function
def broadcast(left: IntTuple, right: IntTuple) -> IntTuple:
    return right
"#,
    );
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer", "main", "shape_extensions"]);
    let errors = changed.errors.collect_display_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].msg().contains("not assignable to `Tensor[[2]]`"));
}

#[test]
fn test_type_shape_dsl_cross_module_helper_cycle_is_rejected() {
    let mut i = Incremental::with_files(vec![
        "left".to_owned(),
        "right".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class Int: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn
"#,
    );
    i.set(
        "left",
        r#"
from right import right
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def left(x: Int) -> Int:
    return right(x)
"#,
    );
    i.set(
        "right",
        r#"
from left import left
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def right(x: Int) -> Int:
    return left(x)
"#,
    );

    let result = i.unchecked(&["left"]);
    let errors = result.errors.collect_display_errors();
    assert_eq!(
        errors.len(),
        1,
        "expected one cycle diagnostic, got {errors:?}"
    );
    // Neither side of a helper cycle can become a validated DSL function, so cycles surface at
    // the ordinary helper-validation boundary rather than through an evaluator-specific error.
    assert!(
        errors[0]
            .msg()
            .contains("DSL helper callee must be a validated"),
        "expected the cross-module helper cycle to be rejected, got {errors:?}"
    );
}

#[test]
fn test_type_shape_dsl_gradual_reexport_target_invalidates_importer() {
    let mut i = Incremental::with_files(vec![
        "main".to_owned(),
        "reexport".to_owned(),
        "shape_extensions".to_owned(),
        "shape_extensions.dsl".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class Int: pass
class IntTuple: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn
"#,
    );
    i.set(
        "shape_extensions.dsl",
        r#"
from typing import Any
class Int:
    @staticmethod
    def gradual() -> Any: ...
class IntTuple:
    @staticmethod
    def gradual() -> Any: ...
"#,
    );
    i.set(
        "reexport",
        "from shape_extensions.dsl import Int as Gradual\n",
    );
    i.set(
        "main",
        r#"
from reexport import Gradual
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def gradual(x: Int) -> Int:
    return Gradual.gradual()
"#,
    );
    i.check(
        &["main"],
        &[
            "main",
            "reexport",
            "shape_extensions",
            "shape_extensions.dsl",
        ],
    );

    i.set(
        "reexport",
        "from shape_extensions.dsl import IntTuple as Gradual\n",
    );
    let changed = i.unchecked(&["main"]);
    changed.check_recompute(&["main", "reexport"]);
    assert!(changed.errors.collect_display_errors().iter().any(|error| {
        error
            .msg()
            .contains("declares return domain `Int`, but `shape_extensions.dsl.IntTuple.gradual()` returns `IntTuple`")
    }));

    // The restore leg covers re-resolving the intrinsic after an edit outside the declaration,
    // and that changing then restoring the reexport invalidates and recovers the importer result.
    i.set(
        "reexport",
        "from shape_extensions.dsl import Int as Gradual\n",
    );
    i.check(&["main"], &["main", "reexport"]);
}

#[test]
#[should_panic]
fn test_incremental_inception_errors() {
    let mut i = Incremental::new();
    i.set("main", "i: int = 'test'");
    i.check(&["main"], &["main"]);
}

#[test]
#[should_panic]
fn test_incremental_inception_recompute() {
    let mut i = Incremental::new();
    i.set("main", "i: int = 1");
    i.check(&["main"], &["main", "foo"]);
}

#[test]
fn test_in_memory_updated_content_recheck() {
    let mut i = Incremental::new();
    i.set("main", "unbound_name # E:");
    i.check(&["main"], &["main"]);
    i.set("main", "bound_name = 3");
    i.check(&["main"], &["main"]);
}

#[test]
fn test_incremental_minimal_recompute() {
    let mut i = Incremental::new();
    i.set("main", "import foo; x = foo.x");
    i.set("foo", "x = 7");
    i.check(&["main"], &["main", "foo"]);
    i.set("foo", "x = 'test'");
    i.check(&["main"], &["main", "foo"]);
    i.set("foo", "x = 'test' # still");
    i.check(&["main"], &["foo"]);
    i.set("main", "import foo; x = foo.x # still");
    i.check(&["main"], &["main"]);

    // We stop depending on `foo`, so no longer have to recompute it even though it is dirty.
    // However, our current state algorithm does so anyway as it can be cheaper to compute
    // everything than do careful graph traversal.
    i.set("foo", "x = True");
    i.set("main", "x = 7");
    i.check(&["main"], &["main", "foo"]); // `foo` is not required here
    i.set("main", "import foo; x = foo.x # still");
    i.check(&["main"], &["main"]); // `foo` is required by this point
}

#[test]
fn test_incremental_cyclic() {
    let mut i = Incremental::new();
    i.set("foo", "import bar; x = 1; y = bar.x");
    i.set("bar", "import foo; x = True; y = foo.x");
    // Initial check fully checks both modules so deps get recorded
    // both ways; subsequent checks rely on those recorded deps to
    // drive invalidation.
    i.check(&["foo", "bar"], &["foo", "bar"]);
    i.set("foo", "import bar; x = 1; y = bar.x # still");
    i.check(&["foo"], &["foo"]);
    i.set("foo", "import bar; x = 'test'; y = bar.x");
    i.check(&["foo"], &["foo", "bar"]);
}

/// Check that the interface is consistent as we change things.
fn test_interface_consistent(code: &str) {
    let mut i = Incremental::new();
    i.set("main", code);
    i.check(&["main"], &["main"]);
    let base = i
        .state
        .transaction()
        .get_solutions(&i.handle("main"))
        .unwrap();

    i.set("main", &format!("{code} # after"));
    i.check(&["main"], &["main"]);
    let suffix = i
        .state
        .transaction()
        .get_solutions(&i.handle("main"))
        .unwrap();

    i.set("main", &format!("# before\n{code}"));
    i.check(&["main"], &["main"]);
    let prefix = i
        .state
        .transaction()
        .get_solutions(&i.handle("main"))
        .unwrap();

    let same = base.first_difference(&base);
    let suffix = suffix.first_difference(&base);
    let prefix = prefix.first_difference(&base);
    assert!(same.is_none(), "{code:?} led to {same:?}");
    assert!(suffix.is_none(), "{code:?} led to {suffix:?}");
    assert!(prefix.is_none(), "{code:?} led to {prefix:?}");
}

#[test]
fn test_interfaces_simple() {
    test_interface_consistent("x: int = 1\ndef f(y: bool) -> list[str]: return []");

    // Important to have a class with a field, as those also have positions
    test_interface_consistent("class X: y: int");
}

#[test]
fn test_interfaces_generic() {
    // Requires dealing with Forall.
    test_interface_consistent("def f[X](x: X) -> X: ...");
    test_interface_consistent(
        "
from typing import TypeVar, Generic
T = TypeVar('T')
class C(Generic[T]): pass",
    );
    test_interface_consistent("class C[T]: x: T");
}

#[test]
fn test_interfaces_counterexamples() {
    // These all failed at one point or another.

    test_interface_consistent(
        "
from typing import TypeVar, Generic
T = TypeVar('T')
class C(Generic[T]): x: T",
    );

    test_interface_consistent(
        "
from typing import TypeVar, Generic
T = TypeVar('T')
class C(Generic[T]): pass
class D(C[T]): pass",
    );

    test_interface_consistent(
        "
from typing import TypeVar
class C: pass
T = TypeVar('T', bound=C)",
    );

    test_interface_consistent(
        "
class C[R]:
    def __init__(self, field: R) -> None:
        self.field = R
",
    );
}

#[test]
fn test_error_clearing_on_dependency() {
    let mut i = Incremental::new();

    i.set("foo", "def xyz() -> int: ...");
    i.set(
        "main",
        "from foo import x # E: Could not import `x` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();

    assert!(
        !errors.ordinary.is_empty(),
        "Expected errors before fixing the dependency"
    );

    i.set("foo", "def x() -> int: ...");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after_fix = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after_fix.ordinary.is_empty(),
        "Expected errors after fixing the dependency"
    );
}

#[test]
fn test_error_clearing_on_dependency_star_import() {
    let mut i = Incremental::new();

    i.set("foo", "def xyz() -> int: ...");
    i.set(
        "main",
        "from foo import *\ny = x # E: Could not find name `x`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();

    assert!(
        !errors.ordinary.is_empty(),
        "Expected errors before fixing the dependency"
    );

    i.set("foo", "def xyz() -> int: ...\nx = 1");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after_fix = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after_fix.ordinary.is_empty(),
        "Expected no errors after fixing the dependency"
    );
}

#[test]
fn test_failed_import_invalidation_via_rdeps() {
    // This tests that when a module's exports change to satisfy a previously-failed import,
    // the module with the failed import is invalidated even if it's not explicitly requested.
    //
    // The key difference from test_error_clearing_on_dependency is that we DON'T include
    // the module with the failed import in the `want` list - we only request a third module
    // that depends on it transitively.
    let mut i = Incremental::new();

    // Setup: bar has a failed import from foo
    i.set("foo", "x = 1"); // foo exists but doesn't export `y`
    i.set("bar", "from foo import y"); // bar tries to import y - FAILS
    i.set("main", "import bar"); // main imports bar

    // Initial check fully checks bar so its failed import gets
    // registered for `invalidate_failed_imports_from` to find later.
    i.unchecked(&["main", "bar"]);

    // Now foo exports `y` - bar's failed import should now succeed
    i.set("foo", "y = 2");

    // Only request main, NOT bar directly.
    // bar wouldn't be invalidated through normal rdeps because:
    //   - bar is not in foo's rdeps (the import failed)
    //   - bar is not in the `want` list
    // But invalidate_failed_imports_from scans for failed imports and invalidates bar.
    let res = i.unchecked(&["main"]);

    // bar should be recomputed because its failed import now succeeds
    assert!(
        res.changed.contains(&"bar".to_owned()),
        "bar should have been recomputed, but changed = {:?}",
        res.changed
    );
}

#[test]
fn test_stale_class() {
    let mut i = Incremental::new();
    i.set("foo", "class C: x: int = 1");
    i.set("bar", "from foo import C; c = C");
    i.set("main", "from bar import c; v = c.x");
    // Initial check fully checks bar so its `c = C` binding gets
    // solved and a TypeEq dep on `foo::C` is recorded; that dep is
    // what drives bar's invalidation when `C` is removed.
    i.check(&["main", "bar"], &["main", "foo", "bar"]);

    i.set("foo", "");
    i.set("main", "from bar import c; v = c.x # hello");
    let res = i.unchecked(&["main", "foo"]);
    res.check_recompute_dedup(&["main", "foo", "bar"]);
    assert_eq!(res.errors.collect_errors().ordinary.len(), 1);
}

#[test]
fn test_stale_typed_dict() {
    let mut i = Incremental::new();

    // We need to set up a dep chain of size 4 (i.e. main -> bar -> baz -> foo) to more reliably
    // force `main` to see a stale TypedDict in `bar` during the recheck.
    // It may still be possible to hide the staleness in certain circumstances, but that's fine since
    // the test would still pass in those cases.
    i.set(
        "foo",
        "from typing import TypedDict\nclass D(TypedDict):\n  x: int",
    );
    i.set("bar", "from foo import D\nclass D2:\n  y: D");
    i.set("baz", "from bar import D2\nclass D3:\n  z: D2");
    i.set(
        "main",
        "from baz import D3\ndef test(d: D3) -> None:\n  d.z.y[\'x\']",
    );
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    i.set("foo", "class D: x: int");

    // `baz` uses `D2` without reading anything of `D`, so it is not rechecked.
    i.check_matches_clean(&["main"], &["main", "foo", "bar"]);
}

#[test]
fn test_type_shape_dsl_body_edit_invalidates_consumer() {
    let mut i = Incremental::with_files(vec![
        "definitions".to_owned(),
        "main".to_owned(),
        "consumer".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class Int: pass
class IntTuple: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn
"#,
    );
    i.set(
        "definitions",
        r#"
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def select(first: Int, second: Int) -> Int:
    return first
"#,
    );
    i.set(
        "main",
        r#"
from definitions import select
from shape_extensions import Int, IntTuple

class Tensor[Shape: IntTuple]: ...
def result() -> Tensor[[select(Int[2], Int[3])]]: ...
"#,
    );
    i.set(
        "consumer",
        r#"
from main import Tensor, result

expected: Tensor[[2]] = result()
"#,
    );
    let initial = i.unchecked(&["consumer"]);
    assert!(initial.errors.collect_display_errors().is_empty());

    i.set(
        "definitions",
        r#"
from shape_extensions import Int, type_shape_dsl_function

@type_shape_dsl_function
def select(first: Int, second: Int) -> Int:
    return second
"#,
    );
    let changed = i.unchecked(&["consumer"]);
    changed.check_recompute(&["consumer", "definitions", "main"]);
    let errors = changed.errors.collect_display_errors();
    assert_eq!(
        errors.len(),
        1,
        "expected one consumer mismatch, got {errors:?}"
    );
    assert_eq!(errors[0].module().name(), ModuleName::from_str("consumer"),);
    assert!(
        errors[0].msg().contains("not assignable to `Tensor[[2]]`"),
        "expected a consumer assignment mismatch, got {errors:?}",
    );
}

#[test]
fn test_incremental_cycle_class() {
    let mut i = Incremental::new();
    i.set("foo", "from bar import Cls");
    i.set(
        "bar",
        r#"
from foo import Cls as C
class Cls:
    def fld(self): return 1
def f(c: C):
    print(c.fld)
"#,
    );
    i.check(&["foo", "bar"], &["foo", "bar"]);

    i.set(
        "bar",
        r#"
from foo import Cls as C
class Cls:
    def fld2(self): return 1
def f(c: C):
    print(c.fld) # E: Object of class `Cls` has no attribute `fld`
"#,
    );
    i.unchecked(&["foo"]); // Used to panic
}

#[test]
fn test_incremental_rdeps() {
    // Make sure we hit the rdeps case
    let mut i = Incremental::new();
    i.require = Some(Require::Everything); // So we don't invalidate based on require
    i.set("foo", "import bar\nclass C: pass\nx = bar.z");
    i.set("bar", "import foo\nz = foo.C\nq: type[foo.C] = foo.x");
    i.check(&["foo"], &["foo", "bar"]);

    i.set("foo", "import bar\nclass Q: pass\nx = bar.z\nclass C: pass");
    i.check(&["foo"], &["bar", "bar", "foo", "foo"]);

    i.check(&["foo", "bar"], &[]); // Nothing appears dirty
}

#[test]
fn test_incremental_rdeps_with_new() {
    // Make sure we hit the rdeps case
    let mut i = Incremental::new();
    i.require = Some(Require::Everything); // So we don't invalidate based on require
    i.set("foo", "import bar\nclass C: pass\nx = bar.z");
    i.set("bar", "import foo\nz = foo.C\nq: type[foo.C] = foo.x");
    i.check(&["foo"], &["foo", "bar"]);

    i.set(
        "foo",
        "import bar\nimport baz\nclass Q: pass\nx = bar.z\nclass C: pass",
    );
    i.set("baz", "import bar");
    i.unchecked(&["foo"]);

    i.check(&["foo", "bar", "baz"], &[]); // Nothing appears dirty
}

/// Test fine-grained dependency tracking: changing an unrelated export should NOT
/// trigger recomputation of a module that only imports a different export.
#[test]
fn test_fine_grained_unrelated_export_no_recompute() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set("main", "from foo import x\nz = x + 1");
    i.check(&["main"], &["main", "foo"]);

    // Change only `y` - main should NOT be recomputed since it only imports `x`
    i.set("foo", "x: int = 1\ny: str = 'changed'");
    i.check(&["main"], &["foo"]);
}

/// Changing an export's type (not its existence) shouldn't invalidate
/// consumers whose only edge on it is an `export_exists` check.
///
/// `bar`'s `from foo import Foo` calls `export_exists(foo, "Foo")`
/// during binding. `main` only imports `value` from `bar`, so `bar`'s
/// `Binding::Import(foo, Foo)` is never solved and no TypeEq dep on
/// `foo::Foo` is recorded — leaving the `export_exists` call as the
/// only edge. Foo's type changing should not invalidate bar.
#[test]
fn test_export_exists_is_existence_level() {
    let mut i = Incremental::new();
    i.set("foo", "Foo: int = 1");
    i.set("bar", "from foo import Foo\nvalue: int = 42");
    i.set("main", "from bar import value\nresult = value");
    // `foo` is also requested so its TypeEq changes get diffed at
    // Solutions level — otherwise the change-detection machinery
    // never fires.
    i.check(&["main", "foo"], &["main", "foo", "bar"]);

    i.set("foo", "Foo: str = 'changed'");
    i.check(&["foo"], &["foo"]);
}
/// Test fine-grained dependency tracking: changing the imported export SHOULD
/// trigger recomputation.
#[test]
fn test_fine_grained_related_export_recompute() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set("main", "from foo import x\nprint(x)");
    i.check(&["main"], &["main", "foo"]);

    // Change `x` - main SHOULD be recomputed since it imports `x`
    i.set("foo", "x: str = 'changed'\ny: int = 2");
    i.check(&["main"], &["foo", "main"]);
}

#[test]
fn test_parse_error_survives_dependency_recompute() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1");
    i.set(
        "main",
        "from foo import x\nprint(x)\nif  # E: Parse error: Expected an expression",
    );
    i.check(&["main"], &["main", "foo"]);

    i.set("foo", "x: str = 'changed'");
    i.check(&["main"], &["foo", "main"]);
}

/// Test fine-grained tracking with unused `import foo` style.
#[test]
fn test_import_module_regular_import_unused() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set("main", "import foo\nz = foo.x");
    i.check(&["main"], &["main", "foo"]);

    i.set("foo", "x: int = 1\ny: str = 'changed'");
    i.check(&["main"], &["foo"]);
}

/// Test mixed import styles: `from foo import x` followed by `import foo`.
/// Should only depend on x.
#[test]
fn test_mixed_import_depends_on_all() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1\ny: int = 2");
    i.set(
        "main",
        "from foo import x\nimport foo\nprint(x); print(foo.y)",
    );
    i.check(&["main"], &["main", "foo"]);

    // Change only `y` - main SHOULD be recomputed because of `import foo`
    i.set("foo", "x: int = 1\ny: str = 'changed'");
    i.check(&["main"], &["foo", "main"]);
}

/// Test incremental behavior with `import foo; bar(foo)` pattern.
#[test]
fn test_import_module_as_argument() {
    let mut i = Incremental::new();
    i.set("foo", "x: int = 1");
    i.set(
        "bar",
        "import types\ndef process(m: types.ModuleType) -> int: return m.x",
    );
    i.set("main", "import foo\nimport bar\nresult = bar.process(foo)");
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change `x` in foo - main should not recompute since it doesn't depend on the type
    i.set("foo", "x: str = 'changed'");
    i.check(&["main"], &["foo"]);
}

/// Test transitive export addition: when a module adds an export that a downstream
/// module re-exports, consumers of the re-export should see the error go away.
///
/// Scenario:
/// - foo (a.py): initially empty
/// - bar (b.py): `from foo import *` (re-exports everything from foo)
/// - main (c.py): `from bar import x` (fails because foo doesn't export x)
///
/// After foo adds `x = 1`, main's import should succeed.
#[test]
fn test_transitive_export_addition_clears_error() {
    let mut i = Incremental::new();

    // Initial state: foo is empty, bar re-exports from foo, main tries to import x from bar
    i.set("foo", "");
    i.set("bar", "from foo import *");
    i.set(
        "main",
        "from bar import x # E: Could not import `x` from `bar`",
    );
    i.check(&["main", "foo", "bar"], &["main", "foo", "bar"]);

    let main_handle = i.handle("main");

    // Verify there's an error before the fix
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.ordinary.is_empty(),
        "Expected errors before foo exports x"
    );

    // Now foo exports x - main's import should succeed
    i.set("foo", "x = 1");
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);

    // Verify the error is gone
    let errors_after_fix = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after_fix.ordinary.is_empty(),
        "Expected no errors after foo exports x, but got: {:?}",
        errors_after_fix.ordinary
    );
}

/// Test that when a type is used via inference (not explicitly imported),
/// changes to that type still trigger recomputation.
///
/// Scenario:
/// - foo: exports class A with field x: int
/// - bar: imports A, creates instance, and re-exports it
/// - main: imports the instance from bar (gets type A via inference, not import)
///
/// When A's field type changes, main should see the update.
#[test]
fn test_inferred_type_changes_trigger_recompute() {
    let mut i = Incremental::new();

    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\ninstance = A()");
    i.set("main", "from bar import instance\ny = instance.x + 1");
    i.check(&["main", "foo", "bar"], &["main", "foo", "bar"]);

    // Change A's field type from int to str - main's arithmetic should now fail
    i.set("foo", "class A:\n    x: str = 'hello'");
    i.set(
        "main",
        "from bar import instance\ny = instance.x + 1 # E: `+` is not supported",
    );
    // `bar` constructs an `A` without reading `x`, so it is not rechecked.
    i.check(&["main"], &["foo", "main"]);
}

/// A property defined through a descriptor class moves when an edit above it changes the length
/// of the file. Its answer does not record its location, so a module that reads the property is
/// not recomputed.
#[test]
fn test_edit_above_descriptor() {
    let mut i = Incremental::new();
    let foo = |value: &str| {
        format!(
            r#"
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

class lazy_property(Generic[T, R]):
    def __init__(self, fget: Callable[[type[T]], R]) -> None: ...
    def __get__(self, obj: object, owner: type[T]) -> R: ...

class A:
    def f(self) -> int:
        return {value}

    @lazy_property
    def p(cls: type["A"]) -> int:
        return 1
"#
        )
    };
    i.set("foo", &foo("1"));
    i.set("main", "from foo import A\nx: int = A.p");
    i.check(&["main", "foo"], &["main", "foo"]);

    i.set("foo", &foo("100000"));
    i.check(&["main", "foo"], &["foo"]);
}

/// The location of a class's `@total_ordering` decorator moves when an edit above the class changes
/// the length of the file. It only says where to report errors, so a module that uses the class is
/// not recomputed.
#[test]
fn test_edit_above_total_ordering() {
    let mut i = Incremental::new();
    let foo = |value: &str| {
        format!(
            r#"
from functools import total_ordering

def f() -> int:
    return {value}

@total_ordering
class A:
    def __eq__(self, other: object) -> bool:
        return True
    def __lt__(self, other: "A") -> bool:
        return True
"#
        )
    };
    i.set("foo", &foo("1"));
    i.set("main", "from foo import A\nx: bool = A() <= A()");
    i.check(&["main", "foo"], &["main", "foo"]);

    i.set("foo", &foo("100000"));
    i.check(&["main", "foo"], &["foo"]);
}

/// Test that when a function's return type changes, callers that import only
/// the function (not the return type) still see the update.
///
/// Scenario:
/// - foo: exports class A with field x and function get_a() -> A
/// - main: imports only get_a, uses the returned value's field
///
/// When A's field type changes, main should see the update.
#[test]
fn test_function_return_type_changes_trigger_recompute() {
    let mut i = Incremental::new();

    i.set(
        "foo",
        "class A:\n    x: int = 1\ndef get_a() -> A:\n    return A()",
    );
    i.set(
        "main",
        "from foo import get_a\nval = get_a()\ny = val.x + 1",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    // Change A's field type from int to str - main's arithmetic should now fail
    i.set(
        "foo",
        "class A:\n    x: str = 'hello'\ndef get_a() -> A:\n    return A()",
    );
    i.set(
        "main",
        "from foo import get_a\nval = get_a()\ny = val.x + 1 # E: `+` is not supported",
    );
    i.check(&["main"], &["foo", "main"]);
}

/// Test that non-overlapping export changes do NOT trigger false cycle detection.
///
/// This simulates a pattern similar to PyTorch's torch.distributed.pipelining.stage module,
/// which imports and re-exports from multiple independent sources:
///   - `from torch.distributed._composable.replicate_with_fsdp import replicate`
///   - `from torch.distributed.fsdp import fully_shard`
///
/// When both sources change in different epochs, the same module (stage) appears multiple
/// times in the change propagation - but with DIFFERENT exports. This should NOT be
/// treated as a cycle because:
///   1. The exports don't overlap - they're independent re-exports
///   2. Each export chain will stabilize independently
///   3. There's no infinite loop risk since no single export keeps changing
#[test]
fn test_non_overlapping_exports_no_false_cycle() {
    let mut i = Incremental::new();

    i.set("foo", "x: int = 1");
    i.set("bar", "y: int = 2");
    i.set("baz", "from foo import x\nfrom bar import y");
    i.set("main", "from baz import x, y\nprint(x, y)");
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    // Now change BOTH sources simultaneously. The hub module will:
    // 1. First re-export the changed `x` (from foo)
    // 2. Then re-export the changed `y` (from bar)
    // This should NOT trigger cycle detection since the exports don't overlap.
    i.set("foo", "x: str = 'changed_x'");
    i.set("bar", "y: str = 'changed_y'");
    i.check(&["main"], &["foo", "bar", "baz", "main"]);
}

/// Test that overlapping export changes DO trigger proper cycle detection.
///
/// A true cycle occurs when the same export keeps changing:
///   - Export X in A depends on export Y in B
///   - Export Y in B depends on export X in A
///   - X changes -> Y changes -> X changes -> would loop forever
///
/// The cycle detection should catch this and force invalidation.
#[test]
fn test_overlapping_exports_cycle_detected() {
    let mut i = Incremental::new();

    // Set up a mutual dependency cycle where both modules export
    // values that depend on each other.
    i.set("foo", "import bar\nx: int = 1\ny = bar.x");
    i.set("bar", "import foo\nx: int = 2\ny = foo.x");
    // Fully check both modules so the cycle deps get recorded.
    i.check(&["foo", "bar"], &["foo", "bar"]);

    // Changing `x` in foo should propagate to bar (which uses foo.x),
    // and potentially back to foo (if bar.x changes). The same export `x`
    // may need to be recomputed multiple times, triggering cycle detection.
    i.set("foo", "import bar\nx: str = 'changed'\ny = bar.x");

    // The cycle detection should handle this gracefully.
    // We use unchecked because the exact recomputation pattern depends on
    // cycle detection behavior.
    let res = i.unchecked(&["foo"]);
    // Both modules should be recomputed to reach stable state
    assert!(res.changed.contains(&"foo".to_owned()));
    assert!(res.changed.contains(&"bar".to_owned()));
}

// Synthetic defense-in-depth reproducer for https://github.com/pyrefly-org/pyrefly/issues/4171.
#[test]
fn test_deep_scc_chain_stabilizes_after_epoch_cap() {
    const LEVELS: usize = 208;

    let mut files = vec!["leaf".to_owned(), "main".to_owned()];
    for level in 0..LEVELS {
        files.push(format!("a_{level}"));
        files.push(format!("b_{level}"));
    }
    let mut i = Incremental::with_files(files);

    i.set("leaf", "value: int = 0");
    for level in 0..LEVELS {
        let previous = if level == 0 {
            "leaf".to_owned()
        } else {
            format!("a_{}", level - 1)
        };
        i.set(
            &format!("a_{level}"),
            &format!(
                "from typing import TYPE_CHECKING\nfrom {previous} import value as value\nif TYPE_CHECKING:\n    from b_{level} import Marker\n"
            ),
        );
        i.set(
            &format!("b_{level}"),
            &format!(
                "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from a_{level} import value\nclass Marker: pass\n"
            ),
        );
    }
    i.set(
        "main",
        &format!(
            "from a_{} import value\nobserved: int = value\n",
            LEVELS - 1
        ),
    );

    let initial = i.unchecked(&["main"]);
    assert_eq!(initial.errors.collect_display_errors().len(), 0);

    i.set("leaf", "value: str = 'changed'");
    let changed = i.unchecked(&["main"]);
    let errors = changed
        .errors
        .collect_display_errors()
        .map(|error| error.msg().to_owned());
    assert_eq!(
        errors.len(),
        1,
        "expected final importer error, got {errors:?}"
    );
    assert!(errors[0].contains("not assignable to `int`"), "{errors:?}");
}

/// Editing `z` changes `foo.x` twice, once directly and once through `bar.y`, which cycle detection
/// reports as a cycle. The recheck that follows recomputes `foo`, `bar` and `main` together, and
/// `bar.w` and `main.v` change in it.
#[test]
fn test_changes_after_cycle_detection_are_committed() {
    let mut i = Incremental::new();
    i.set("foo", "import bar\nz = 1\nx = (bar.y, z)");
    i.set("bar", "import foo\ny = foo.z\nw = foo.x");
    i.set("main", "import bar\nv = bar.w");
    i.check(&["foo", "bar", "main"], &["bar", "foo", "main"]);

    i.set("foo", "import bar\nz = ''\nx = (bar.y, z)");
    i.check_matches_clean(
        &["foo", "bar", "main"],
        &["bar", "bar", "foo", "foo", "foo", "main", "main"],
    );
}

/// Test a more complex non-overlapping case with a chain of re-exports.
///
/// This models a longer dependency chain where multiple intermediate modules
/// re-export from different sources, similar to:
///   torch.distributed.fsdp -> torch.distributed.fsdp._fully_shard -> _fully_shard.py
#[test]
fn test_reexport_chain_non_overlapping() {
    let mut i = Incremental::new();

    // Create a chain: source -> intermediate -> hub -> main
    // with two parallel chains that don't share exports
    i.set("foo", "a: int = 1\nb: int = 2");
    i.set("bar", "from foo import a"); // bar re-exports only `a`
    i.set("baz", "from foo import b"); // baz re-exports only `b`
    i.set("main", "from bar import a\nfrom baz import b");
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    // Change `a` - only bar and main should be affected (fine-grained tracking)
    i.set("foo", "a: str = 'new_a'\nb: int = 2");
    i.check(&["main"], &["foo", "bar", "main"]);

    // Change `b` - only baz and main should be affected
    i.set("foo", "a: str = 'new_a'\nb: str = 'new_b'");
    i.check(&["main"], &["foo", "baz", "main"]);
}

#[test]
fn test_class_field_type_change_propagates() {
    let mut i = Incremental::new();

    // Set up a chain: main -> baz -> bar -> foo
    // where foo.A has a field, bar.B uses A, baz.C uses B, main uses C
    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set("baz", "from bar import B\nclass C:\n    b: B");
    i.set(
        "main",
        "from baz import C\ndef f(c: C) -> int:\n    return c.b.a.x",
    );
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    // Change A's field type from int to str - modules should be recomputed
    i.set("foo", "class A:\n    x: str = 'hello'");
    // `bar` declares a field of type `A` without reading `x`, so it is not rechecked.
    i.check_matches_clean(&["main"], &["main", "foo"]);
}

/// Test that class base type change propagates through the dependency chain.
///
/// When a class's base class changes, modules that use the derived class
/// should be recomputed.
#[test]
fn test_class_base_type_change_propagates() {
    let mut i = Incremental::new();

    // foo.Base has method m, bar.Derived extends Base, main uses Derived
    i.set("foo", "class Base:\n    def m(self) -> int: return 1");
    i.set("bar", "from foo import Base\nclass Derived(Base): pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.m()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change Base's method return type - main should be recomputed
    i.set("foo", "class Base:\n    def m(self) -> str: return 'hello'");
    // `bar` declares `Derived` without reading `m`, so it is not rechecked.
    i.check_matches_clean(&["main"], &["main", "foo"]);
}

/// Test that class MRO change propagates when a base class is added/removed.
///
/// Changing the inheritance hierarchy should invalidate dependents.
#[test]
fn test_class_mro_change_propagates() {
    let mut i = Incremental::new();

    // foo.Mixin has method mix, bar.C does NOT inherit from Mixin
    // Note: foo is not in the dependency chain yet since bar doesn't import it
    i.set("foo", "class Mixin:\n    def mix(self) -> int: return 1");
    i.set("bar", "class C: pass");
    i.set(
        "main",
        "from bar import C\ndef f(c: C):\n    c.mix() # E: Object of class `C` has no attribute `mix`",
    );
    i.check(&["main"], &["main", "bar"]);

    // Change C to inherit from Mixin - main's error should go away
    // Now foo is part of the dependency chain since bar imports it
    i.set("bar", "from foo import Mixin\nclass C(Mixin): pass");
    i.set("main", "from bar import C\ndef f(c: C):\n    c.mix()");
    i.check(&["main"], &["main", "foo", "bar"]);
}

/// Test that base class field changes propagate when only the derived class is imported.
///
/// When main imports only Derived (not Base), and Base's fields change,
/// main should still see the updated fields through Derived.
#[test]
fn test_base_class_field_change_derived_import_only() {
    let mut i = Incremental::new();

    // foo.Base has field x, bar.Derived extends Base, main uses Derived.x
    i.set("foo", "class Base:\n    x: int = 1");
    i.set("bar", "from foo import Base\nclass Derived(Base): pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.x",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change x to str - main should see the updated type (error)
    i.set("foo", "class Base:\n    x: str = 'hello'");
    // `bar` declares `Derived` without reading `x`, so it is not rechecked.
    i.check_matches_clean(&["foo", "main"], &["main", "foo"]);
}

/// Test that dataclass field changes propagate through the dependency chain.
///
/// Dataclass synthesized fields (like __init__ parameters) should trigger
/// recomputation when they change.
#[test]
fn test_dataclass_field_change_propagates() {
    let mut i = Incremental::new();

    // foo.Data is a dataclass with field x: int
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: int",
    );
    i.set("bar", "from foo import Data\nclass Wrapper:\n    d: Data");
    i.set(
        "main",
        "from bar import Wrapper\ndef f(w: Wrapper) -> int:\n    return w.d.x",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change the dataclass field type - main should be recomputed
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: str",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that adding a new field to a class propagates correctly.
///
/// When a class gains a new field, modules using that class should be
/// recomputed so they can access the new field.
#[test]
fn test_class_field_addition_propagates() {
    let mut i = Incremental::new();

    // foo.A has only field x
    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B):\n    b.a.y # E: Object of class `A` has no attribute `y`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add field y to A - main's error should go away
    i.set("foo", "class A:\n    x: int = 1\n    y: int = 2");
    i.set("main", "from bar import B\ndef f(b: B):\n    b.a.y");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that removing a field from a class propagates correctly.
///
/// When a class loses a field, modules using that field should be
/// recomputed and see the error.
#[test]
fn test_class_field_removal_propagates() {
    let mut i = Incremental::new();

    // foo.A has fields x and y
    i.set("foo", "class A:\n    x: int = 1\n    y: int = 2");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B) -> int:\n    return b.a.y",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Remove field y from A - main should see an error
    i.set("foo", "class A:\n    x: int = 1");
    i.set(
        "main",
        "from bar import B\ndef f(b: B) -> int:\n    return b.a.y # E: Object of class `A` has no attribute `y`",
    );
    // Removing `y` changes the fields that `A` declares. `main` looks `y` up in them, and `bar`
    // reads them while it solves `B.a`, so both are rechecked.
    i.check(&["main"], &["bar", "foo", "main"]);
}

/// Test that method signature changes propagate through the dependency chain.
///
/// When a class method's signature changes, callers should be recomputed.
#[test]
fn test_class_method_signature_change_propagates() {
    let mut i = Incremental::new();

    // foo.A has method m(self) -> int
    i.set("foo", "class A:\n    def m(self) -> int: return 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B) -> int:\n    return b.a.m()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change method to require an argument - main should see an error
    i.set("foo", "class A:\n    def m(self, n: int) -> int: return n");
    // `bar` declares a field of type `A` without reading `m`, so it is not rechecked.
    i.check_matches_clean(&["main"], &["main", "foo"]);
}

/// Test generic class type parameter changes propagate.
///
/// When a generic class's type parameters change, modules using the class
/// with specific type arguments should be recomputed.
#[test]
fn test_generic_class_type_param_change_propagates() {
    let mut i = Incremental::new();

    // foo.Container is a simple class with field items: list[int]
    i.set("foo", "class Container:\n    items: list[int]");
    i.set(
        "bar",
        "from foo import Container\nclass Wrapper:\n    c: Container",
    );
    i.set(
        "main",
        "from bar import Wrapper\ndef f(w: Wrapper) -> int:\n    return w.c.items[0]",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change items to list[str] - main should see a type error
    i.set("foo", "class Container:\n    items: list[str]");
    // `bar` declares a field of type `Container` without reading `items`, so it is not
    // rechecked.
    i.check_matches_clean(&["main"], &["main", "foo"]);
}

/// Test that NamedTuple field changes propagate correctly.
///
/// NamedTuple has synthesized fields similar to dataclass.
#[test]
fn test_namedtuple_field_change_propagates() {
    let mut i = Incremental::new();

    // foo.Point is a NamedTuple with x: int, y: int
    i.set(
        "foo",
        "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: int\n    y: int",
    );
    i.set(
        "bar",
        "from foo import Point\nclass Line:\n    start: Point",
    );
    i.set(
        "main",
        "from bar import Line\ndef length(l: Line) -> int:\n    return l.start.x + l.start.y",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change field types to str - main should be recomputed
    i.set(
        "foo",
        "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: str\n    y: str",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that Protocol changes propagate correctly.
///
/// When a Protocol's method signature changes, implementors and users
/// should be recomputed.
#[test]
fn test_protocol_change_propagates() {
    let mut i = Incremental::new();

    // foo.Proto is a Protocol with method m() -> int
    i.set(
        "foo",
        "from typing import Protocol\nclass Proto(Protocol):\n    def m(self) -> int: ...",
    );
    i.set(
        "bar",
        "from foo import Proto\nclass Impl:\n    def m(self) -> int: return 1",
    );
    i.set(
        "main",
        "from foo import Proto\nfrom bar import Impl\ndef f(p: Proto) -> int:\n    return p.m()\nx: Proto = Impl()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change Protocol method to return str - main should be recomputed
    i.set(
        "foo",
        "from typing import Protocol\nclass Proto(Protocol):\n    def m(self) -> str: ...",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo"]);
}

/// Test four-level dependency chain with class field change.
///
/// This is a more thorough version of test_stale_typed_dict that verifies
/// propagation through a 4-level chain.
#[test]
fn test_four_level_class_field_chain() {
    let mut i = Incremental::new();

    // Chain: main -> baz -> bar -> foo
    // Each level wraps the previous level's class
    i.set("foo", "class A:\n    val: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set("baz", "from bar import B\nclass C:\n    b: B");
    i.set(
        "main",
        "from baz import C\ndef get_val(c: C) -> int:\n    return c.b.a.val",
    );
    i.check(&["main"], &["main", "foo", "bar", "baz"]);

    i.set("foo", "class A:\n    val: str = 'hello'");
    // `bar` declares a field of type `A` without reading `val`, so it is not rechecked.
    i.check_matches_clean(&["main"], &["main", "foo"]);
}

/// Test that star import invalidates when the export *set* changes.
///
/// `from X import *` records a wildcard dependency on X's export set:
/// adding or removing an exported name changes the set and invalidates
/// the importer. Changes to the *content* of an export that's already
/// in the set don't invalidate via the wildcard — only via that
/// specific name's dep, if it's used.
#[test]
fn test_star_import_invalidates_on_export_set_change() {
    let mut i = Incremental::new();

    // bar uses star import from foo
    i.set("foo", "class A:\n    x: int = 1\nclass B:\n    y: int = 2");
    i.set("bar", "from foo import *\nclass C:\n    a: A");
    i.set(
        "main",
        "from bar import C\ndef f(c: C) -> int:\n    return c.a.x",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change B's *content* (not its existence). bar doesn't actually
    // use B, so the wildcard dep doesn't fire and bar is not
    // recomputed; main is also unaffected.
    i.set(
        "foo",
        "class A:\n    x: int = 1\nclass B:\n    y: str = 'changed'",
    );
    i.check_ignoring_expectations(&["main"], &["foo"]);

    // Add a new export to foo. The wildcard *set* changes, which
    // invalidates bar (via its wildcard dep). bar's exports
    // consequently gain `D`, which is an existence change in bar's
    // exports — so main is also invalidated.
    i.set(
        "foo",
        "class A:\n    x: int = 1\nclass B:\n    y: str = 'changed'\nclass D:\n    z: int = 3",
    );
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);
}

/// Test that enum member changes propagate correctly.
#[test]
fn test_enum_member_change_propagates() {
    let mut i = Incremental::new();

    i.set(
        "foo",
        "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2",
    );
    i.set(
        "bar",
        "from foo import Color\nclass Palette:\n    primary: Color",
    );
    i.set(
        "main",
        "from bar import Palette\nfrom foo import Color\ndef f(p: Palette):\n    if p.primary == Color.RED: pass",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add a new enum member - main should be recomputed
    i.set(
        "foo",
        "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
    );
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test fine-grained tracking when importing a class: changing an unrelated
/// function should NOT trigger recomputation.
#[test]
fn test_class_import_unrelated_function_change_no_recompute() {
    let mut i = Incremental::new();

    // foo has class A and unrelated function f; main imports only A
    i.set("foo", "class A:\n    x: int = 1\ndef f() -> int: return 42");
    i.set(
        "main",
        "from foo import A\ndef use_a(a: A) -> int: return a.x",
    );
    i.check(&["main"], &["main", "foo"]);

    // Change only f - main should NOT be recomputed since it doesn't import f
    i.set(
        "foo",
        "class A:\n    x: int = 1\ndef f() -> str: return 'changed'",
    );
    i.check(&["main"], &["foo"]);
}

#[test]
fn test_mixed_import_failed_export_invalidated() {
    let mut i = Incremental::new();

    // Initial state: foo exports x but not y
    i.set("foo", "x = 1");
    i.set(
        "main",
        "from foo import x, y  # E: Could not import `y` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error before the fix
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.ordinary.is_empty(),
        "Expected error before foo exports y"
    );

    i.set("foo", "x = 1\ny = 2");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);
    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.ordinary.is_empty(),
        "Expected error after foo exports y"
    );
}

/// Test that disjoint-base changes propagate to downstream inheritance.
///
/// When a base class gains or loses disjoint-base status, modules that
/// inherit through it alongside another disjoint base should gain or clear
/// an `incompatible disjoint bases` diagnostic.
#[test]
fn test_disjoint_base_change_propagates_to_inheritance() {
    let mut i = Incremental::new();

    // foo.Left is NOT a disjoint base initially - no error for class Combined(Left, Right)
    i.set("foo", "class Left: pass");
    i.set(
        "bar",
        "from typing_extensions import disjoint_base\n@disjoint_base\nclass Right: pass",
    );
    i.set(
        "main",
        "from foo import Left\nfrom bar import Right\nclass Combined(Left, Right): pass",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Make Left disjoint - main should be recomputed and gain incompatible-disjoint-bases error
    i.set(
        "foo",
        "from typing_extensions import disjoint_base\n@disjoint_base\nclass Left: pass",
    );
    i.set(
        "main",
        "from foo import Left\nfrom bar import Right\nclass Combined(Left, Right): pass # E: incompatible disjoint bases",
    );
    i.check(&["main"], &["foo", "main"]);
}

/// Test that disjoint-base changes propagate to downstream narrowing.
///
/// When a base class gains disjoint-base status, modules narrowing against
/// it via `isinstance` should be recomputed: an intersection that was
/// previously inhabited should narrow to `Never`.
#[test]
fn test_disjoint_base_change_propagates_to_narrowing() {
    let mut i = Incremental::new();

    // foo.Base is NOT a disjoint base initially; the intersection with Other can be non-empty,
    // so `assert_never` should error.
    i.set("foo", "class Base: pass");
    i.set(
        "bar",
        "from typing_extensions import disjoint_base, assert_never\n@disjoint_base\nclass Other: pass",
    );
    i.set(
        "main",
        r#"
from typing_extensions import assert_never
from foo import Base
from bar import Other
def f(x: Base) -> None:
    if isinstance(x, Other):
        assert_never(x) # E: not assignable to parameter `arg` with type `Never`
"#,
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Make Base a disjoint base - the intersection is now Never, so assert_never should accept.
    i.set(
        "foo",
        "from typing_extensions import disjoint_base\n@disjoint_base\nclass Base: pass",
    );
    i.set(
        "main",
        r#"
from typing_extensions import assert_never
from foo import Base
from bar import Other
def f(x: Base) -> None:
    if isinstance(x, Other):
        assert_never(x)
"#,
    );
    i.check(&["main"], &["foo", "main"]);
}

/// Test that abstract class check changes propagate correctly (KeyAbstractClassCheck).
///
/// When a class becomes concrete (by implementing abstract methods), modules
/// that try to instantiate it should be recomputed and errors should clear.
#[test]
fn test_abstract_class_check_change_propagates() {
    let mut i = Incremental::new();

    // foo.Base is abstract (has abstract method), bar.Impl does NOT implement it initially
    i.set(
        "foo",
        "from abc import ABC, abstractmethod\nclass Base(ABC):\n    @abstractmethod\n    def m(self) -> int: ...",
    );
    i.set("bar", "from foo import Base\nclass Impl(Base): pass");
    i.set(
        "main",
        "from bar import Impl\nx = Impl() # E: Cannot instantiate `Impl` because the following members are abstract: `m`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Implement the abstract method - main should be recomputed and error should go away
    i.set(
        "bar",
        "from foo import Base\nclass Impl(Base):\n    def m(self) -> int: return 1",
    );
    i.set("main", "from bar import Impl\nx = Impl()");
    i.check(&["main"], &["bar", "main"]);
}

/// Test that class metadata changes propagate correctly (KeyClassMetadata).
///
/// When a class becomes a Protocol or stops being one, modules using it
/// should be recomputed.
#[test]
fn test_class_metadata_protocol_change_propagates() {
    let mut i = Incremental::new();

    // foo.P is NOT a Protocol initially - main uses it as a regular class
    i.set("foo", "class P:\n    def m(self) -> int: return 1");
    i.set("bar", "class Q:\n    def m(self) -> int: return 2");
    i.set(
        "main",
        "from foo import P\nfrom bar import Q\nx: P = Q() # E: `Q` is not assignable to `P`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Make P a Protocol - main should be recomputed and Q should be assignable
    i.set(
        "foo",
        "from typing import Protocol\nclass P(Protocol):\n    def m(self) -> int: ...",
    );
    i.set("main", "from foo import P\nfrom bar import Q\nx: P = Q()");
    i.check(&["main"], &["foo", "main"]);
}

/// Test that type parameter variance changes propagate correctly (KeyVariance).
///
/// When a generic class's variance changes (e.g., from invariant to covariant),
/// modules using that class should be recomputed.
#[test]
fn test_variance_change_propagates() {
    let mut i = Incremental::new();

    // foo.Container is invariant (default) - main has an error trying to assign Container[Derived] to Container[Base]
    i.set(
        "foo",
        "from typing import Generic, TypeVar\nT = TypeVar('T')\nclass Container(Generic[T]):\n    def get(self) -> T: ...",
    );
    i.set("bar", "class Base: pass\nclass Derived(Base): pass");
    i.set(
        "main",
        "from foo import Container\nfrom bar import Base, Derived\ndef f(c: Container[Derived]) -> Container[Base]:\n    return c # E: Returned type `Container[Derived]` is not assignable to declared return type `Container[Base]`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Make Container covariant - main should be recomputed and error should go away
    i.set(
        "foo",
        "from typing import Generic, TypeVar\nT = TypeVar('T', covariant=True)\nclass Container(Generic[T]):\n    def get(self) -> T: ...",
    );
    i.set(
        "main",
        "from foo import Container\nfrom bar import Base, Derived\ndef f(c: Container[Derived]) -> Container[Base]:\n    return c",
    );
    i.check(&["main"], &["foo", "main"]);
}

/// Test that adding a base class to a derived class propagates (KeyClassBaseType).
///
/// When a class gains a base class, modules that use the derived class
/// should be recomputed and see the inherited methods.
#[test]
fn test_adding_base_class_propagates() {
    let mut i = Incremental::new();

    // foo.Base has method m, bar.Derived does NOT extend Base initially
    i.set("foo", "class Base:\n    def m(self) -> int: return 1");
    i.set("bar", "class Derived: pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.m() # E: Object of class `Derived` has no attribute `m`",
    );
    i.check(&["main"], &["main", "bar"]);

    // Change Derived to extend Base - main should be recomputed and error should go away
    i.set("bar", "from foo import Base\nclass Derived(Base): pass");
    i.set(
        "main",
        "from bar import Derived\ndef f(d: Derived) -> int:\n    return d.m()",
    );
    i.check(&["main"], &["main", "foo", "bar"]);
}

/// Test that adding a dataclass field propagates (KeyClassSynthesizedFields).
///
/// When a dataclass gains a new field, its __init__ parameters change and
/// modules using the class should be recomputed.
#[test]
fn test_adding_dataclass_field_propagates() {
    let mut i = Incremental::new();

    // foo.Data is a dataclass with field x: int - main tries to use missing field y
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: int",
    );
    i.set("bar", "from foo import Data\nclass Wrapper:\n    d: Data");
    i.set(
        "main",
        "from foo import Data\na = Data(x=1, y=2) # E: Unexpected keyword argument `y`",
    );
    i.check(&["main"], &["main", "foo"]);

    // Add field y to the dataclass - main should be recomputed and error should go away
    i.set(
        "foo",
        "from dataclasses import dataclass\n@dataclass\nclass Data:\n    x: int\n    y: int",
    );
    i.set("main", "from foo import Data\na = Data(x=1, y=2)");
    i.check(&["main"], &["foo", "main"]);
}

/// Test that adding a field to a class propagates (KeyClassField).
///
/// When a class gains a new field, modules accessing that field should be
/// recomputed and no longer see an error.
#[test]
fn test_adding_class_field_propagates() {
    let mut i = Incremental::new();

    // foo.A has only field x - main tries to access missing field y
    i.set("foo", "class A:\n    x: int = 1");
    i.set("bar", "from foo import A\nclass B:\n    a: A");
    i.set(
        "main",
        "from bar import B\ndef f(b: B):\n    b.a.y # E: Object of class `A` has no attribute `y`",
    );
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add field y to A - main should be recomputed and error should go away
    i.set("foo", "class A:\n    x: int = 1\n    y: int = 2");
    i.set("main", "from bar import B\ndef f(b: B):\n    b.a.y");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
}

/// Test that making a class generic propagates (KeyTParams).
///
/// When a class gains type parameters (becomes generic), modules using
/// the class with type arguments should be recomputed.
#[test]
fn test_making_class_generic_propagates() {
    let mut i = Incremental::new();

    // foo.Container is not generic initially - main tries to use it with type args
    i.set("foo", "class Container:\n    items: list[int]");
    i.set(
        "bar",
        "from foo import Container\nclass Wrapper:\n    c: Container",
    );
    i.set(
        "main",
        "from foo import Container\nx: Container[int] # E: Expected 0 type arguments for `Container`, got 1",
    );
    i.check(&["main"], &["main", "foo"]);

    // Make Container generic - main should be recomputed and error should go away
    i.set(
        "foo",
        "from typing import Generic, TypeVar\nT = TypeVar('T')\nclass Container(Generic[T]):\n    items: list[int]",
    );
    i.set("main", "from foo import Container\nx: Container[int]");
    i.check(&["main"], &["foo", "main"]);
}

#[test]
fn test_adding_mro_attr_propagates() {
    let mut i = Incremental::new();
    i.set("foo", "class Foo: pass");
    i.set("bar", "from foo import Foo\nclass Bar(Foo): pass");
    i.set("main", "from bar import Bar\nBar().foo");
    i.check_ignoring_expectations(&["main"], &["main", "foo", "bar"]);
    i.set("foo", "class Foo: foo: int = 1");
    i.check(&["main"], &["main", "foo", "bar"]);
}

/// Test that adding a name listed in __all__ clears the error.
///
/// When __all__ lists a name that doesn't exist, there should be an error.
/// After adding the missing name, the error should disappear.
#[test]
fn test_dunder_all_missing_name_error_clears() {
    let mut i = Incremental::new();

    // foo has __all__ = ["test"] but doesn't define test - should error
    i.set(
        "foo",
        "__all__ = [\"test\"] # E: Name `test` is listed in `__all__` but is not defined in the module",
    );
    i.check(&["foo"], &["foo"]);

    let foo_handle = i.handle("foo");

    // Verify there's an error with the expected message
    let errors = i
        .state
        .transaction()
        .get_errors([&foo_handle])
        .collect_errors();
    assert!(
        !errors.ordinary.is_empty(),
        "Expected error when __all__ lists undefined name"
    );
    assert!(
        errors.ordinary.iter().any(|e| e
            .msg()
            .contains("Name `test` is listed in `__all__` but is not defined in the module")),
        "Expected error message about missing __all__ name, but got: {:?}",
        errors.ordinary.iter().map(|e| e.msg()).collect::<Vec<_>>()
    );

    // Add the missing name - error should disappear
    i.set("foo", "__all__ = [\"test\"]\ntest = 1");
    i.check_ignoring_expectations(&["foo"], &["foo"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&foo_handle])
        .collect_errors();
    assert!(
        errors_after.ordinary.is_empty(),
        "Expected no errors after defining the missing name, but got: {:?}",
        errors_after.ordinary
    );
}

/// Test that modifying __all__ to include a name clears errors in importing files.
///
/// When foo has `x` and `y` but `__all__ = ["x"]`, a star import only brings in `x`.
/// A file using `y` after `from foo import *` will error until `__all__` is updated.
/// This tests export information affecting errors in another file.
///
/// Test that errors are properly cleared when __all__ changes.
/// When a name is added to __all__, star importers should see the change.
#[test]
fn test_dunder_all_star_import_error_clears() {
    let mut i = Incremental::new();

    // foo has x and y, but __all__ only exports x
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\"]");
    // main does star import and tries to use y - should error
    i.set(
        "main",
        "from foo import *\nz = y # E: Could not find name `y`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error with the expected message
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.ordinary.is_empty(),
        "Expected error when using name not in __all__"
    );
    assert!(
        errors
            .ordinary
            .iter()
            .any(|e| e.msg().contains("Could not find name `y`")),
        "Expected error about missing name y, but got: {:?}",
        errors.ordinary.iter().map(|e| e.msg()).collect::<Vec<_>>()
    );

    // Update __all__ to include y - error should disappear
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\", \"y\"]");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.ordinary.is_empty(),
        "Expected no errors after adding y to __all__, but got: {:?}",
        errors_after.ordinary
    );
}

/// Test that adding a new definition invalidates importers of that name.
/// When a new name is added to a module, modules that try to import it
/// (and previously got an error) should be recomputed.
#[test]
fn test_name_existence_change_invalidates_importer() {
    let mut i = Incremental::new();

    // foo only has x
    i.set("foo", "x = 1");
    // main tries to import y which doesn't exist - should error
    i.set(
        "main",
        "from foo import y # E: Could not import `y` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.ordinary.is_empty(),
        "Expected error when importing non-existent name"
    );

    // Add y to foo - error should disappear
    i.set("foo", "x = 1\ny = 2");
    // main should be recomputed because y now exists
    i.check_ignoring_expectations(&["foo"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.ordinary.is_empty(),
        "Expected no errors after adding y to foo, but got: {:?}",
        errors_after.ordinary
    );
}

/// Test that adding a name NOT in __all__ still invalidates importers of that name.
/// __all__ controls what `from M import *` gets, but explicit imports like
/// `from M import name` can access any name in the module. When that name is
/// added or removed, importers should be invalidated.
#[test]
fn test_name_not_in_dunder_all_invalidates_importer() {
    let mut i = Incremental::new();

    // foo has __all__ = ["x"] but only x exists
    i.set("foo", "__all__ = ['x']\nx = 1");
    // main tries to import y which doesn't exist - should error
    i.set(
        "main",
        "from foo import y # E: Could not import `y` from `foo`",
    );
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's an error
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.ordinary.is_empty(),
        "Expected error when importing non-existent name"
    );

    // Add y to foo, but NOT to __all__. The import should still work.
    i.set("foo", "__all__ = ['x']\nx = 1\ny = 2");
    // main should be recomputed because y now exists
    i.check_ignoring_expectations(&["foo"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.ordinary.is_empty(),
        "Expected no errors after adding y to foo (even though y is not in __all__), but got: {:?}",
        errors_after.ordinary
    );
}

/// Test that wildcard re-exports properly propagate invalidation.
///
/// When module A re-exports module B's __all__ (e.g., via `from B import *`),
/// and B's __all__ changes, star importers of A should be invalidated.
#[test]
fn test_wildcard_reexport_invalidation() {
    let mut i = Incremental::new();

    // foo exports x via __all__
    i.set("foo", "x = 1\n__all__ = [\"x\"]");
    // bar re-exports foo's __all__ via from foo import *
    i.set("bar", "from foo import *");
    // main uses star import from bar
    i.set("main", "from bar import *\nz = x");
    i.check(&["main"], &["main", "foo", "bar"]);

    // Add y to foo's __all__ - main should be invalidated because
    // bar's effective wildcard set changed (even though bar's __all__ entries didn't)
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\", \"y\"]");
    // bar is invalidated because it depends on foo with wildcard = true
    // main is invalidated because bar's effective wildcard set changed
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);
}

/// Test that type changes propagate through transitive star import chains.
///
/// When module A re-exports module B's exports (via `from B import *`),
/// and a type in B changes, consumers of A that use that type should be invalidated.
#[test]
fn test_wildcard_reexport_type_change_invalidation() {
    let mut i = Incremental::new();

    // foo exports x via __all__
    i.set("foo", "x: int = 1\n__all__ = [\"x\"]");
    // bar re-exports foo's __all__ via from foo import * (pure pass-through)
    i.set("bar", "from foo import *");
    // main uses star import from bar and uses x
    i.set("main", "from bar import *\nz: int = x");
    i.check(&["main"], &["main", "foo", "bar"]);

    // Change x's type from int to str - main should be invalidated because
    // it uses x and the type changed, even though bar is just a pass-through.
    i.set("foo", "x: str = 'hello'\n__all__ = [\"x\"]");
    // foo is invalidated because its source changed
    // bar is invalidated because it depends on foo with wildcard = true
    // main is invalidated because x's type changed and main uses x
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);
}

/// Test that star imports invalidate on type changes to unused variable names.
#[test]
fn test_star_import_unused_variable_type_change() {
    let mut i = Incremental::new();

    // foo exports x and y
    i.set("foo", "x: int = 1\ny: int = 2\n__all__ = ['x', 'y']");
    // bar does star import but only uses x
    i.set("bar", "from foo import *\nz = x + 1");
    i.check(&["bar"], &["bar", "foo"]);

    // Change y's type - bar IS invalidated even though it doesn't use y
    i.set(
        "foo",
        "x: int = 1\ny: str = 'changed'\n__all__ = ['x', 'y']",
    );
    i.check_ignoring_expectations(&["bar"], &["foo", "bar"]);

    // If we change x's type, bar SHOULD be invalidated
    i.set(
        "foo",
        "x: str = 'now a string'\ny: str = 'changed'\n__all__ = ['x', 'y']",
    );
    i.check_ignoring_expectations(&["bar"], &["foo", "bar"]);
}

/// Test that star imports don't surface missing-name errors when __all__ is invalid.
///
/// When `__all__` lists a name that doesn't exist, the error should be reported
/// on the `__all__` definition, not on star-importers.
#[test]
fn test_dunder_all_star_import_missing_definition_no_import_error() {
    let mut i = Incremental::new();

    // foo has __all__ = ["x", "y"] but only defines x - y is missing
    i.set(
        "foo",
        "x = 1\n__all__ = [\"x\", \"y\"] # E: Name `y` is listed in `__all__` but is not defined",
    );
    // main does star import and tries to use y - should not error here
    i.set("main", "from foo import *\nz = y");
    i.check(&["main", "foo"], &["main", "foo"]);

    let main_handle = i.handle("main");

    // Verify there's no error in the importing module
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors.ordinary.is_empty(),
        "Expected no errors in star importer, but got: {:?}",
        errors.ordinary
    );

    // Add the missing definition - still no errors
    i.set("foo", "x = 1\ny = 2\n__all__ = [\"x\", \"y\"]");
    i.check_ignoring_expectations(&["main"], &["foo", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.ordinary.is_empty(),
        "Expected no errors after adding y definition, but got: {:?}",
        errors_after.ordinary
    );
}

/// Test that transitive star import errors clear when a missing name is added.
///
/// Similar to test_dunder_all_star_import_missing_definition_error_clears but:
/// - No __all__ is used (relies on default public exports)
/// - Uses transitive star imports (foo -> bar -> main)
#[test]
fn test_transitive_star_import_missing_name_error_clears() {
    let mut i = Incremental::new();

    // foo initially empty
    i.set("foo", "");
    // bar re-exports from foo via star import
    i.set("bar", "from foo import *");
    // main does star import from bar and tries to use x - should fail
    i.set("main", "from bar import *\nz = x");
    i.check_ignoring_expectations(&["main", "foo", "bar"], &["main", "foo", "bar"]);

    let main_handle = i.handle("main");

    // Verify there's an error
    let errors = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        !errors.ordinary.is_empty(),
        "Expected error when using undefined name via transitive star import"
    );

    // Add x to foo - error should disappear in main
    i.set("foo", "x = 1");
    i.check_ignoring_expectations(&["main"], &["foo", "bar", "main"]);

    let errors_after = i
        .state
        .transaction()
        .get_errors([&main_handle])
        .collect_errors();
    assert!(
        errors_after.ordinary.is_empty(),
        "Expected no errors after adding x to foo, but got: {:?}",
        errors_after.ordinary
    );
}

#[test]
fn test_class_index_change_only_invalidates() {
    let mut i = Incremental::new();

    i.set("foo", "class A:\n    attr: int = 1");
    i.set("bar", "from foo import A\nx = A.attr");
    i.check(&["bar"], &["foo", "bar"]);

    // Now add class B before A, shifting A to index 1
    i.set(
        "foo",
        "class B:\n    other: str = 'hello'\nclass A:\n    attr: int = 1",
    );
    i.check(&["foo"], &["foo", "bar"]);
}

#[test]
fn test_class_index_and_key_change_invalidates_dependents() {
    let mut i = Incremental::new();

    i.set("foo", "class A:\n    attr: int = 1");
    i.set("bar", "from foo import A\nx = A.attr");
    i.check(&["bar"], &["foo", "bar"]);

    i.set(
        "foo",
        "class B:\n    other: str = 'hello'\nclass A:\n    attr: bool = True",
    );
    i.check(&["foo"], &["foo", "bar"]);
}

/// Test that when a function's deprecation metadata changes, dependents are invalidated.
///
/// This tests the `get_deprecated` metadata dependency. When a function gains or loses
/// the `@deprecated` decorator, modules that import it should be recomputed so they
/// can show or hide deprecation warnings.
#[test]
fn test_deprecation_metadata_change_invalidates() {
    let mut i = Incremental::new();

    // foo.f is NOT deprecated initially
    i.set("foo", "def f() -> int: return 1");
    // # E: comments are present for the second check when deprecation warnings appear
    i.set("main", "from foo import f # E:\nx = f() # E:");
    i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);

    // Add @deprecated decorator - main should be recomputed to show deprecation warning
    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use g instead')\ndef f() -> int: return 1",
    );
    i.check(&["foo", "main"], &["foo", "main"]);
}

/// Test that when a class's deprecation metadata changes, dependents are invalidated.
///
/// Similar to test_deprecation_metadata_change_invalidates but for classes.
#[test]
fn test_class_deprecation_metadata_change_invalidates() {
    let mut i = Incremental::new();

    // foo.C is NOT deprecated initially
    i.set("foo", "class C:\n    x: int = 1");
    // The second check expects deprecation warnings at the import and constructor call.
    i.set(
        "main",
        "from foo import C # E: `C` is deprecated\nc = C() # E: `foo.C` is deprecated",
    );
    i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);

    // Add @deprecated decorator to class - main should be recomputed
    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use D instead')\nclass C:\n    x: int = 1",
    );
    i.check(&["foo", "main"], &["foo", "main"]);
}

/// A class reference depends on deprecation metadata even when no constructor is called.
#[test]
fn test_class_reference_deprecation_metadata_change_invalidates() {
    let mut i = Incremental::new();

    i.set("foo", "class C: pass");
    i.set("main", "import foo\ncls = foo.C # E: use D instead");
    i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);

    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use D instead')\nclass C: pass",
    );
    i.check(&["foo", "main"], &["foo", "main"]);
}

/// Class aliases carry deprecation changes through their exported type.
#[test]
fn test_class_alias_deprecation_change_invalidates() {
    let mut i = Incremental::new();
    i.set("foo", "class C: pass\nAlias = C");
    i.set("main", "import foo\ncls = foo.Alias # E: use D instead");
    i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);

    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use D instead')\nclass C: pass\nAlias = C # E: use D instead",
    );
    i.check(&["foo", "main"], &["foo", "main"]);

    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use E instead')\nclass C: pass\nAlias = C # E: use E instead",
    );
    let result = i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);
    let errors = result.errors.collect_display_errors();
    assert_eq!(errors.len(), 2);
    assert!(
        errors
            .iter()
            .all(|error| error.msg().contains("use E instead"))
    );

    i.set("foo", "class C: pass\nAlias = C");
    let result = i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);
    assert!(result.errors.collect_display_errors().is_empty());
}

/// Specialized class aliases carry deprecation changes through their exported type.
#[test]
fn test_specialized_class_alias_deprecation_change_invalidates() {
    let mut i = Incremental::new();
    i.set("foo", "class C[T]: pass\nAlias = C[int]");
    i.set("main", "import foo\nfoo.Alias() # E: use D instead");
    i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);

    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('use D instead')\nclass C[T]: pass\nAlias = C[int] # E: use D instead",
    );
    i.check(&["foo", "main"], &["foo", "main"]);

    i.set("foo", "class C[T]: pass\nAlias = C[int]");
    let result = i.check_ignoring_expectations(&["foo", "main"], &["foo", "main"]);
    assert!(result.errors.collect_display_errors().is_empty());
}

/// Reading a class object does not depend on the types of its methods.
#[test]
fn test_class_reference_method_change_does_not_invalidate() {
    let mut i = Incremental::new();
    i.set("foo", "class C:\n    def method(self) -> int: return 1");
    i.set("main", "import foo\ncls = foo.C");
    i.check(&["foo", "main"], &["foo", "main"]);

    i.set("foo", "class C:\n    def method(self) -> str: return 'new'");
    i.check(&["foo", "main"], &["foo"]);
}

/// Test that when an export changes from direct definition to re-export, dependents are invalidated.
///
/// This tests the `reexport_source` metadata dependency. When an export changes from being
/// defined in this module to being re-exported from another module, dependents should
/// be recomputed.
#[test]
fn test_reexport_status_change_invalidates() {
    let mut i = Incremental::new();

    // foo.x is defined directly in foo
    i.set("foo", "x: int = 1");
    i.set("bar", "y: int = 2");
    i.set("main", "from foo import x\nz = x + 1");
    i.check(&["foo", "main"], &["foo", "main"]);

    // Change foo.x to be a re-export from bar
    // Since the type doesn't change (still int), main should still be recomputed
    // so that go-to-definition points to bar.y instead of the old foo.x
    i.set("foo", "from bar import y as x");
    let res = i.unchecked(&["foo", "main"]);
    assert!(res.changed.contains(&"foo".to_owned()));
    assert!(res.changed.contains(&"bar".to_owned()));
    assert!(res.changed.contains(&"main".to_owned()));
}

/// Test that when a re-export's source changes, dependents are invalidated.
///
/// When foo re-exports from bar, changing which module foo re-exports from
/// should invalidate dependents.
#[test]
fn test_reexport_source_change_invalidates() {
    let mut i = Incremental::new();

    // foo re-exports x from bar
    i.set("bar", "x: int = 1");
    i.set("baz", "x: str = 'hello'");
    i.set("foo", "from bar import x");
    i.set("main", "from foo import x\ny = x + 1 # E:");
    // Initial check - no error yet (x is int), but # E: is present for later
    i.check_ignoring_expectations(&["foo", "main"], &["bar", "foo", "main"]);

    // Change foo to re-export from baz instead - main should be recomputed (type changes)
    // Now x is str, so x + 1 is a type error
    i.set("foo", "from baz import x");
    i.check(&["foo", "main"], &["baz", "foo", "main"]);
}

/// Test that when a special export changes, dependents are invalidated.
///
/// This tests the `is_special_export` metadata dependency. When a name gains or loses
/// special export status (like TypeVar, cast, etc.), modules using it should be recomputed.
/// Note: This is a somewhat contrived test since special exports are typically in stdlib,
/// but it validates the dependency tracking.
#[test]
fn test_special_export_usage_change_invalidates() {
    let mut i = Incremental::new();

    // main uses cast from typing
    i.set(
        "main",
        "from typing import cast\nx: int = cast(int, 'hello')",
    );
    i.check(&["main"], &["main"]);

    // Change to use a different cast that's not special - main should be recomputed
    i.set("foo", "def cast(ty, val): return val");
    i.set("main", "from foo import cast\nx: int = cast(int, 'hello')");
    i.check(&["foo", "main"], &["foo", "main"]);
}

/// Test that when a docstring changes, hover info is updated correctly.
///
/// This tests the `docstring_range` metadata dependency. When a function's docstring
/// changes, the hover information should reflect the new docstring.
/// Note: This test primarily validates that docstring changes trigger recomputation
/// for accurate hover info, even if the type signature doesn't change.
#[test]
fn test_docstring_change_invalidates() {
    let mut i = Incremental::new();

    // foo.f has a docstring
    i.set(
        "foo",
        "def f() -> int:\n    \"\"\"Original docstring.\"\"\"\n    return 1",
    );
    i.set("main", "from foo import f\nx = f()");
    i.check(&["foo", "main"], &["foo", "main"]);

    // Change the docstring - foo should be recomputed (main is not recomputed since
    // the type signature didn't change)
    i.set(
        "foo",
        "def f() -> int:\n    \"\"\"Updated docstring with new info.\"\"\"\n    return 1",
    );
    i.check(&["foo", "main"], &["foo"]);
}

/// Test that when a class docstring changes, it is properly recomputed.
#[test]
fn test_class_docstring_change_invalidates() {
    let mut i = Incremental::new();

    // foo.C has a docstring
    i.set(
        "foo",
        "class C:\n    \"\"\"Original class docstring.\"\"\"\n    x: int = 1",
    );
    i.set("main", "from foo import C\nc = C()");
    i.check(&["foo", "main"], &["foo", "main"]);

    // Change the class docstring - foo should be recomputed (main is not recomputed
    // since the type signature didn't change)
    i.set(
        "foo",
        "class C:\n    \"\"\"Updated class docstring.\"\"\"\n    x: int = 1",
    );
    i.check(&["foo", "main"], &["foo"]);
}

/// Test that changing the `Doc` text of a parameter recomputes only the module that defines it.
#[test]
fn test_annotated_doc_change_invalidates() {
    let mut i = Incremental::new();
    let foo = |doc: &str| {
        format!(
            "from typing import Annotated\n\
             class Doc:\n    def __init__(self, documentation: str, /) -> None: ...\n\
             def f(x: Annotated[int, Doc(\"{doc}\")]) -> int:\n    return x\n\
             class C:\n    def m(self, x: Annotated[int, Doc(\"{doc}\")]) -> None: ...\n"
        )
    };

    i.set("foo", &foo("Original"));
    i.set("main", "from foo import C, f\nx = f(1)\nC().m(1)");
    i.check(&["foo", "main"], &["foo", "main"]);

    // Change the `Doc` text, which also moves the ranges after it - foo should be recomputed
    // (main is not recomputed since the type signatures didn't change)
    i.set("foo", &foo("Updated with more words"));
    i.check(&["foo", "main"], &["foo"]);
}

/// Test that when __all__ contains Module entries (like `__all__.extend(other.__all__)`),
/// changing those entries triggers wildcard fallback and invalidates star importers.
///
/// This tests the `wildcard_fallback` behavior in `changed_name_existence`.
/// When __all__ contains Module entries that change, we can't compute the exact
/// set of names that changed without a lookup, so we emit a coarse Wildcard signal.
#[test]
fn test_dunder_all_module_entry_change_invalidates() {
    let mut i = Incremental::new();

    // bar exports x
    i.set("bar", "x = 1\n__all__ = ['x']");
    // baz exports y
    i.set("baz", "y = 2\n__all__ = ['y']");
    // foo imports bar and re-exports bar's __all__
    i.set(
        "foo",
        "import bar\nfrom bar import *\n__all__ = []\n__all__.extend(bar.__all__)",
    );
    // main does star import from foo and uses x
    i.set("main", "from foo import *\nz = x");
    i.check(
        &["main", "foo", "bar", "baz"],
        &["main", "foo", "bar", "baz"],
    );

    // Now change foo to re-export baz's __all__ instead of bar's
    // This changes a Module entry in __all__, triggering wildcard fallback
    i.set(
        "foo",
        "import baz\nfrom baz import *\n__all__ = []\n__all__.extend(baz.__all__)",
    );
    // main should be recomputed because the wildcard set changed
    i.check_ignoring_expectations(&["foo"], &["foo", "main"]);
}

#[test]
fn test_type_shape_dsl_parameter_domain_edit_revalidates_indexed_locals() {
    let mut i = Incremental::with_files(vec![
        "consumer".to_owned(),
        "helpers".to_owned(),
        "main".to_owned(),
        "shape_extensions".to_owned(),
    ]);
    i.set(
        "shape_extensions",
        r#"
class Int: pass
class IntTuple: pass
class IntTuples: pass
def type_shape_dsl_function[F](fn: F) -> F: return fn
"#,
    );
    i.set(
        "helpers",
        r#"
from shape_extensions import IntTuple, IntTuples, type_shape_dsl_function

@type_shape_dsl_function
def first(shapes: IntTuples) -> IntTuple:
    selected = shapes[0]
    return selected
"#,
    );
    i.set(
        "main",
        r#"
from helpers import first
from shape_extensions import IntTuple

class ShapeBox[Shape: IntTuple]: ...
def result() -> ShapeBox[first(tuple[IntTuple[2, 3], IntTuple[4]])]: ...
"#,
    );
    i.set(
        "consumer",
        r#"
from main import ShapeBox, result
from shape_extensions import IntTuple

expected: ShapeBox[IntTuple[2, 3]] = result()
"#,
    );
    let initial = i.unchecked(&["consumer", "helpers"]);
    assert!(initial.errors.collect_display_errors().is_empty());

    i.set(
        "helpers",
        r#"
from shape_extensions import IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def first(shapes: IntTuple) -> IntTuple:
    selected = shapes[0]
    return selected  # E: local return domain must match  # E: Returned type `Int[int]` is not assignable
"#,
    );
    let changed = i.unchecked(&["consumer", "helpers"]);
    changed.check_recompute(&["consumer", "helpers", "main"]);
    changed.check_errors();
}

/// Whether an imported `Final` settles a test depends on whether a module on the way to it defines
/// or star-imports it under a test of the environment, which can change without changing its type.
#[test]
fn test_final_moved_under_environment_test_rechecks_importer() {
    let mut i = Incremental::new();
    let errors = |i: &mut Incremental| {
        i.unchecked(&["main"])
            .errors
            .collect_display_errors()
            .map(|error| error.msg().to_owned())
    };
    i.set("bar", "from typing import Final\nDEBUG: Final = False");
    i.set("foo", "from bar import DEBUG as DEBUG");
    i.set(
        "main",
        "from foo import DEBUG\nif DEBUG:\n    x = 1\nelse:\n    x = ''\nobserved: str = x",
    );
    assert_eq!(errors(&mut i), ["This code is unreachable"]);

    i.set(
        "bar",
        "import sys\nfrom typing import Final\nif sys.platform != 'win32':\n    DEBUG: Final = False",
    );
    let unsettled = errors(&mut i);
    assert_eq!(unsettled.len(), 1, "{unsettled:?}");
    assert!(
        unsettled[0].contains("not assignable to `str`"),
        "{unsettled:?}"
    );

    i.set("bar", "from typing import Final\nDEBUG: Final = False");
    i.set("foo", "from bar import *");
    assert_eq!(errors(&mut i), ["This code is unreachable"]);

    i.set(
        "foo",
        "import sys\nfrom bar import *\nif sys.platform == 'win32':\n    DEBUG = True",
    );
    let unsettled = errors(&mut i);
    assert_eq!(unsettled.len(), 1, "{unsettled:?}");
    assert!(
        unsettled[0].contains("not assignable to `str`"),
        "{unsettled:?}"
    );

    i.set("foo", "from bar import *");
    assert_eq!(errors(&mut i), ["This code is unreachable"]);

    i.set(
        "foo",
        "import sys\nif sys.platform != 'win32':\n    from bar import *",
    );
    let unsettled = errors(&mut i);
    assert_eq!(unsettled.len(), 1, "{unsettled:?}");
    assert!(
        unsettled[0].contains("not assignable to `str`"),
        "{unsettled:?}"
    );
}

/// A star import in a suite this configuration rules out brings in whatever the other module
/// defines, so defining the constant there must recheck the importer.
#[test]
fn test_constant_defined_behind_skipped_star_import_rechecks_importer() {
    let mut i = Incremental::new();
    let errors = |i: &mut Incremental| {
        i.unchecked(&["main"])
            .errors
            .collect_display_errors()
            .map(|error| error.msg().to_owned())
    };
    i.set("bar", "from typing import Final\nDEBUG: Final = False");
    i.set("baz", "");
    i.set(
        "foo",
        "import sys\nfrom bar import DEBUG as DEBUG\nif sys.platform == 'win32':\n    from baz import *",
    );
    i.set(
        "main",
        "from foo import DEBUG\nif DEBUG:\n    x = 1\nelse:\n    x = ''\nobserved: str = x",
    );
    assert_eq!(errors(&mut i), ["This code is unreachable"]);

    i.set("baz", "from typing import Final\nDEBUG: Final = True");
    let unsettled = errors(&mut i);
    assert_eq!(unsettled.len(), 1, "{unsettled:?}");
    assert!(
        unsettled[0].contains("not assignable to `str`"),
        "{unsettled:?}"
    );
}

/// Module `b` of the type-parameter tests. It uses `a.Foo` directly, and it exposes `Foo` and
/// copies of `Foo`'s type parameters to other modules: as a class object, in aliases, in the bounds
/// and defaults of its own classes, and in function signatures.
const TPARAMS_B: &str = r#"
from typing import Generic, TypeAlias, TypeVar, reveal_type
from a import Foo

S = TypeVar("S", default=Foo[int])
R = TypeVar("R", bound=Foo[int])

FooAlias = Foo
FooBool = Foo[bool]
FooExplicit: TypeAlias = Foo
type FooTypeAlias = Foo

class BDefault[T = Foo[int]]: ...
class BBound[T: Foo[int]]: ...
class LegacyDefault(Generic[S]): ...
class LegacyBound(Generic[R]): ...

def make() -> Foo: ...
def make_bool() -> Foo[bool]: ...
def take_int(x: Foo[int]) -> None: ...
def ident[X: Foo[int]](x: X) -> X: ...

x1: Foo[bool]
x2: Foo[str]
x3: Foo[bool, str]
reveal_type(Foo())

def direct(v: Foo, w: Foo[bool]) -> Foo[int]:
    reveal_type(v)
    return w
"#;

/// More modules that use `a.Foo`: `b_specialize` imports it from `a`, and each `c_*` module sees it
/// only through what `b` exports. Each imports as little as possible and avoids assignability
/// checks between specializations of `Foo` unless they are what it tests, so that it is rechecked
/// only when what it reads changes.
const TPARAMS_USERS: &[(&str, &str)] = &[
    // Only specializes `Foo`, which reads its type parameters and nothing else of it.
    (
        "b_specialize",
        r#"
from a import Foo
x1: Foo[bool]
x2: Foo[str]
x3: Foo[bool, str]
"#,
    ),
    // `Foo`'s class object, re-bound in `b`, only specialized.
    (
        "c_specialize",
        r#"
from b import FooAlias
y1: FooAlias[bool]
y2: FooAlias[str]
y3: FooAlias[bool, str]
"#,
    ),
    // `Foo`'s class object, re-bound in `b`.
    (
        "c_class",
        r#"
from typing import reveal_type
from b import FooAlias
y1: FooAlias[bool]
y2: FooAlias[str]
y3: FooAlias[bool, str]
reveal_type(FooAlias())
def f(v: FooAlias) -> FooAlias[int]:
    reveal_type(v)
    return v
def g(w: FooAlias[bool]) -> FooAlias[int]:
    return w
"#,
    ),
    // A class in `b` with a scoped type parameter bounded by `Foo[int]`.
    (
        "c_bound",
        r#"
from typing import reveal_type
from b import BBound
y: BBound[int]
def f(x: BBound) -> None:
    reveal_type(x)
"#,
    ),
    // A class in `b` with a legacy type variable bounded by `Foo[int]`.
    (
        "c_legacy_bound",
        r#"
from typing import reveal_type
from b import LegacyBound
y: LegacyBound[int]
def f(x: LegacyBound) -> None:
    reveal_type(x)
"#,
    ),
    // A class in `b` with a scoped type parameter defaulting to `Foo[int]`.
    (
        "c_default",
        r#"
from typing import reveal_type
from b import BDefault
def default_of[T](x: BDefault[T]) -> T: ...
def f(x: BDefault) -> None:
    reveal_type(x)
    reveal_type(default_of(x))
"#,
    ),
    // A class in `b` with a legacy type variable defaulting to `Foo[int]`.
    (
        "c_legacy_default",
        r#"
from typing import reveal_type
from b import LegacyDefault
def default_of[T](x: LegacyDefault[T]) -> T: ...
def f(x: LegacyDefault) -> None:
    reveal_type(x)
    reveal_type(default_of(x))
"#,
    ),
    // Uses the classes in `b` whose scoped type parameters embed `Foo[int]`, without reading
    // anything of `Foo` itself.
    (
        "c_never",
        r#"
from typing import Never
from b import BBound, BDefault
def f(x: BBound[Never], y: BDefault) -> None: ...
"#,
    ),
    // Uses the classes in `b` whose legacy type variables embed `Foo[int]`, without reading
    // anything of `Foo` itself.
    (
        "c_legacy_never",
        r#"
from typing import Never
from b import LegacyBound, LegacyDefault
def f(x: LegacyBound[Never], y: LegacyDefault) -> None: ...
"#,
    ),
    // Functions in `b` whose signatures mention `Foo`.
    (
        "c_function",
        r#"
from typing import reveal_type
from b import ident, make, make_bool
reveal_type(make())
reveal_type(make_bool())
y: int = make_bool()
ident(1)
"#,
    ),
    // Assignability between signatures in `b`, which depends on `Foo`'s variance.
    (
        "c_function_variance",
        r#"
from b import make_bool, take_int
take_int(make_bool())
"#,
    ),
    // An implicit type alias in `b`.
    (
        "c_alias",
        r#"
from typing import reveal_type
from b import FooBool
y: int = FooBool()
def f(x: FooBool) -> None:
    reveal_type(x)
"#,
    ),
    // An explicit `TypeAlias` in `b`.
    (
        "c_explicit_alias",
        r#"
from typing import reveal_type
from b import FooExplicit
y: FooExplicit[bool]
def f(x: FooExplicit) -> None:
    reveal_type(x)
"#,
    ),
    // A `type` statement in `b`.
    (
        "c_type_alias",
        r#"
from typing import reveal_type
from b import FooTypeAlias
y: int = FooTypeAlias()
def f(x: FooTypeAlias) -> None:
    reveal_type(x)
"#,
    ),
];

/// What a rebuild must reproduce for `module`: its errors, and the answers of all its exports,
/// re-exports included, in `Debug` form. The `Debug` form includes the type parameters that
/// classes and type arguments carry, so a stale copy of them is visible even where no error shows
/// it.
fn module_snapshot(i: &Incremental, errors: &[Error], module: &str) -> Vec<String> {
    let mut res = errors
        .iter()
        .filter(|e| e.path().as_path() == Path::new(module))
        .map(|e| {
            format!(
                "error {} {:?}: {}",
                e.display_range(),
                e.error_kind(),
                e.msg()
            )
        })
        .collect::<Vec<_>>();
    let transaction = i.state.transaction();
    let handle = i.handle(module);
    let solutions = transaction.get_solutions(&handle).unwrap();
    for name in transaction.get_exports(&handle).keys() {
        let Some(ty) = solutions.get_hashed_opt(Hashed::new(&KeyExport(name.clone()))) else {
            continue;
        };
        res.push(format!("export {name}: {ty:?}"));
        if let Type::ClassDef(cls) = ty
            && cls.module_path() == handle.path()
            && let Some(tparams) = solutions.get_hashed_opt(Hashed::new(&KeyTParams(cls.index())))
        {
            res.push(format!("export {name} type parameters: {tparams:?}"));
        }
    }
    res.sort();
    res
}

/// What `rebuild_against_clean` observed for one edit.
struct Rebuild {
    /// The modules that the recheck recomputed, sorted, once per recomputation.
    rechecked: Vec<String>,
    /// The `module_snapshot` of every module after the recheck.
    rebuilt: SmallMap<String, Vec<String>>,
    /// The `module_snapshot` of every module in a clean check of the edited sources.
    clean: SmallMap<String, Vec<String>>,
}

impl Rebuild {
    /// The modules whose errors or exported answers differ from the clean check, each with the
    /// lines that differ.
    fn stale(&self) -> SmallMap<&str, Vec<String>> {
        self.rebuilt
            .iter()
            .filter_map(|(m, got)| {
                let want = self.clean.get(m).unwrap();
                (got != want).then(|| {
                    let stale = got
                        .iter()
                        .filter(|x| !want.contains(x))
                        .map(|x| format!("    stale:   {x}"));
                    let missing = want
                        .iter()
                        .filter(|x| !got.contains(x))
                        .map(|x| format!("    missing: {x}"));
                    (m.as_str(), stale.chain(missing).collect())
                })
            })
            .collect()
    }
}

/// Check every module of `files`, apply `edits`, recheck every module with `threads` threads, and
/// check the edited sources from scratch. Imports that no file provides resolve against the
/// packages in `site_package_path`.
fn rebuild_against_clean(
    files: &[(&str, &str)],
    edits: &[(&str, &str)],
    threads: ThreadCount,
    site_package_path: &[PathBuf],
) -> Rebuild {
    let modules = files.map(|(m, _)| *m);
    let checked = |sources: &[(&str, &str)]| {
        let mut i = Incremental::with_site_packages(
            modules.map(|m| (*m).to_owned()),
            threads,
            site_package_path.to_vec(),
        );
        for (m, contents) in sources {
            i.set(m, contents);
        }
        let result = i.unchecked(&modules);
        (i, result)
    };
    let snapshots = |i: &Incremental, result: &IncrementalResult| {
        let errors = result.errors.collect_display_errors();
        modules
            .iter()
            .map(|m| ((*m).to_owned(), module_snapshot(i, &errors, m)))
            .collect::<SmallMap<_, _>>()
    };
    let (mut rebuilt, _) = checked(files);
    for (m, contents) in edits {
        rebuilt.set(m, contents);
    }
    let result = rebuilt.unchecked(&modules);
    let edited = files.map(|(m, contents)| {
        let contents = edits
            .iter()
            .find(|(edited, _)| edited == m)
            .map_or(*contents, |(_, contents)| *contents);
        (*m, contents)
    });
    let (clean, clean_result) = checked(&edited);
    Rebuild {
        rechecked: result.changed.clone(),
        rebuilt: snapshots(&rebuilt, &result),
        clean: snapshots(&clean, &clean_result),
    }
}

/// Check that rechecking after `edits` gives every module of `files` the errors and exported
/// answers that a clean check of the edited sources gives, at several thread counts, except for
/// exactly the modules in `stale`. A correct recheck leaves none, so a test lists modules here only
/// to document a known bug.
fn assert_rebuild_matches_clean(files: &[(&str, &str)], edits: &[(&str, &str)], stale: &[&str]) {
    let mut expected = stale.to_vec();
    expected.sort();
    for threads in [1, 3, 8] {
        let rebuild = rebuild_against_clean(
            files,
            edits,
            ThreadCount::NumThreads(NonZeroUsize::new(threads).unwrap()),
            &[],
        );
        let differences = rebuild.stale();
        let mut actual = differences.keys().copied().collect::<Vec<_>>();
        actual.sort();
        assert!(
            actual == expected,
            "stale modules after the recheck at {threads} threads: expected {expected:?}, got {actual:?} (rechecked: {:?})\n{}",
            rebuild.rechecked,
            differences
                .iter()
                .map(|(m, lines)| format!("`{m}` differs from a clean check\n{}", lines.join("\n")))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// The errors of `module` after rechecking `edits`, which must be the same at 1, 3 and 8 threads.
fn rebuilt_errors(files: &[(&str, &str)], edits: &[(&str, &str)], module: &str) -> Vec<String> {
    let [first, rest @ ..] = [1, 3, 8].map(|threads| {
        let rebuild = rebuild_against_clean(
            files,
            edits,
            ThreadCount::NumThreads(NonZeroUsize::new(threads).unwrap()),
            &[],
        );
        rebuild
            .rebuilt
            .get(module)
            .unwrap()
            .iter()
            .filter(|line| line.starts_with("error "))
            .cloned()
            .collect::<Vec<_>>()
    });
    for errors in rest {
        assert_eq!(errors, first);
    }
    first
}

/// The modules whose errors or exported answers differ from a clean check after a rebuild in
/// `assert_tparams_rebuild_matches_clean`, for each way it applies the edit. A correct rebuild
/// leaves none, so a test lists modules here only to document a known bug.
struct StaleModules {
    a_edited: &'static [&'static str],
    a_and_b_edited: &'static [&'static str],
}

impl StaleModules {
    const NONE: Self = Self {
        a_edited: &[],
        a_and_b_edited: &[],
    };
}

/// Check that changing `Foo` in module `a` from `before` to `after` and rechecking gives every
/// module the errors and exported answers that checking the edited sources from scratch gives,
/// except for exactly the modules in `stale`.
///
/// The edit is checked twice: on its own, and together with a comment-only edit to `b`. The
/// comment forces `b` to be rechecked, so a module that imports from `b` is stale only if what `b`
/// exports hides the change.
fn assert_tparams_rebuild_matches_clean(before: &str, after: &str, stale: StaleModules) {
    let files = ["a", "b"]
        .into_iter()
        .chain(TPARAMS_USERS.iter().map(|(m, _)| *m))
        .collect::<Vec<_>>();
    let checked = |a: &str, b: &str| {
        let mut i = Incremental::with_files(files.map(|x| (*x).to_owned()));
        i.set("a", a);
        i.set("b", b);
        for (m, contents) in TPARAMS_USERS {
            i.set(m, contents);
        }
        let errors = i.unchecked(&files).errors.collect_display_errors();
        (i, errors)
    };
    let touched_b = format!("{TPARAMS_B}# touched\n");
    let mut failures = Vec::new();
    for (variant, b_after, expected_stale) in [
        ("a edited", TPARAMS_B, stale.a_edited),
        ("a and b edited", &touched_b, stale.a_and_b_edited),
    ] {
        let (mut rebuilt, initial_errors) = checked(before, TPARAMS_B);
        rebuilt.set("a", after);
        if b_after != TPARAMS_B {
            rebuilt.set("b", b_after);
        }
        let result = rebuilt.unchecked(&files);
        let rebuilt_errors = result.errors.collect_display_errors();
        let (clean, clean_errors) = checked(after, b_after);
        let b_errors = |errors: &[Error]| {
            errors
                .iter()
                .filter(|e| e.path().as_path() == Path::new("b"))
                .map(|e| e.msg())
                .collect::<Vec<_>>()
        };
        assert_ne!(
            b_errors(&initial_errors),
            b_errors(&clean_errors),
            "the edit does not change the errors of `b`, so the test observes nothing"
        );
        let mut actual_stale = Vec::new();
        let mut unexpected = Vec::new();
        for m in &files {
            let got = module_snapshot(&rebuilt, &rebuilt_errors, m);
            let want = module_snapshot(&clean, &clean_errors, m);
            if got == want {
                continue;
            }
            actual_stale.push(*m);
            if !expected_stale.contains(m) {
                let stale = got
                    .iter()
                    .filter(|x| !want.contains(x))
                    .map(|x| format!("    stale:   {x}"));
                let missing = want
                    .iter()
                    .filter(|x| !got.contains(x))
                    .map(|x| format!("    missing: {x}"));
                unexpected.push(format!(
                    "`{m}` differs from a clean check\n{}",
                    stale.chain(missing).collect::<Vec<_>>().join("\n")
                ));
            }
        }
        actual_stale.sort();
        let mut expected_stale = expected_stale.to_vec();
        expected_stale.sort();
        if actual_stale != expected_stale {
            failures.push(format!(
                "[{variant}] stale modules: expected {expected_stale:?}, got {actual_stale:?} (rechecked: {:?})\n{}",
                result.changed,
                unexpected.join("\n")
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn test_tparams_pep695_bound_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        "class Foo[T: int]: ...",
        "class Foo[T: str]: ...",
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_pep695_constraints_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        "class Foo[T: (int, bytes)]: ...",
        "class Foo[T: (str, bytes)]: ...",
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_pep695_default_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        "class Foo[T = int]: ...",
        "class Foo[T = str]: ...",
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_pep695_variance_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        "class Foo[T]:\n    def get(self) -> T: ...",
        "class Foo[T]:\n    def get(self) -> T: ...\n    def set(self, x: T) -> None: ...",
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_pep695_arity_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        "class Foo[T]: ...",
        "class Foo[T, U]: ...",
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_pep695_becoming_generic_propagates() {
    assert_tparams_rebuild_matches_clean("class Foo: ...", "class Foo[T]: ...", StaleModules::NONE);
}

#[test]
fn test_tparams_pep695_becoming_non_generic_propagates() {
    assert_tparams_rebuild_matches_clean("class Foo[T]: ...", "class Foo: ...", StaleModules::NONE);
}

#[test]
fn test_tparams_pep695_order_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        "class Foo[T: int, U: str]: ...",
        "class Foo[U: str, T: int]: ...",
        StaleModules::NONE,
    );
}

/// `Foo` declared with legacy type variables.
fn legacy_foo(tvars: &str, class: &str) -> String {
    format!("from typing import Generic, TypeVar\n{tvars}\n{class}\n")
}

#[test]
fn test_tparams_legacy_bound_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        &legacy_foo("T = TypeVar('T', bound=int)", "class Foo(Generic[T]): ..."),
        &legacy_foo("T = TypeVar('T', bound=str)", "class Foo(Generic[T]): ..."),
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_legacy_constraints_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        &legacy_foo("T = TypeVar('T', int, bytes)", "class Foo(Generic[T]): ..."),
        &legacy_foo("T = TypeVar('T', str, bytes)", "class Foo(Generic[T]): ..."),
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_legacy_default_change_propagates() {
    assert_tparams_rebuild_matches_clean(
        &legacy_foo(
            "T = TypeVar('T', default=int)",
            "class Foo(Generic[T]): ...",
        ),
        &legacy_foo(
            "T = TypeVar('T', default=str)",
            "class Foo(Generic[T]): ...",
        ),
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_legacy_variance_change_propagates() {
    let class = "class Foo(Generic[T]):\n    def get(self) -> T: ...";
    assert_tparams_rebuild_matches_clean(
        &legacy_foo("T = TypeVar('T')", class),
        &legacy_foo("T = TypeVar('T', covariant=True)", class),
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_legacy_arity_change_propagates() {
    let tvars = "T = TypeVar('T')\nU = TypeVar('U')";
    assert_tparams_rebuild_matches_clean(
        &legacy_foo(tvars, "class Foo(Generic[T]): ..."),
        &legacy_foo(tvars, "class Foo(Generic[T, U]): ..."),
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_legacy_becoming_generic_propagates() {
    assert_tparams_rebuild_matches_clean(
        &legacy_foo("T = TypeVar('T')", "class Foo: ..."),
        &legacy_foo("T = TypeVar('T')", "class Foo(Generic[T]): ..."),
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_legacy_becoming_non_generic_propagates() {
    assert_tparams_rebuild_matches_clean(
        &legacy_foo("T = TypeVar('T')", "class Foo(Generic[T]): ..."),
        &legacy_foo("T = TypeVar('T')", "class Foo: ..."),
        StaleModules::NONE,
    );
}

#[test]
fn test_tparams_legacy_order_change_propagates() {
    let tvars = "T = TypeVar('T', bound=int)\nU = TypeVar('U', bound=str)";
    assert_tparams_rebuild_matches_clean(
        &legacy_foo(tvars, "class Foo(Generic[T, U]): ..."),
        &legacy_foo(tvars, "class Foo(Generic[U, T]): ..."),
        StaleModules::NONE,
    );
}

/// Check that rebuilding modules whose type parameters refer to classes, including their own
/// class, after a comment-only edit leaves their exports unchanged, so that the module `user`,
/// which imports from them, is not rechecked. Fixpoint iteration builds copies of type parameters
/// inside other type parameters, so this checks that comparing those copies in export diffing does
/// not report spurious changes, at several thread counts.
fn assert_tparams_exports_stable(sources: &[(&str, &str)], user: &str) {
    let mut files = sources.map(|(m, _)| (*m).to_owned());
    files.push("user".to_owned());
    let names = files.map(|x| x.as_str());
    for threads in [1, 2, 3, 8] {
        for _ in 0..3 {
            let mut i = Incremental::with_files_and_threads(
                files.clone(),
                ThreadCount::NumThreads(NonZeroUsize::new(threads).unwrap()),
            );
            for (m, contents) in sources {
                i.set(m, contents);
            }
            i.set("user", user);
            i.unchecked(&names);
            for (m, contents) in sources {
                i.set(m, &format!("{contents}\n# comment\n"));
            }
            let res = i.unchecked(&names);
            assert!(
                !res.changed.contains(&"user".to_owned()),
                "a comment-only edit changed the exports at {threads} threads: rechecked {:?}",
                res.changed
            );
        }
    }
}

#[test]
fn test_tparams_exports_stable_self_bound() {
    assert_tparams_exports_stable(
        &[("rec", "class N[T: 'N']: ...\nclass M[T: 'list[M]']: ...")],
        "from rec import M, N\nx: N[N]\ny: M[list[M]]",
    );
}

#[test]
fn test_tparams_exports_stable_mutual_bound() {
    assert_tparams_exports_stable(
        &[("rec", "class A[T: 'B']: ...\nclass B[U: 'A']: ...")],
        "from rec import A, B\nx: A[B[A]]",
    );
}

#[test]
fn test_tparams_exports_stable_mutual_bound_across_modules() {
    assert_tparams_exports_stable(
        &[
            ("rec", "from rec2 import B\nclass A[T: B]: ..."),
            ("rec2", "from rec import A\nclass B[U: A]: ..."),
        ],
        "from rec import A\nfrom rec2 import B\nx: A[B[A]]",
    );
}

#[test]
fn test_tparams_exports_stable_self_default() {
    assert_tparams_exports_stable(
        &[(
            "rec",
            "class D[T = 'D']: ...\nclass P[T = 'list[P]']: ...\nclass Q[T, U = 'Q[int]']: ...",
        )],
        "from rec import D, P, Q\nx: D\ny: P\nz: Q[str]",
    );
}

#[test]
fn test_tparams_exports_stable_self_constraints() {
    assert_tparams_exports_stable(
        &[("rec", "class C[T: ('C', int)]: ...")],
        "from rec import C\nx: C[int]",
    );
}

#[test]
fn test_tparams_exports_stable_restricted_bound() {
    assert_tparams_exports_stable(
        &[(
            "rec",
            "class Foo[X: int]: ...\nclass B[T: Foo[int]]: ...\ny = B[Foo[bool]]()",
        )],
        "from rec import B, Foo, y\nx: B[Foo[bool]] = y",
    );
}

#[test]
fn test_tparams_exports_stable_legacy() {
    assert_tparams_exports_stable(
        &[(
            "rec",
            r#"
from typing import Generic, TypeVar
T = TypeVar("T", bound="L")
D = TypeVar("D", default="LD")
U = TypeVar("U", bound="LB")
V = TypeVar("V", bound="LA")
class L(Generic[T]): ...
class LD(Generic[D]): ...
class LA(Generic[U]): ...
class LB(Generic[V]): ...
class Mixed[W: L[L]]: ...
"#,
        )],
        "from rec import L, LA, LB, LD, Mixed\nx: L[L]\ny: LD\nz: LA[LB[LA]]\nw: Mixed[L[L]]",
    );
}

/// Check `main`, which imports `bar`, in a project where `foo` names two files. `main` sees
/// `shadow/foo.py`, while `bar` is checked with its own configuration, sees `lib/foo.py` and
/// re-exports it. The type parameters that reach `main` through `bar` are declared in
/// `lib/foo.py`, so their bounds and defaults must come from there, even though `foo` names
/// `shadow/foo.py` in `main`.
fn check_with_shadowed_module(shadow_foo: &str) {
    const LIB_FOO: &str = r#"
from typing import Generic, TypeVar
A = TypeVar("A")
B = TypeVar("B", default=int)
C = TypeVar("C", bound=int)
class Box(Generic[B]): ...
def f(x: C) -> C: ...
def g[T: int](x: T) -> T: ...
"#;
    const MAIN: &str = r#"
from typing import assert_type
from bar import Box, f, g
def use(b: Box) -> None:
    assert_type(b, Box[int])
f(1)
f("x")  # E: `str` is not assignable to upper bound `int` of type variable `C`
g(1)
g("x")  # E: `str` is not assignable to upper bound `int` of type variable `T`
"#;
    init_test();
    let config = |modules: &[(&str, &str)]| {
        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        let mut sourcedb = MapDatabase::new(config.get_sys_info());
        for (name, path) in modules {
            sourcedb.insert(
                ModuleName::from_str(name),
                ModulePath::memory(PathBuf::from(path)),
            );
        }
        config.source_db = Some(ArcId::new(Box::new(sourcedb)));
        config.configure();
        ArcId::new(config)
    };
    let main_config = config(&[
        ("main", "main.py"),
        ("foo", "shadow/foo.py"),
        ("bar", "lib/bar.py"),
    ]);
    let lib_config = config(&[("foo", "lib/foo.py"), ("bar", "lib/bar.py")]);
    let sys_info = main_config.get_sys_info();
    let (main_load, main_fallback) = (main_config.dupe(), main_config.dupe());
    let finder = ConfigFinder::new_custom(
        Box::new(move |_, path| {
            Ok(Some(if path.as_path().starts_with("lib") {
                lib_config.dupe()
            } else {
                main_config.dupe()
            }))
        }),
        Box::new(move |_| (main_load.dupe(), Vec::new())),
        Box::new(move |_, _| (main_fallback.dupe(), Vec::new())),
        Box::new(|config, _| config),
        Box::new(|| {}),
    );
    let state = State::new(finder, TEST_THREAD_COUNT);
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction.as_mut().set_memory(
        [
            ("main.py", MAIN),
            ("shadow/foo.py", shadow_foo),
            ("lib/foo.py", LIB_FOO),
            ("lib/bar.py", "from foo import *\n"),
        ]
        .map(|(path, contents)| {
            (
                PathBuf::from(path),
                Some(Arc::new(FileContents::from_source(contents.to_owned()))),
            )
        })
        .to_vec(),
    );
    let main = Handle::new(
        ModuleName::from_str("main"),
        ModulePath::memory(PathBuf::from("main.py")),
        sys_info,
    );
    state.run_with_committing_transaction(
        transaction,
        &[main.dupe()],
        Require::Everything,
        None,
        None,
    );
    state
        .transaction()
        .get_errors([&main])
        .check_against_expectations()
        .unwrap();
}

#[test]
fn test_shadowed_module_tparams_absent_from_shadowing_module() {
    check_with_shadowed_module(
        r#"
from typing import TypeVar
A = TypeVar("A")
"#,
    );
}

#[test]
fn test_shadowed_module_tparams_differ_in_shadowing_module() {
    check_with_shadowed_module(
        r#"
from typing import Generic, TypeVar
A = TypeVar("A")
B = TypeVar("B", default=str)
C = TypeVar("C", bound=str)
class Box(Generic[B]): ...
def g[T: str](x: T) -> T: ...
"#,
    );
}

/// Editing a type parameter's bound rechecks the modules that read the bound, and not the ones
/// that only use its class.
// Bug: the bound is part of `Foo`'s `KeyTParams`, which `bar` reads to fill in `Foo`'s type
// arguments, so `bar` is rechecked too.
#[test]
fn test_tparams_bound_change_rechecks_only_readers() {
    let mut i = Incremental::new();
    i.set("foo", "class Foo[T: int]: ...");
    i.set(
        "main",
        "from foo import Foo\ndef f(x: Foo[bool]) -> None: ...",
    );
    i.set("bar", "from foo import Foo\ndef g(x: Foo) -> None: ...");
    i.check(&["foo", "main", "bar"], &["foo", "main", "bar"]);
    i.set("foo", "class Foo[T: str]: ...");
    i.check_ignoring_expectations(&["foo", "main", "bar"], &["bar", "foo", "main"]);
}

/// The messages of the errors in the last check.
fn error_messages(res: &IncrementalResult) -> Vec<String> {
    res.errors
        .collect_display_errors()
        .map(|error| error.msg().to_owned())
}

/// Renaming a class keeps its index, but the types that refer to it, such as the return type of
/// `make`, compare unequal, since the name is part of the class. So the rename recomputes `main`,
/// whose error names the class.
#[test]
fn test_class_rename_invalidates_class_dependency() {
    let mut i = Incremental::new();
    i.set("foo", "class Foo: ...\ndef make() -> Foo: ...");
    i.set(
        "main",
        "from foo import make\nx: int = make()  # E: `Foo` is not assignable to `int`",
    );
    i.check(&["main", "foo"], &["foo", "main"]);

    i.set("foo", "class Bar: ...\ndef make() -> Bar: ...");
    let res = i.unchecked(&["main", "foo"]);
    res.check_recompute(&["foo", "main"]);
    assert_eq!(error_messages(&res), ["`Bar` is not assignable to `int`"]);
}

/// Renaming a class keeps its index. `main` shows the class's name, which it reads from the class's
/// row in `foo`'s metadata, so the rename rechecks it, and its union of the class with `Mid` is
/// sorted by the new name.
#[test]
fn test_class_rename_at_same_index_matches_clean() {
    let foo = |name: &str| format!("class {name}: ...\nclass Mid: ...\nA = {name}\n");
    let main = "from foo import A, Mid\ndef f(c: bool):\n    return A() if c else Mid()\nx: int = f(True)\n";
    assert_rebuild_matches_clean(
        &[("foo", &foo("Abc")), ("main", main)],
        &[("foo", &foo("Zed"))],
        &[],
    );
}

/// An index that is out of range after its module drops classes reads the module's placeholder
/// row. Without an import cycle, every module that holds such an index is rechecked before the
/// check is committed, so, as in a clean check, no error or exported type shows the placeholder.
#[test]
fn test_dropped_classes_show_no_placeholder() {
    let files = [
        ("foo", "class A: ...\nclass B:\n    x: int = 1\nv = B()\n"),
        ("bar", "from foo import v\nw = [v]\n"),
        (
            "main",
            "from typing import reveal_type\nfrom bar import w\nreveal_type(w)\nreveal_type(w[0].x)\n",
        ),
    ];
    // `B` moves to index 0, and the old index 1 is out of range.
    assert_rebuild_matches_clean(
        &files,
        &[("foo", "class B:\n    x: int = 1\nv = B()\n")],
        &[],
    );
    // Every class is dropped.
    assert_rebuild_matches_clean(&files, &[("foo", "v = 1\n")], &[]);
}

/// Deprecating a class must recompute `main`, which sees the class only through the return type
/// of `make`. The comment keeps every range the same, so only the deprecation changes.
#[test]
fn test_class_deprecation_invalidates_class_dependency() {
    let mut i = Incremental::new();
    i.set(
        "foo",
        "from warnings import deprecated\n#deprecated('x')\nclass Foo: ...\ndef make() -> Foo: ...",
    );
    i.set(
        "main",
        "from foo import make\nx: int = make()  # E: `Foo` is not assignable to `int`",
    );
    i.check(&["main", "foo"], &["foo", "main"]);

    i.set(
        "foo",
        "from warnings import deprecated\n@deprecated('x')\nclass Foo: ...\ndef make() -> Foo: ...",
    );
    i.unchecked(&["main", "foo"])
        .check_recompute(&["foo", "main"]);
}

/// Making a class a protocol must recompute `main`, which uses the class, so `main` sees that
/// `Bar` now matches `Foo`. The padding keeps every range the same.
#[test]
fn test_class_protocol_change_invalidates_class_dependency() {
    let mut i = Incremental::new();
    i.set(
        "foo",
        "from typing import Protocol\nclass Foo(        ):\n    def m(self) -> int: ...",
    );
    i.set(
        "main",
        "from foo import Foo\nclass Bar:\n    def m(self) -> int: ...\nx: Foo = Bar()  # E: `Bar` is not assignable to `Foo`",
    );
    i.check(&["main", "foo"], &["foo", "main"]);

    i.set(
        "foo",
        "from typing import Protocol\nclass Foo(Protocol):\n    def m(self) -> int: ...",
    );
    let res = i.unchecked(&["main", "foo"]);
    res.check_recompute(&["foo", "main"]);
    assert_eq!(error_messages(&res), Vec::<String>::new());
}

/// Edits that do not change `Foo`'s definition do not recompute `main`, which only depends on
/// `Foo`: an edit above `Foo` only moves it, and adding a class after `Foo` leaves `Foo` and its
/// index the same.
#[test]
fn test_unrelated_class_edit_does_not_invalidate_class_dependency() {
    let mut i = Incremental::new();
    i.set(
        "foo",
        "def f() -> int:\n    return 1\nclass Foo: ...\ndef make() -> Foo: ...",
    );
    i.set(
        "main",
        "from foo import make\nx: int = make()  # E: `Foo` is not assignable to `int`",
    );
    i.check(&["main", "foo"], &["foo", "main"]);

    i.set(
        "foo",
        "def f() -> int:\n    return 12\nclass Foo: ...\ndef make() -> Foo: ...",
    );
    i.check(&["main", "foo"], &["foo"]);

    i.set(
        "foo",
        "def f() -> int:\n    return 12\nclass Foo: ...\ndef make() -> Foo: ...\nclass Bar: ...",
    );
    i.check(&["main", "foo"], &["foo"]);
}

/// Rebuilding a module with fewer classes moves the classes after the removed one to lower
/// indices. The modules that depend on them, directly or through a chain, must end up reporting
/// the current classes.
#[test]
fn test_fewer_classes_rebuild_reports_current_classes() {
    let mut i = Incremental::new();
    // A chain of modules makes it more likely that `main` reads `bar`'s types while `foo` is
    // being rebuilt.
    i.set("foo", "class Foo: ...\nclass Bar:\n    x: int = 1");
    i.set("bar", "from foo import Bar\nbar = Bar()");
    i.set("baz", "from bar import bar\nbaz = bar");
    i.set(
        "main",
        "from typing import reveal_type\nfrom baz import baz\nreveal_type(baz)  # E: revealed type: Bar\ny: str = baz.x  # E: `int` is not assignable to `str`",
    );
    i.check(&["main", "bar", "baz"], &["foo", "bar", "baz", "main"]);

    i.set("foo", "class Bar:\n    x: int = 1");
    let res = i.unchecked(&["main", "foo"]);
    assert_eq!(
        error_messages(&res),
        ["revealed type: Bar", "`int` is not assignable to `str`"]
    );
}

/// A module can see its own classes through the modules it imports. `foo` is rebuilt with fewer
/// classes and reads `Sub` from `bar`, which is in an import cycle with `foo`. `bar` is only
/// rebuilt after `foo`'s exports change, so `foo` first reads the MRO of `Sub` that `bar` computed
/// against the old `foo`, which names `Foo` at its old index. `foo` treats it as a class it no
/// longer has; rebuilding `bar` then changes the MRO of `Sub`, so `foo` is rebuilt against it.
#[test]
fn test_own_class_read_through_cycle_after_fewer_classes() {
    let mut i = Incremental::new();
    i.set(
        "foo",
        "from bar import Sub\nclass Baz: ...\nclass Foo: ...\nsub = Sub()",
    );
    i.set("bar", "from foo import Foo\nclass Sub(Foo): ...");
    i.check(&["foo", "bar"], &["foo", "bar"]);

    i.set("foo", "from bar import Sub\nclass Foo: ...\nsub = Sub()");
    i.check_matches_clean(&["foo", "bar"], &["bar", "foo", "foo"]);
}

/// Like `test_own_class_read_through_cycle_after_fewer_classes`, but `foo` reads its enum `E` back
/// through `bar` and narrows by one of its members, which looks up the fields and the metadata of
/// `E` at its old index. `foo` leaves the value unnarrowed until it is rebuilt against the new
/// `bar`.
#[test]
fn test_own_enum_read_through_cycle_after_fewer_classes() {
    let foo = |classes: &str| {
        format!(
            "from enum import Enum\nfrom bar import DEFAULT, get\n{classes}class E(Enum):\n    X = 1\n    Y = 2\nx = get()\nif x is not DEFAULT:\n    y = x\n"
        )
    };
    let mut i = Incremental::new();
    i.set("foo", &foo("class Baz: ...\n"));
    i.set(
        "bar",
        "from typing import Final\nfrom foo import E\ndef get() -> E: ...\nDEFAULT: Final = E.X",
    );
    i.check(&["foo", "bar"], &["foo", "bar"]);

    i.set("foo", &foo(""));
    i.check_matches_clean(&["foo", "bar"], &["bar", "foo", "foo"]);
}

/// Like `test_own_class_read_through_cycle_after_fewer_classes`, but the old index of `B` is still
/// in range, at the class `C` that moved into it, so `foo` first reads `x` from `C`. `y` changes
/// again when `foo` is rebuilt against the new `bar`, so the cycle fallback rebuilds both modules
/// once more.
#[test]
fn test_own_class_read_through_cycle_at_index_of_other_class() {
    let foo = |classes: &str| {
        format!(
            "from bar import Sub\n{classes}class B:\n    x: int = 0\nclass C:\n    x: str = \"\"\ny = Sub().x\n"
        )
    };
    let mut i = Incremental::new();
    i.set("foo", &foo("class A: ...\n"));
    i.set("bar", "from foo import B\nclass Sub(B): ...");
    i.check(&["foo", "bar"], &["foo", "bar"]);

    i.set("foo", &foo(""));
    i.check_matches_clean(&["foo", "bar"], &["bar", "bar", "foo", "foo", "foo"]);
}

// In the `test_stale_cycle_` tests below, the edit makes `a` read back, through `b`, a value that
// `b` computed against the old `a`. When that value names a class of `a`, it names it by its old
// index. A class compares by its module, index, name and a few other facts, but not by its fields
// or type parameters, so the exports of `a` that `b` reads often compare equal to their old
// answers. Then `b` is not rechecked, and `a`, `b` and `main`, which observes them, keep the stale
// value. Where `a` and `b` only refer to each other, as in most of these tests, a clean check gives
// `Unknown`.

/// `a` drops `A`, which moves `B` to index 0, and reads back `w`, which names `B` at index 1.
// Bug: `a` reports that `<placeholder>`, the row of the out-of-range old index of `B`, has no
// attribute `x`, and `main` reveals `<placeholder>`.
#[test]
fn test_stale_cycle_class_out_of_range_attribute_in_owner() {
    assert_rebuild_matches_clean(
        &[
            ("a", "class A: ...\nclass B:\n    x: int = 1\nv = B()\n"),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass B:\n    x: int = 1\nv = w\nz = v.x\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, z\nreveal_type(v)\nreveal_type(z)\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// Like `test_stale_cycle_class_out_of_range_attribute_in_owner`, but `v` names `B` at its new
/// index, so it changes and `b` is rechecked. `a`'s exports then change again, which `run_internal`
/// reports as a cycle, so its fallback rechecks `a`, `b` and `main` together.
#[test]
fn test_stale_cycle_class_out_of_range_corrected_by_recheck() {
    assert_rebuild_matches_clean(
        &[
            ("a", "class A: ...\nclass B:\n    x: int = 1\nv = B()\n"),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass B:\n    x: int = 1\nv: B = B()\nz = w.x\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, z\nreveal_type(v)\nreveal_type(z)\n",
            ),
        ],
        &[],
    );
}

/// Like `test_stale_cycle_class_out_of_range_attribute_in_owner`, through a cycle of three modules:
/// `foo` reads back `x` through `bar` and `baz`.
// Bug: `foo` reports that `<placeholder>`, the row of the out-of-range old index of `B`, has no
// attribute `x`, and `main` reveals `<placeholder>`.
#[test]
fn test_stale_cycle_of_three_modules_class_out_of_range() {
    assert_rebuild_matches_clean(
        &[
            ("foo", "class A: ...\nclass B:\n    x: int = 1\nx = B()\n"),
            ("bar", "import baz\ny = baz.z\n"),
            ("baz", "import foo\nz = foo.x\n"),
            ("main", "import foo\n"),
        ],
        &[
            (
                "foo",
                "import bar\nclass B:\n    x: int = 1\nx = bar.y\nq = x.x\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom foo import x, q\nreveal_type(x)\nreveal_type(q)\n",
            ),
        ],
        &["foo", "bar", "baz", "main"],
    );
}

/// `a` drops `A`, which moves `B` to index 0, and reads back `w`, which names `B` at index 1. Only
/// `main` reads its fields.
// Bug: `main` reveals `<placeholder>`, the row of the out-of-range old index of `B`, and reports
// that `<placeholder>` has no attribute `x`.
#[test]
fn test_stale_cycle_class_out_of_range() {
    let files = [
        ("a", "class A: ...\nclass B:\n    x: int = 1\nv = B()\n"),
        ("b", "from a import v\nw = v\n"),
        ("main", "import a\n"),
    ];
    let edits = [
        ("a", "from b import w\nclass B:\n    x: int = 1\nv = w\n"),
        (
            "main",
            "from typing import reveal_type\nfrom a import v\nreveal_type(v)\nreveal_type(v.x)\n",
        ),
    ];
    assert_rebuild_matches_clean(&files, &edits, &["a", "b", "main"]);
    assert_eq!(
        rebuilt_errors(&files, &edits, "main"),
        [
            "error 3:12-15 RevealType: revealed type: <placeholder>",
            "error 4:12-17 RevealType: revealed type: Unknown",
            "error 4:13-16 MissingAttribute: Object of class `<placeholder>` has no attribute `x`",
        ]
    );
}

/// `a` swaps `A` and `B`, and reads back `w`, which names index 0 as `A`, where `a` now defines
/// `B`.
// Bug: `main` reveals `B`, the class now at the old index of `A`, reports that `B` has no attribute
// `x`, and reveals `v.y` as `str`.
#[test]
fn test_stale_cycle_class_displaced() {
    let files = [
        (
            "a",
            "class A:\n    x: int = 1\nclass B:\n    y: str = ''\nv = A()\n",
        ),
        ("b", "from a import v\nw = v\n"),
        ("main", "import a\n"),
    ];
    let edits = [
        (
            "a",
            "from b import w\nclass B:\n    y: str = ''\nclass A:\n    x: int = 1\nv = w\n",
        ),
        (
            "main",
            "from typing import reveal_type\nfrom a import v\nreveal_type(v)\nreveal_type(v.x)\nreveal_type(v.y)\n",
        ),
    ];
    assert_rebuild_matches_clean(&files, &edits, &["a", "b", "main"]);
    assert_eq!(
        rebuilt_errors(&files, &edits, "main"),
        [
            "error 3:12-15 RevealType: revealed type: B",
            "error 4:12-17 RevealType: revealed type: Unknown",
            "error 4:13-16 MissingAttribute: Object of class `B` has no attribute `x`",
            "error 5:12-17 RevealType: revealed type: str",
        ]
    );
}

/// `a` swaps `A[T]` and `B`, and reads back `w`, `A[str]`, whose index 0 now names the non-generic
/// `B`.
// Bug: `main` reveals `B[str]`, with `B`, the class now at the old index of `A`, and `v.get()` as
// `int`, and reports that `B[str]` is not assignable to `A[str]`.
#[test]
fn test_stale_cycle_generic_class_displaced_by_plain_class() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T]:\n    def get(self) -> T: ...\nclass B:\n    def get(self) -> int: ...\nv = A[str]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass B:\n    def get(self) -> int: ...\nclass A[T]:\n    def get(self) -> T: ...\nv = w\nu = v.get()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, A, B\nreveal_type(v)\nreveal_type(v.get())\nx: B = v\ny: A[str] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `a` swaps `A` and `B[T]`, and reads back `w`, `A`, whose index 0 now names the generic `B`.
// Bug: `main` reveals `B`, the class now at the old index of `A`, and `v.get()` as the escaped type
// variable `T` of `B`.
#[test]
fn test_stale_cycle_plain_class_displaced_by_generic_class() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A:\n    def get(self) -> int: ...\nclass B[T]:\n    def get(self) -> T: ...\nv = A()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass B[T]:\n    def get(self) -> T: ...\nclass A:\n    def get(self) -> int: ...\nv = w\nu = v.get()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, B\nreveal_type(v)\nreveal_type(v.get())\nx: B[int] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `A` gains a type parameter, and `a` reads back `w`, `A[int]`.
// Bug: `main` reveals `A[int]`, and `u` and `v.get()` as the escaped type variable `U`.
#[test]
fn test_stale_cycle_type_parameter_added() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T]:\n    def get(self) -> T: ...\nv = A[int]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[T, U]:\n    def get(self) -> U: ...\nv = w\nu = v.get()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, u, A\nreveal_type(v)\nreveal_type(u)\nreveal_type(v.get())\nx: A[int, str] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `A` loses a type parameter, and `a` reads back `w`, `A[int, str]`.
// Bug: `main` reveals `A[int, str]`, and `u` and `v.get()` as the escaped type variable `T`.
#[test]
fn test_stale_cycle_type_parameter_removed() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T, U]:\n    def get(self) -> U: ...\nv = A[int, str]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[T]:\n    def get(self) -> T: ...\nv = w\nu = v.get()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, u, A\nreveal_type(v)\nreveal_type(u)\nreveal_type(v.get())\nx: A[int] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// The type variable of `A` becomes a ParamSpec, and `a` reads back `w`, `A[int]`.
// Bug: `a` and `main` report that `v.call(1)` expects `*`-unpacked `P.args` and `**`-unpacked
// `P.kwargs`, and `main` reveals `A[int]` and `v.call` as `(ParamSpec(P)) -> None`, and reports that
// `A[int]` is not assignable to `A[[int]]`.
#[test]
fn test_stale_cycle_type_var_becomes_param_spec() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T]:\n    def call(self, x: T) -> T: ...\nv = A[int]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[**P]:\n    def call(self, *args: P.args, **kwargs: P.kwargs) -> None: ...\nv = w\nv.call(1)\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, A\nreveal_type(v)\nv.call(1)\nreveal_type(v.call)\nx: A[[int]] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// The type variable of `A` becomes a TypeVarTuple, and `a` reads back `w`, `A[int]`.
// Bug: `a` and `main` report that `tuple[Literal[1]]` is not assignable to `tuple[*Ts]` in
// `v.call(1)`, and `main` reveals `A[int]` and reports that `A[int]` is not assignable to `A[int]`.
#[test]
fn test_stale_cycle_type_var_becomes_type_var_tuple() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T]:\n    def call(self, x: T) -> T: ...\nv = A[int]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[*Ts]:\n    def call(self, *args: *Ts) -> tuple[*Ts]: ...\nv = w\nv.call(1)\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, A\nreveal_type(v)\nreveal_type(v.call(1))\nreveal_type(v.call)\nx: A[int] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// The ParamSpec of `A` becomes a type variable, and `a` reads back `w`, `A[[int, str]]`.
// Bug: `a` and `main` report that `Literal[1]` is not assignable to `T` in `v.call(1)`, and `main`
// reveals `A[[int, str]]` and `v.call(1)` as `T`, and reports that `A[[int, str]]` is not
// assignable to `A[int]`.
#[test]
fn test_stale_cycle_param_spec_becomes_type_var() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[**P]:\n    def call(self, *args: P.args, **kwargs: P.kwargs) -> None: ...\nv = A[[int, str]]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[T]:\n    def call(self, x: T) -> T: ...\nv = w\nu = v.call(1)\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, A\nreveal_type(v)\nreveal_type(v.call(1))\nx: A[int] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `A[*Ts]` becomes `A[T, *Ts]`, and `a` reads back `w`, `A[int, str]`, whose arguments all belong
/// to `*Ts`.
// Bug: `a` and `main` report that `Literal[1]` is not assignable to `T`, and `main` reports that
// `tuple[Literal['x']]` is not assignable to `tuple[*Ts]`, reveals `A[int, str]` and
// `v.call(1, 'x')` as `tuple[*Ts]`, and reports that `A[int, str]` is not assignable to
// `A[int, str]`.
#[test]
fn test_stale_cycle_type_var_added_before_type_var_tuple() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[*Ts]:\n    def call(self, *args: *Ts) -> tuple[*Ts]: ...\nv = A[int, str]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[T, *Ts]:\n    def call(self, x: T, *args: *Ts) -> tuple[*Ts]: ...\nv = w\nu = v.call(1)\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, A\nreveal_type(v)\nreveal_type(v.call(1, 'x'))\nx: A[int, str] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `A` becomes invariant, and `a` reads back `w`, `A[bool]`.
// Bug: `a` and `main` report that `A[bool]` is not assignable to `A[int]`, and `main` reveals
// `A[bool]`.
#[test]
fn test_stale_cycle_variance_changes() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T]:\n    def get(self) -> T: ...\nv = A[bool]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[T]:\n    def get(self) -> T: ...\n    def set(self, x: T) -> None: ...\nv = w\nu: A[int] = v\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, A\nreveal_type(v)\nx: A[int] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// The bound of the type variable of `A` changes from `int` to `str`, and `a` reads back `w`,
/// `A[int]`.
// Bug: `main` reveals `A[int]`, and `v.get()` as the escaped type variable `T`.
#[test]
fn test_stale_cycle_bound_changes() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T: int]:\n    def get(self) -> T: ...\nv = A[int]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[T: str]:\n    def get(self) -> T: ...\nv = w\nu = v.get()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, A\nreveal_type(v)\nreveal_type(v.get())\nx: A[int] = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `A` loses a type parameter, and `a` reads back `w`, `A[int, str]`, and uses it as an `A[int]`.
// Bug: `main` reveals `A[int, str]`, and `z` as the escaped type variable `T`.
#[test]
fn test_stale_cycle_type_parameter_removed_owner_specializes() {
    assert_rebuild_matches_clean(
        &[
            ("a", "class A[T, U]:\n    x: U\nv = A[int, str]()\n"),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass A[T]:\n    x: T\nv = w\ny: A[int] = v\nz = v.x\nclass C(A[int]): ...\nq: A[int] = C()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, z\nreveal_type(v)\nreveal_type(z)\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `a` drops `f` and reads back `g`, `b`'s alias of the old `f`. `a`'s exports change again after
/// `b` is rechecked, which `run_internal` reports as a cycle, so its fallback rechecks `a`, `b` and
/// `main` together.
#[test]
fn test_stale_cycle_function_removed() {
    assert_rebuild_matches_clean(
        &[
            ("a", "def f[T: int](x: T) -> T: ...\n"),
            ("b", "from a import f\ng = f\n"),
            ("main", "import a\n"),
        ],
        &[
            ("a", "from b import g\nh = g\n"),
            (
                "main",
                "from typing import reveal_type\nfrom a import h\nreveal_type(h)\n",
            ),
        ],
        &[],
    );
}

/// `a` declares `k` before `f` and reads back `g`, `b`'s alias of the old `f`, whose type parameter
/// has the declaration index that the type parameter of `k` now has. As in
/// `test_stale_cycle_function_removed`, the recheck ends in the cycle fallback.
#[test]
fn test_stale_cycle_function_type_parameter_displaced() {
    assert_rebuild_matches_clean(
        &[
            ("a", "def f[T: int](x: T) -> T: ...\n"),
            ("b", "from a import f\ng = f\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import g\ndef k[S: str](x: S) -> S: ...\ndef f[T: int](x: T) -> T: ...\nh = g\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import h\nreveal_type(h(1))\nh('x')\n",
            ),
        ],
        &[],
    );
}

/// `a` drops `e`, declared before `f`, and reads back `g`, `b`'s alias of the old `f`, whose type
/// parameter has a declaration index that `a` no longer has. As in
/// `test_stale_cycle_function_removed`, the recheck ends in the cycle fallback.
#[test]
fn test_stale_cycle_function_type_parameter_removed() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "def e[R](x: R) -> R: ...\ndef f[T: int](x: T) -> T: ...\n",
            ),
            ("b", "from a import f\ng = f\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import g\ndef f[T: int](x: T) -> T: ...\nh = g\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import h\nreveal_type(h(1))\nh('x')\n",
            ),
        ],
        &[],
    );
}

/// `a` declares `Z[**P = [int, str]]` before `A`, and reads back `w` and `e`, which name index 0 as
/// `A`, where `a` now defines `Z`.
// Bug: `a` and `main` report that `Z`, the class now at the old index of `A`, has no attribute
// `get`, and `main` reveals `Z[str]` and `Z`.
#[test]
fn test_stale_cycle_class_displaced_by_param_spec_default() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class A[T = int]:\n    def get(self) -> T: ...\nv = A[str]()\nd = A()\n",
            ),
            ("b", "from a import v, d\nw = v\ne = d\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w, e\nclass Z[**P = [int, str]]: ...\nclass A[T = int]:\n    def get(self) -> T: ...\nv = w\nd = e\nu = d.get()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, d\nreveal_type(v)\nreveal_type(d)\nreveal_type(d.get())\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `a` swaps `A` and `B`, whose legacy type parameter `U` defaults to `T`, and reads back `w`,
/// `A[int, int]`, which names index 0 as `A`, where `a` now defines `B`.
// Bug: `a` and `main` report that `B`, the class now at the old index of `A`, has no attribute
// `get`, and `main` reveals `B[int, int]`.
#[test]
fn test_stale_cycle_legacy_default_owner_displaced() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "from typing import Generic, TypeVar\nT = TypeVar('T')\nU = TypeVar('U', default=T)\nclass A(Generic[T, U]):\n    def get(self) -> U: ...\nclass B(Generic[U]): ...\nv = A[int]()\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from typing import Generic, TypeVar\nfrom b import w\nT = TypeVar('T')\nU = TypeVar('U', default=T)\nclass B(Generic[U]): ...\nclass A(Generic[T, U]):\n    def get(self) -> U: ...\nv = w\nu = v.get()\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v\nreveal_type(v)\nreveal_type(v.get())\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `a` renames `Abc`, at index 0, to `Zzz`, and reads back `w`, the union `Abc | Zed`.
// Bug: `main` reveals `Zed | Zzz`, with the new name of the class at index 0, and
// `list[Zed | Zzz]`.
#[test]
fn test_stale_cycle_union_after_class_rename() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "class Abc: ...\nclass Zed: ...\ndef mk(c: bool):\n    return Abc() if c else Zed()\nv = mk(True)\n",
            ),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\nclass Zzz: ...\nclass Zed: ...\nv = w\nu = [v, Zed(), Zzz()]\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, u, Zed, Zzz\nreveal_type(v)\nreveal_type(u)\nx: Zed | Zzz = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `a` drops `A` and reads back `w`, which names `B` at its old index, while the annotation `B` in
/// `a` names it at its new index.
// Bug: `a` and `main` report that `<placeholder>`, the row of the out-of-range old index of `B`,
// is not assignable to `B`, and `main` reveals `<placeholder>`.
#[test]
fn test_stale_cycle_class_index_of_two_versions() {
    assert_rebuild_matches_clean(
        &[
            ("a", "class A: ...\nclass B: ...\nv = B()\n"),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            ("a", "from b import w\nclass B: ...\nv = w\nu: B = v\n"),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, B\nreveal_type(v)\nx: B = v\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `a` swaps the enum `E` and the class `P`, and reads back `w` and `y`, which name index 0 as `E`,
/// where `a` now defines `P`.
// Bug: `main` reveals `v` as `P`, the class now at the old index of `E`, after narrowing.
#[test]
fn test_stale_cycle_enum_displaced() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "from enum import Enum\nclass E(Enum):\n    X = 1\n    Y = 2\nclass P: ...\ndef mk() -> E: ...\nv = mk()\nx = E.X\n",
            ),
            ("b", "from a import v, x\nw = v\ny = x\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from enum import Enum\nfrom b import w, y\nclass P: ...\nclass E(Enum):\n    X = 1\n    Y = 2\nv = w\nx = y\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, x\nif v is not x:\n    reveal_type(v)\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// Like `test_stale_cycle_enum_displaced`, with uppercase constants, whose types keep the member
/// literal, so `main` narrows by the member.
// Bug: `main` reveals `XM` as `Literal[P.X]`, where `P` is the class now at the old index of `E`,
// and `V` as `P` after narrowing by `is not` and `!=`.
#[test]
fn test_stale_cycle_enum_constant_displaced() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "from enum import Enum\nclass E(Enum):\n    X = 1\n    Y = 2\nclass P: ...\ndef mk() -> E: ...\nV = mk()\nXM = E.X\n",
            ),
            ("b", "from a import V, XM\nW = V\nYM = XM\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from enum import Enum\nfrom b import W, YM\nclass P: ...\nclass E(Enum):\n    X = 1\n    Y = 2\nV = W\nXM = YM\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import V, XM\nreveal_type(XM)\nif V is not XM:\n    reveal_type(V)\nif V != XM:\n    reveal_type(V)\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// `a` drops the type alias `A0`, and reads back `w`, which refers to `A1` at its old index, now
/// out of range.
// Bug: `main` reveals `int | list[A1]` and `list[A1]`.
#[test]
fn test_stale_cycle_type_alias_out_of_range() {
    assert_rebuild_matches_clean(
        &[
            ("a", "type A0 = int\ntype A1 = list[A1] | int\nv: A1 = 1\n"),
            ("b", "from a import v\nw = v\n"),
            ("main", "import a\n"),
        ],
        &[
            (
                "a",
                "from b import w\ntype A1 = list[A1] | int\nv = w\nu = [v]\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom a import v, u\nreveal_type(v)\nreveal_type(u)\n",
            ),
        ],
        &["a", "b", "main"],
    );
}

/// Without a cycle: `a` declares `k` before `f`, which moves the declaration of the type parameter
/// of `f`. `main` calls `g`, `b`'s alias of `f`.
#[test]
fn test_function_type_parameter_displaced() {
    assert_rebuild_matches_clean(
        &[
            ("a", "def f[T: int](x: T) -> T: ...\n"),
            ("b", "from a import f\ng = f\n"),
            ("main", "import b\n"),
        ],
        &[
            (
                "a",
                "def k[S: str](x: S) -> S: ...\ndef f[T: int](x: T) -> T: ...\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom b import g\nreveal_type(g(1))\ng('x')\n",
            ),
        ],
        &[],
    );
}

/// Without a cycle: `a` declares a type alias with a ParamSpec default before `A`, which moves the
/// declaration of the type parameter of `A`. The instances of `A` that `b` keeps pair with the
/// parameters of the moved `A`, so `main` sees the same types as in a clean check.
#[test]
fn test_class_type_parameter_default_displaced() {
    assert_rebuild_matches_clean(
        &[
            ("a", "class A[T = int]:\n    def get(self) -> T: ...\n"),
            ("b", "from a import A\nw = A[str]()\ne: A = A()\n"),
            ("main", "import b\n"),
        ],
        &[
            (
                "a",
                "type Z[**P = [int, str]] = int\nclass A[T = int]:\n    def get(self) -> T: ...\n",
            ),
            (
                "main",
                "from typing import reveal_type\nimport b\nreveal_type(b.w)\nreveal_type(b.e)\nreveal_type(b.w.get())\nreveal_type(b.e.get())\n",
            ),
        ],
        &[],
    );
}

/// Without a cycle: moving `A` down `a` moves the declaration of its type parameter. The `A[int]`
/// that `b` keeps pairs with the parameter of the moved `A`, so `main` reveals `int`.
#[test]
fn test_generic_class_moved() {
    assert_rebuild_matches_clean(
        &[
            ("a", "class A[T]:\n    def get(self) -> T: ...\n"),
            ("b", "from a import A\nw = A[int]()\n"),
            ("main", "import b\n"),
        ],
        &[
            ("a", "x = 1\nclass A[T]:\n    def get(self) -> T: ...\n"),
            (
                "main",
                "from typing import reveal_type\nimport b\nreveal_type(b.w)\nreveal_type(b.w.get())\n",
            ),
        ],
        &[],
    );
}

/// Moving a generic class gives its type parameters no new identity, so the modules that keep
/// its instances are not rechecked.
#[test]
fn test_generic_class_moved_rechecks_only_its_module() {
    let mut i = Incremental::new();
    i.set("foo", "class A[T]:\n    def get(self) -> T: ...\n");
    i.set("bar", "from foo import A\nw = A[int]()\n");
    i.set(
        "main",
        "from typing import assert_type\nfrom bar import w\nassert_type(w.get(), int)\n",
    );
    i.check(&["main"], &["main", "foo", "bar"]);
    i.set("foo", "x = 1\n\nclass A[T]:\n    def get(self) -> T: ...\n");
    i.check(&["main"], &["foo"]);
}

/// A legacy generic class moves down its module, and another module subclasses an instance of it.
#[test]
fn test_legacy_generic_class_moved_under_subclass() {
    assert_rebuild_matches_clean(
        &[
            (
                "a",
                "from typing import Generic, TypeVar\nT = TypeVar('T')\nclass A(Generic[T]):\n    def get(self) -> T: ...\n",
            ),
            ("b", "from a import A\nclass B(A[int]): ...\n"),
            ("main", "import b\n"),
        ],
        &[
            (
                "a",
                "from typing import Generic, TypeVar\nT = TypeVar('T')\nx = 1\nclass A(Generic[T]):\n    def get(self) -> T: ...\n",
            ),
            (
                "main",
                "from typing import reveal_type\nfrom b import B\nreveal_type(B().get())\nx: int = B().get()\n",
            ),
        ],
        &[],
    );
}

/// A generic class moves down its module, and another module returns an instance of it.
#[test]
fn test_generic_class_moved_under_return_annotation() {
    assert_rebuild_matches_clean(
        &[
            ("a", "class A[T]:\n    def get(self) -> T: ...\n"),
            ("b", "from a import A\ndef make() -> A[int]: ...\n"),
            ("main", "import b\n"),
        ],
        &[
            ("a", "\nclass A[T]:\n    def get(self) -> T: ...\n"),
            (
                "main",
                "import b\nx: int = b.make().get()\ny: str = b.make().get()\n",
            ),
        ],
        &[],
    );
}

/// A module in an import cycle can build a union from classes that it reads back through the
/// cycle before the other module is rebuilt. `foo` loses its first class, so each of its classes
/// moves down one index, and `foo` first reads `members` from the `bar` that was checked against
/// the old `foo`. `z` then holds each old class beside the new class at the same index, which is
/// the same class by identity but has another name and position. Sorting the union must not
/// panic, and the rebuild must equal a clean check at every thread count.
#[test]
fn test_displaced_classes_in_union_through_cycle() {
    let names = (0..24).map(|i| format!("A{i:02}")).collect::<Vec<_>>();
    let union = names.join(" | ");
    let classes = names
        .iter()
        .map(|name| format!("class {name}: ...\n"))
        .collect::<String>();
    let foo = |padding: &str| {
        format!(
            "from bar import members\n{padding}{classes}def cond() -> bool: ...\ndef get() -> {union}: ...\nz = members if cond() else get()\n"
        )
    };
    let bar = format!(
        "from foo import {}\ndef get() -> {union}: ...\nmembers = get()\n",
        names.join(", ")
    );
    assert_rebuild_matches_clean(
        &[("foo", &foo("class Pad: ...\n")), ("bar", &bar)],
        &[("foo", &foo(""))],
        &[],
    );
}

/// A union keeps its class members ordered by name, so renaming a class in one module can reorder
/// a union in another, even when no export of the first module changes and the second module reads
/// nothing else about the class. The rename changes `obj`, whose class has a new name, so `bar` is
/// rechecked and its rebuild equals a clean check.
#[test]
fn test_class_rename_reorders_union_in_importer() {
    let foo = |name: &str| {
        format!("def make():\n    class {name}: ...\n    return {name}()\nobj = make()\n")
    };
    let bar = r#"
from foo import obj
class Bar: ...
def f(c: bool):
    return obj if c else Bar()
"#;
    let exported = |solutions: &Solutions| {
        ["obj", "Bar", "f"]
            .map(|name| format!("{:?}", solutions.get(&KeyExport(Name::new_static(name)))))
    };

    let mut i = Incremental::new();
    i.set("foo", &foo("Abc"));
    i.set("bar", bar);
    i.check(&["bar"], &["bar", "foo"]);
    i.set("foo", &foo("Foo"));
    i.unchecked(&["bar"]).check_recompute(&["bar", "foo"]);
    let incremental = i
        .state
        .transaction()
        .get_solutions(&i.handle("bar"))
        .unwrap();

    let mut clean = Incremental::new();
    clean.set("foo", &foo("Foo"));
    clean.set("bar", bar);
    clean.check(&["bar"], &["bar", "foo"]);
    let expected = clean
        .state
        .transaction()
        .get_solutions(&clean.handle("bar"))
        .unwrap();

    assert_eq!(exported(&incremental), exported(&expected));
}

/// The error for a descriptor whose `__get__` cannot be called is reported at the read in `main`,
/// so it moves with `main`, not with the file that defines the descriptor.
#[test]
fn test_descriptor_error_location_after_edit_in_defining_module() {
    let mut i = Incremental::new();
    let foo = |padding: &str| {
        format!(
            r#"{padding}
class D:
    @property
    def __get__(self, extra: int) -> int: ...

class C:
    d = D()
"#
        )
    };
    let main = format!("from foo import C\nv = C().d\n{}", "x = 1\n".repeat(40));
    i.set("foo", &foo(""));
    i.set("main", &main);
    i.check_matches_fresh_check(&["main", "foo"]);

    i.set("foo", &foo("# a\n# b\n# c"));
    i.check_matches_fresh_check(&["main", "foo"]);
}

/// A dataclass reports the descriptors of its own fields only, so an edit to the module of its
/// base class does not leave an error at a stale location in the subclass's module.
#[test]
fn test_dataclass_descriptor_error_location_after_edit_in_base_module() {
    let mut i = Incremental::new();
    let foo = |padding: &str| {
        format!(
            r#"{padding}
from dataclasses import dataclass

class NonData:
    def __get__(self, obj: object, objtype: object) -> int: ...

@dataclass
class Base:
    x: NonData = NonData()
"#
        )
    };
    let main = format!(
        "from dataclasses import dataclass\nfrom foo import Base\n@dataclass\nclass Child(Base):\n    y: int = 0\n{}",
        "z = 1\n".repeat(40)
    );
    i.set("foo", &foo(""));
    i.set("main", &main);
    i.check_matches_fresh_check(&["main", "foo"]);

    i.set("foo", &foo("# a\n# b\n# c"));
    i.check_matches_fresh_check(&["main", "foo"]);
}

/// A module that reads one field of a class is not rechecked when another field of the class
/// changes type: it depends on the keys of the class that it reads, not on the whole class.
#[test]
fn test_reader_of_unchanged_class_field_is_not_rechecked() {
    let mut i = Incremental::new();
    i.set("foo", "class Hub:\n    a: int = 1\n    b: int = 2");
    i.set(
        "bar",
        "from foo import Hub\ndef f(h: Hub) -> int:\n    return h.a",
    );
    i.set(
        "main",
        "from foo import Hub\ndef g(h: Hub) -> int:\n    return h.b",
    );
    i.check(&["main", "bar", "foo"], &["main", "bar", "foo"]);

    i.set("foo", "class Hub:\n    a: int = 1\n    b: str = ''");
    i.check_matches_clean(&["main", "bar", "foo"], &["foo", "main"]);
}

/// Declaring another field changes the fields that the class declares, which every attribute
/// lookup on the class reads, so every module that looks up an attribute of the class is
/// rechecked. Moving the class, which changes no declaration, rechecks no reader.
#[test]
fn test_class_declarations_change_rechecks_attribute_readers() {
    let mut i = Incremental::new();
    i.set("foo", "class Hub:\n    a: int = 1\n    b: int = 2");
    i.set(
        "bar",
        "from foo import Hub\ndef f(h: Hub) -> int:\n    return h.a",
    );
    i.set(
        "main",
        "from foo import Hub\ndef g(h: Hub) -> int:\n    return h.b",
    );
    i.check(&["main", "bar", "foo"], &["main", "bar", "foo"]);

    i.set(
        "foo",
        "# moved\n\nclass Hub:\n    a: int = 1\n    b: int = 2",
    );
    i.check(&["main", "bar", "foo"], &["foo"]);

    i.set(
        "foo",
        "# moved\n\nclass Hub:\n    a: int = 1\n    b: int = 2\n    c: int = 3",
    );
    i.check_matches_clean(&["main", "bar", "foo"], &["bar", "foo", "main"]);
}

const FIELDS: ClassRowChange = ClassRowChange {
    fields: true,
    def: false,
};
const DEF: ClassRowChange = ClassRowChange {
    fields: false,
    def: true,
};
const BOTH: ClassRowChange = ClassRowChange {
    fields: true,
    def: true,
};

/// The class rows that differ between two versions of a module, by class index.
fn class_row_changes(old: &str, new: &str) -> Vec<(u32, ClassRowChange)> {
    let rows = |code: &str| {
        let (state, handle) = TestEnv::one("main", code).to_state();
        state
            .transaction()
            .get_bindings_metadata(&handle("main"))
            .unwrap()
    };
    rows(old)
        .changed_class_rows(&rows(new))
        .map(|(class, change)| (class.0, change))
        .collect()
}

#[test]
fn test_class_row_changes_of_definition_facts() {
    let fields = "    x: int = 0";
    // A rename, of the class or of a class that encloses it.
    assert_eq!(
        class_row_changes(
            &format!("class A:\n{fields}"),
            &format!("class B:\n{fields}")
        ),
        [(0, DEF)]
    );
    assert_eq!(
        class_row_changes("class A:\n    class B: ...", "class C:\n    class B: ..."),
        [(0, DEF), (1, DEF)]
    );
    // Becoming a protocol.
    assert_eq!(
        class_row_changes(
            &format!("from typing import Protocol\nclass A:\n{fields}"),
            &format!("from typing import Protocol\nclass A(Protocol):\n{fields}"),
        ),
        [(0, DEF)]
    );
    // Becoming deprecated.
    assert_eq!(
        class_row_changes(
            &format!("from warnings import deprecated\nclass A:\n{fields}"),
            &format!("from warnings import deprecated\n@deprecated('old')\nclass A:\n{fields}"),
        ),
        [(0, DEF)]
    );
    // Declaring type parameters.
    assert_eq!(
        class_row_changes(
            &format!("class A:\n{fields}"),
            &format!("class A[T]:\n{fields}")
        ),
        [(0, DEF)]
    );
}

#[test]
fn test_class_row_changes_ignore_positions() {
    // Moving a class changes neither its fields nor its definition.
    assert_eq!(
        class_row_changes(
            "class A:\n    x: int = 0",
            "def f() -> None: ...\n\nclass A:\n    x: int = 0"
        ),
        []
    );
    // A field change leaves the definition alone.
    assert_eq!(
        class_row_changes("class A:\n    x: int = 0", "class A:\n    y: int = 0"),
        [(0, FIELDS)]
    );
}

#[test]
fn test_class_row_changes_of_added_and_removed_classes() {
    let two = "class A: ...\nclass B: ...";
    assert_eq!(class_row_changes(two, "class A: ..."), [(1, BOTH)]);
    assert_eq!(class_row_changes("class A: ...", two), [(1, BOTH)]);
}

/// Edits to the fields of a class in `foo`, read by `main`.
#[test]
fn test_class_field_edits_match_clean() {
    let main = r#"
from foo import C
def f(c: C) -> int:
    return c.x
y = C().z
w = C.x
"#;
    for (before, after) in [
        // A field changes type.
        (
            "class C:\n    x: int = 0\n    z: int = 0",
            "class C:\n    x: str = ''\n    z: int = 0",
        ),
        // A field is removed.
        (
            "class C:\n    x: int = 0\n    z: int = 0",
            "class C:\n    z: int = 0",
        ),
        // A field is added.
        (
            "class C:\n    z: int = 0",
            "class C:\n    x: int = 0\n    z: int = 0",
        ),
        // A field stops being initialized on the class.
        (
            "class C:\n    x: int = 0\n    z: int = 0",
            "class C:\n    x: int\n    z: int = 0",
        ),
        // A field becomes a property.
        (
            "class C:\n    x: int = 0\n    z: int = 0",
            "class C:\n    @property\n    def x(self) -> str: ...\n    z: int = 0",
        ),
        // A method changes signature.
        (
            "class C:\n    def x(self) -> int: ...\n    z: int = 0",
            "class C:\n    def x(self, n: int) -> int: ...\n    z: int = 0",
        ),
    ] {
        assert_rebuild_matches_clean(&[("foo", before), ("main", main)], &[("foo", after)], &[]);
    }
}

/// Edits to the bases of a class in `bar`, whose attributes `main` reads through its MRO from
/// `foo`, and an edit to an inherited field in `foo`.
#[test]
fn test_class_base_edits_match_clean() {
    let foo = "class A:\n    x: int = 0\nclass B:\n    x: str = ''\n    y: str = ''";
    let main = r#"
from bar import C
def f(c: C) -> int:
    return c.x
y = C().y
"#;
    for (bar_before, bar_after) in [
        (
            "from foo import A, B\nclass C(A): ...",
            "from foo import A, B\nclass C(B): ...",
        ),
        (
            "from foo import A, B\nclass C(A, B): ...",
            "from foo import A, B\nclass C(B, A): ...",
        ),
        (
            "from foo import A, B\nclass C(A, B): ...",
            "from foo import A, B\nclass C: ...",
        ),
    ] {
        assert_rebuild_matches_clean(
            &[("foo", foo), ("bar", bar_before), ("main", main)],
            &[("bar", bar_after)],
            &[],
        );
    }
    assert_rebuild_matches_clean(
        &[
            ("foo", foo),
            ("bar", "from foo import A, B\nclass C(A, B): ..."),
            ("main", main),
        ],
        &[(
            "foo",
            "class A:\n    x: bytes = b''\nclass B:\n    x: str = ''\n    y: str = ''",
        )],
        &[],
    );
}

/// Edits to what `ClassMetadata` records about a class in `foo`, which `main` uses.
#[test]
fn test_class_metadata_edits_match_clean() {
    for (before, after, main) in [
        // A dataclass becomes frozen.
        (
            "from dataclasses import dataclass\n@dataclass\nclass D:\n    x: int",
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass D:\n    x: int",
            "from foo import D\nd = D(1)\nd.x = 2",
        ),
        // A class becomes final.
        (
            "from typing import final\nclass D: ...",
            "from typing import final\n@final\nclass D: ...",
            "from foo import D\nclass E(D): ...",
        ),
        // A class stops being a protocol.
        (
            "from typing import Protocol\nclass D(Protocol):\n    def m(self) -> int: ...",
            "from typing import Protocol\nclass D:\n    def m(self) -> int: ...",
            "from foo import D\nclass Impl:\n    def m(self) -> int: return 0\nd: D = Impl()",
        ),
        // An enum gains a member.
        (
            "from enum import Enum\nclass D(Enum):\n    A = 1",
            "from enum import Enum\nclass D(Enum):\n    A = 1\n    B = 2",
            "from foo import D\nx = D.B",
        ),
        // A TypedDict stops being total.
        (
            "from typing import TypedDict\nclass D(TypedDict):\n    a: int",
            "from typing import TypedDict\nclass D(TypedDict, total=False):\n    a: int",
            "from foo import D\nd: D = {}",
        ),
        // A class loses its metaclass.
        (
            "class M(type):\n    def hello(cls) -> int: ...\nclass D(metaclass=M): ...",
            "class M(type):\n    def hello(cls) -> int: ...\nclass D: ...",
            "from foo import D\nx = D.hello()",
        ),
        // A named tuple gains a field.
        (
            "from typing import NamedTuple\nclass D(NamedTuple):\n    a: int",
            "from typing import NamedTuple\nclass D(NamedTuple):\n    a: int\n    b: str",
            "from foo import D\nd = D(1)",
        ),
    ] {
        assert_rebuild_matches_clean(&[("foo", before), ("main", main)], &[("foo", after)], &[]);
    }
}

/// Edits that change which class a definition index names, and the type parameters of a
/// legacy generic class.
#[test]
fn test_class_definition_edits_match_clean() {
    let main = r#"
from foo import A, B
def f(a: A, b: B) -> int:
    return a.x + b.y
"#;
    for (before, after) in [
        // A class is inserted above `A`, so `A` and `B` get new indices.
        (
            "class A:\n    x: int = 0\nclass B:\n    y: int = 0",
            "class Z:\n    x: str = ''\nclass A:\n    x: int = 0\nclass B:\n    y: int = 0",
        ),
        // `A` and `B` swap places.
        (
            "class A:\n    x: int = 0\nclass B:\n    y: int = 0",
            "class B:\n    y: int = 0\nclass A:\n    x: int = 0",
        ),
        // `A` is renamed.
        (
            "class A:\n    x: int = 0\nclass B:\n    y: int = 0",
            "class A2:\n    x: int = 0\nclass B:\n    y: int = 0",
        ),
    ] {
        assert_rebuild_matches_clean(&[("foo", before), ("main", main)], &[("foo", after)], &[]);
    }
    assert_rebuild_matches_clean(
        &[
            (
                "foo",
                "from typing import Generic, TypeVar\nT = TypeVar('T')\nclass G(Generic[T]):\n    x: T",
            ),
            ("main", "from foo import G\ng: G[int] = G()\ny = g.x"),
        ],
        &[("foo", "class G:\n    x: str = ''")],
        &[],
    );
}

/// A dataclass field added in a base class in `foo` changes the `__init__` of a dataclass in
/// `bar` that `main` calls.
#[test]
fn test_inherited_dataclass_field_edit_matches_clean() {
    assert_rebuild_matches_clean(
        &[
            (
                "foo",
                "from dataclasses import dataclass\n@dataclass\nclass Base:\n    a: int",
            ),
            (
                "bar",
                "from dataclasses import dataclass\nfrom foo import Base\n@dataclass\nclass Child(Base):\n    b: str",
            ),
            ("main", "from bar import Child\nc = Child(1, '')"),
        ],
        &[(
            "foo",
            "from dataclasses import dataclass\n@dataclass\nclass Base:\n    a: int\n    c: int",
        )],
        &[],
    );
}

/// A model in `bar` binds a custom manager from `foo`, which is in an import cycle with `bar`, so
/// the bound manager carries an effective base, and `foo` reads it back. The edit removes the
/// manager's `Manager` base. `foo` is rebuilt first and reads the bound manager that `bar` computed
/// against the old `foo`, whose effective base names a class that is no longer a base of the
/// manager. The rebuild must not panic, and at every thread count it must match a clean check.
#[test]
fn test_django_manager_loses_manager_base_through_cycle() {
    let foo = |bases: &str| {
        format!(
            "from django.db import models\nfrom bar import Article\nclass ArticleManager{bases}:\n    def published(self): ...\nmanager = Article.objects\nrows = manager.all()\npublished = manager.published()\n"
        )
    };
    let bar = "from django.db import models\nfrom foo import ArticleManager\nclass Article(models.Model):\n    objects = ArticleManager()\n";
    let site_package_path = [PathBuf::from(env::var("DJANGO_TEST_PATH").unwrap())];
    for threads in [1, 3, 8] {
        let rebuild = rebuild_against_clean(
            &[("foo", &foo("(models.Manager)")), ("bar", bar)],
            &[("foo", &foo(""))],
            ThreadCount::NumThreads(NonZeroUsize::new(threads).unwrap()),
            &site_package_path,
        );
        assert_eq!(
            rebuild.stale(),
            SmallMap::new(),
            "stale modules after the recheck at {threads} threads"
        );
    }
}
