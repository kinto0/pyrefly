/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `functools.partial` edge cases beyond the core `partial.rs` bank, plus the closed
//! issue #149 regression. Divergences are `bug=`-marked with the correct behavior recorded inline
//! as `# WANT: ...`.

use crate::functools_testcase;

// Regression: https://github.com/facebook/pyrefly/issues/149 (closed — pyrefly already passes)
functools_testcase!(
    test_partial_keyword_bind_callable_arg,
    r#"
from __future__ import annotations
from functools import partial
from typing import Callable, Match
def bar(a: Match[str], b: int) -> str: return f'{a}{b}'
def zoo(a: Callable[[Match[str]], str]) -> None: return None
zoo(partial(bar, b=99))
"#,
);

functools_testcase!(
    test_partial_edge_construct_too_many_bound,
    r#"
from typing import reveal_type
import functools
def target(a: int, b: str, c: float) -> bytes: return b""
p = functools.partial(target, 1, "x", 2.0, 99)  # E: Expected 3 positional arguments, got 4 in function `target`
reveal_type(p)  # E: revealed type: partial[bytes]
r = p()
reveal_type(r)  # E: revealed type: bytes
"#,
);

functools_testcase!(
    test_partial_edge_construct_duplicate_bound_kw,
    r#"
from typing import reveal_type
import functools
def target(a: int, b: int) -> int: return 0
p = functools.partial(target, 1, a=2)  # E: Multiple values for argument `a` in function `target`
reveal_type(p)  # E: revealed type: partial[int]
"#,
);

functools_testcase!(
    test_partial_edge_construct_bound_kwonly_wrong_type,
    r#"
from typing import reveal_type
import functools
def kwonly(a: int, *, b: str) -> int: return 0
p = functools.partial(kwonly, 1, b=5)  # E: Argument `Literal[5]` is not assignable to parameter `b` with type `str` in function `kwonly`
reveal_type(p)  # E: revealed type: (*, b: str = ...) -> int
"#,
);

functools_testcase!(
    test_partial_edge_call_missing_remaining,
    r#"
from typing import reveal_type
import functools
def f(a: int, b: str) -> bytes: return b""
p = functools.partial(f, 1)
reveal_type(p)  # E: revealed type: (b: str) -> bytes
p()  # E: Missing argument `b`
"#,
);

functools_testcase!(
    bug = "partial over a bound method drops remaining-arg type checking: p(2) passes int where b: str is expected, but pyrefly emits no error",
    test_partial_edge_target_bound_method_badcall,
    r#"
from typing import reveal_type
import functools
class C:
    def m(self, a: int, b: str) -> float: return 0.0
p = functools.partial(C().m, 1)
reveal_type(p)  # E: revealed type: partial[float]
p(2)  # WANT: Argument 1 to "m" has incompatible type "int"; expected "str"
"#,
);

functools_testcase!(
    test_partial_edge_target_typed_lambda_badcall,
    r#"
import functools
from typing import Callable, reveal_type
g: Callable[[int, str], bytes] = lambda a, b: b""
p = functools.partial(g, 1)
reveal_type(p)  # E: revealed type: (str) -> bytes
p(2)  # E: Argument `Literal[2]` is not assignable to parameter with type `str`
"#,
);

functools_testcase!(
    test_partial_edge_positional_only_marker,
    r#"
from typing import reveal_type
import functools
def g(a: int, b: int, /) -> bytes: return b""
p = functools.partial(g, 1)
reveal_type(p)  # E: revealed type: (b: int, /) -> bytes
p(b=2)  # E: Expected argument `b` to be positional
g(1, b=2)  # E: Expected argument `b` to be positional in function `g`
"#,
);

functools_testcase!(
    test_partial_edge_keyword_only_marker_positional,
    r#"
import functools
from typing import assert_type
def k(a: int, *, b: str) -> bytes: return b""
p = functools.partial(k)
assert_type(p(1, b="x"), bytes)
p(1, "x")  # E: Expected argument `b` to be passed by name
"#,
);

functools_testcase!(
    bug = "nested partial loses precision: partial(partial, foo) returns Unknown, so the inner call's return type and arg-type checking are both lost",
    test_partial_edge_nested_partial_wrongtype,
    r#"
from typing import reveal_type
import functools
def foo(x: int) -> int: return x
p = functools.partial(functools.partial, foo)
# WANT: revealed type: int
reveal_type(p()(1))  # E: revealed type: Unknown
p()("no")  # WANT: Argument "no" to "foo" has incompatible type "str"; expected "int"
"#,
);

// ===== Inheritance dimension =====

// The residual of a partial over a function with a base-class parameter must accept a subclass
// instance (Liskov) and reject an unrelated type. Binding one positional leaves the base-class
// parameter in the residual; subtype substitution must still hold there.
functools_testcase!(
    test_partial_edge_base_param_accepts_subclass,
    r#"
from typing import reveal_type
import functools
class Base: ...
class Sub(Base): ...
class Other: ...
def sink(tag: int, node: Base) -> bytes: return b""
p = functools.partial(sink, 1)
reveal_type(p)  # E: revealed type: (node: Base) -> bytes
p(Sub())
p(Base())
p(Other())  # E: Argument `Other` is not assignable to parameter `node` with type `Base`
"#,
);
// A partial over a bound method reached through a subclass that overrides it currently defers to
// the stub (bound-method targets are deferred), so no residual arg-checking happens.
functools_testcase!(
    bug = "partial over an overridden bound method defers to the stub, so the residual's remaining arg is not type-checked",
    test_partial_edge_overridden_bound_method,
    r#"
from typing import reveal_type
import functools
class Base:
    def act(self, a: int, b: str) -> float: return 0.0
class Sub(Base):
    def act(self, a: int, b: str) -> float: return 1.0
p = functools.partial(Sub().act, 1)
reveal_type(p)  # E: revealed type: partial[float]
p(2)  # WANT: Argument `Literal[2]` is not assignable to parameter `b` with type `str`
"#,
);
