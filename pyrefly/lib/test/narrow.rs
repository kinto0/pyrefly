/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_is,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is None:
        assert_type(x, None)
    assert_type(x, str | None)
    "#,
);

testcase!(
    bug = "PyTorch TODO: implement attribute narrowing",
    test_attr_refine,
    r#"
from typing import Any, Optional, reveal_type

class N:
    type: Optional[Any]

def add_inference_rule(n: N):
    reveal_type(n) # E: revealed type: N
    reveal_type(n.type) # E: revealed type: Any | None
    n.type = 3
    reveal_type(n.type + 3) # E: revealed type: int | Unknown # E: `+` is not supported between `None` and `Literal[3]`
"#,
);

testcase!(
    test_truthy_falsy,
    r#"
from typing import assert_type, Literal
def f(x: str | None, y: bool):
    if x:
        assert_type(x, str)
    if y:
        assert_type(y, Literal[True])
    else:
        assert_type(y, Literal[False])
    "#,
);

testcase!(
    test_eq,
    r#"
from typing import assert_type
def f(x: str | None):
    if x == None:
        assert_type(x, None)
    "#,
);

testcase!(
    test_neq,
    r#"
from typing import assert_type
def f(x: str | None):
    if x != None:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_not,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is not None:
        assert_type(x, str)
    "#,
);

testcase!(
    test_if_else,
    r#"
from typing import assert_type
def f(x: str | None):
    if x is None:
        assert_type(x, None)
    else:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_subtype,
    r#"
from typing import assert_type
class A: pass
class B(A): pass
def f(x: type[A]):
    if x is B:
        assert_type(x, type[B])
    "#,
);

testcase!(
    test_is_never,
    r#"
from typing import assert_type, Never
def f(x: str):
    if x is None:
        assert_type(x, Never)
    "#,
);

testcase!(
    test_is_not_bool_literal,
    r#"
from typing import assert_type, Literal, Never
def f1(x: bool):
    if x is not True:
        assert_type(x, Literal[False])
def f2(x: Literal[True] | str):
    if x is not True:
        assert_type(x, str)
    "#,
);

testcase!(
    test_is_not_enum_literal,
    r#"
from typing import assert_type, Literal
import enum
class E(enum.Enum):
    X = 1
    Y = 2
def f1(x: Literal[E.X, E.Y]):
    if x is not E.X:
        assert_type(x, Literal[E.Y])
def f2(x: E | int):
    if x is not E.X:
        assert_type(x, Literal[E.Y] | int)
    "#,
);

testcase!(
    test_tri_enum,
    r#"
from typing import assert_type, Literal
import enum
class E(enum.Enum):
    X = 1
    Y = 2
    Z = 3
def f(x: E):
    if x is E.X:
       assert_type(x, Literal[E.X])
    elif x is E.Y:
       assert_type(x, Literal[E.Y])
    else:
       assert_type(x, Literal[E.Z])
    "#,
);

testcase!(
    test_is_classdef,
    r#"
from typing import assert_type
class A: pass
class B: pass
def f1(x: type[A] | type[B]):
    if x is A:
        assert_type(x, type[A])
    else:
        # Note that we cannot narrow to `type[B]` here, as `type` is covariant and `x` may be a
        # subtype of `A`.
        assert_type(x, type[A] | type[B])
    "#,
);

testcase!(
    bug = "`Literal[False] | bool` should collapse to `bool`",
    test_and,
    r#"
from typing import assert_type, Literal, Never
def f(x: bool | None):
    if x is True and x is None:
        assert_type(x, Never)
    else:
        assert_type(x, Literal[False] | bool | None)
    "#,
);

testcase!(
    test_and_multiple_vars,
    r#"
from typing import assert_type, Literal
def f(x: bool | None, y: bool | None):
    if x is True and y is False:
        assert_type(x, Literal[True])
        assert_type(y, Literal[False])
    "#,
);

testcase!(
    test_or,
    r#"
from typing import assert_type, Literal
def f(x: bool | None):
    if x == True or x is None:
        assert_type(x, Literal[True] | None)
    else:
        assert_type(x, Literal[False])
    "#,
);

testcase!(
    test_elif,
    r#"
from typing import assert_type
def f(x: str | None, y: int | None):
    if x is None:
        assert_type(x, None)
        assert_type(y, int | None)
    elif y is None:
        assert_type(x, str)
        assert_type(y, None)
    else:
        assert_type(x, str)
        assert_type(y, int)
    "#,
);

testcase!(
    test_not,
    r#"
from typing import assert_type
def f(x: str | None):
    if not x is None:
        assert_type(x, str)
    else:
        assert_type(x, None)
    "#,
);

testcase!(
    bug = "`Literal[False, True] | bool` should collapse to `bool`",
    test_not_and,
    r#"
from typing import assert_type, Literal
def f(x: bool | None):
    if not (x is True and x is None):
        assert_type(x, Literal[False, True] | bool | None)
    "#,
);

testcase!(
    test_assert,
    r#"
from typing import assert_type
def f(x: str | None):
    assert x is not None
    assert_type(x, str)
    "#,
);

testcase!(
    test_while_else,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    assert_type(x, None)
    x = f()
    assert_type(x, str | None)
else:
    assert_type(x, str)
assert_type(x, str)
    "#,
);

testcase!(
    test_while_break,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    break
assert_type(x, str | None)
    "#,
);

testcase!(
    test_while_break_else,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
while x is None:
    if f():
        break
else:
    assert_type(x, str)
assert_type(x, str | None)
    "#,
);

testcase!(
    bug = "Unwanted EXPECTED error",
    test_while_overwrite,
    r#"
from typing import assert_type, Literal
def f() -> str | None: ...
x = f()
while x is None:  # E: EXPECTED None <: Literal[42] | str
    if f():
        x = 42
        break
assert_type(x, Literal[42] | str)
    "#,
);

testcase!(
    bug = "At narrowing-time it's still a Var instead of a bool",
    test_while_narrow,
    r#"
from typing import assert_type, Literal, reveal_type
def test(x: bool, z: bool):
    while x:
        assert_type(x, bool)  # should be Literal[True]
    while y := z:
        assert_type(y, bool)  # should be Literal[True]
        assert_type(z, bool)  # should be Literal[True]
    "#,
);
testcase!(
    test_nested_function,
    r#"
from typing import assert_type
def foo(x: int | None) -> None:
    def include():
        if x is not None:
            assert_type(x, int)
    "#,
);

testcase!(
    test_multiple_is,
    r#"
from typing import assert_type, Never
def f(x: bool | None, y: bool | None):
    if x is None is None:
        assert_type(x, None)
    if y is None is True:
        assert_type(y, Never)
    "#,
);

testcase!(
    test_class_body,
    r#"
from typing import assert_type
def f() -> str | None: ...
x = f()
class C:
    if x is None:
        assert_type(x, None)
    "#,
);

testcase!(
    test_walrus_target,
    r#"
from typing import assert_type
def f() -> str | None:
    pass
if x := f():
    assert_type(x, str)
    "#,
);

testcase!(
    test_walrus_value,
    r#"
from typing import assert_type
def f(x: int | None):
    if y := x:
        assert_type(x, int)
        assert_type(y, int)
    "#,
);

testcase!(
    test_walrus_comparison,
    r#"
from typing import assert_type
def f() -> str | None:
    pass
if (x := f()) is None:
    assert_type(x, None)
    "#,
);

testcase!(
    test_match_enum_fallback,
    r#"
from typing import assert_type, Literal
from enum import Enum
class E(Enum):
    X = 1
    Y = 2
    Z = 3
def f(e: E):
    match e:
        case E.X:
            assert_type(e, Literal[E.X])
        case E.Y:
            assert_type(e, Literal[E.Y])
        case _:
            assert_type(e, Literal[E.Z])
    "#,
);

testcase!(
    test_match_or,
    r#"
from typing import assert_type, Literal
def f(e: bool | None):
    match e:
        case True | None:
            assert_type(e, Literal[True] | None)
        case _:
            assert_type(e, Literal[False])
    "#,
);

testcase!(
    test_ternary,
    r#"
from typing import assert_type
def f(x: str | None, y: int):
    z = x if x else y
    assert_type(x, str | None)
    assert_type(y, int)
    assert_type(z, str | int)
    "#,
);

testcase!(
    test_is_supertype,
    r#"
from typing import Literal, assert_type
import enum
class E(enum.Enum):
    X = 1
def f(x: Literal[E.X], y: E):
    if x is y:
        assert_type(x, Literal[E.X])
    "#,
);

testcase!(
    test_isinstance,
    r#"
from typing import assert_type
def f(x: str | int):
    if isinstance(x, str):
        assert_type(x, str)
    else:
        assert_type(x, int)
    "#,
);

testcase!(
    test_isinstance_union,
    r#"
from typing import assert_type
def f(x: str | int | None):
    if isinstance(x, str | int):
        assert_type(x, str | int)
    else:
        assert_type(x, None)
    "#,
);

testcase!(
    test_isinstance_tuple,
    r#"
from typing import assert_type
def f(x: str | int | None):
    if isinstance(x, (str, int)):
        assert_type(x, str | int)
    else:
        assert_type(x, None)
    "#,
);

testcase!(
    bug = "issubclass union narrowing is not yet supported",
    test_issubclass_union,
    r#"
from typing import assert_type
def f(x: type[int | str | bool]):
    if issubclass(x, str | int):  # E: Expected class object, got type[int | str]
        assert_type(x, type[str] | type[int])  # E: assert_type(type[bool | int | str], type[int] | type[str])
    else:
        assert_type(x, type[bool])  # E: assert_type(type[bool | int | str], type[bool])
    "#,
);

testcase!(
    bug = "issubclass tuple narrowing is not yet supported",
    test_issubclass_tuple,
    r#"
from typing import assert_type
def f(x: type[int | str | bool]):
    if issubclass(x, (str, int)):  # E: Expected class object, got tuple[type[str], type[int]]
        assert_type(x, type[str] | type[int])  # E: assert_type(type[bool | int | str], type[int] | type[str])
    else:
        assert_type(x, type[bool])  # E: assert_type(type[bool | int | str], type[bool])
    "#,
);

testcase!(
    test_isinstance_alias,
    r#"
from typing import assert_type
X = int
def f(x: str | int):
    if isinstance(x, X):
        assert_type(x, int)
    "#,
);

testcase!(
    test_isinstance_error,
    r#"
from typing import assert_type
def f(x: int | list[int]):
    if isinstance(x, list[int]):  # E: Expected class object
        assert_type(x, int | list[int])
    "#,
);

testcase!(
    test_isinstance_aliased,
    r#"
from typing import assert_type
istype = isinstance
def f(x: int | str):
    if istype(x, int):
        assert_type(x, int)
    "#,
);

testcase!(
    test_guarded_attribute_access_and,
    r#"
class A:
    x: str
class B:
    pass
def f(x: A | B):
    return isinstance(x, A) and x.x
    "#,
);

testcase!(
    test_guarded_attribute_access_or,
    r#"
class A:
    x: str
def f(x: A | None):
    return x is None or x.x
    "#,
);

testcase!(
    test_and_chain_with_walrus,
    r#"
from typing import assert_type, Literal

class A: ...
class B: ...

def test(x: A | B):
    y = isinstance(x, A) and (z := True)
    assert_type(x, A | B)
    assert_type(z, Literal[True])
    "#,
);

testcase!(
    test_typeguard_basic,
    r#"
from typing import TypeGuard, assert_type
class Cat:
    color: str
class Dog:
    pass
def is_black_cat(x: Cat | Dog) -> TypeGuard[Cat]:
    return isinstance(x, Cat) and x.color == "black"
def f(x: Cat | Dog):
    if is_black_cat(x):
        assert_type(x, Cat)
    else:
        assert_type(x, Cat | Dog)
    is_black_cat(1)  # E: Argument `Literal[1]` is not assignable to parameter `x` with type `Cat | Dog` in function `is_black_cat`
    "#,
);

testcase!(
    test_typeis,
    r#"
from typing import TypeIs, assert_type
class Cat:
    color: str
class Dog:
    pass
def is_cat(x: Cat | Dog) -> TypeIs[Cat]:
    return isinstance(x, Cat)
def f(x: Cat | Dog):
    if is_cat(x):
        assert_type(x, Cat)
    else:
        assert_type(x, Dog)
    "#,
);

testcase!(
    test_typeis_union,
    r#"
from typing import TypeIs, assert_type
class A: ...
class B: ...
class C: ...
def is_a_or_b(x: object) -> TypeIs[A | B]:
    return isinstance(x, A) or isinstance(x, B)
def f(x:  A | B | C, y: A | C):
    if is_a_or_b(x):
        assert_type(x, A | B)
    else:
        assert_type(x, C)
    if is_a_or_b(y):
        assert_type(y, A)
    else:
        assert_type(y, C)
    "#,
);

testcase!(
    test_issubclass,
    r#"
from typing import assert_type
class A: ...
class B(A): ...
def f(x: type[B] | type[int]):
    if issubclass(x, A):
        assert_type(x, type[B])
    else:
        assert_type(x, type[int])
    "#,
);

testcase!(
    test_issubclass_error,
    r#"
def f(x: int):
    if issubclass(x, int):  # E: Argument `int` is not assignable to parameter with type `type`
        return True
    "#,
);

testcase!(
    test_typeguard_instance_method,
    r#"
from typing import TypeGuard, assert_type
class C:
    def is_positive_int(self, x: object) -> TypeGuard[int]:
        return isinstance(x, int) and x > 0
def f(c: C, x: int | str):
    if c.is_positive_int(x):
        assert_type(x, int)
    "#,
);

testcase!(
    test_typeguard_generic_function,
    r#"
from typing import TypeGuard, assert_type
def f[T](x: object, y: T, z: T) -> TypeGuard[int]: ...
def f2[T](x: object, y: T) -> TypeGuard[T]: ...
def g(x: int | str):
    if f(x, 0, 0):
        assert_type(x, int)
    if f2(x, ""):
        assert_type(x, str)
    "#,
);

testcase!(
    test_implicit_else,
    r#"
from typing import assert_type
def f(x: int | None):
    if not x:
        return
    assert_type(x, int)
    "#,
);

testcase!(
    test_narrowed_elif_test,
    r#"
def f(x: int | None, y: bool):
    if not x:
        pass
    elif x > 42:
        pass
"#,
);

testcase!(
    test_narrow_comprehension,
    r#"
from typing import assert_type
def f(xs: list[int | None]):
    ys = [x for x in xs if x]
    assert_type(ys, list[int])
"#,
);

// Note: the narrowing code isn't actually what's giving us this behavior,
// it comes from flow-aware type information taking precedence over static
// annotations. But the end result is narrowing behavior.
testcase!(
    test_assignment_and_narrowing,
    r#"
from typing import assert_type, Literal
def foo(x: int | str):
    y: int | str = x
    assert_type(x, int | str)
    assert_type(y, int | str)
    x = 42
    y = 42
    assert_type(x, Literal[42])
    assert_type(y, Literal[42])
    "#,
);

testcase!(
    test_bad_typeguard_return,
    r#"
from typing import TypeGuard
def f(x) -> TypeGuard[str]:
    return "oops"  # E: Returned type `Literal['oops']` is not assignable to expected return type `bool` of type guard functions
def g(x) -> TypeGuard[str]:  # E: Function declared to return `TypeGuard[str]` but is missing an explicit `return`
    pass
    "#,
);

testcase!(
    test_isinstance_any_second,
    r#"
from typing import Any
def f(x: int | str, y: Any):
    if isinstance(x, y):
        pass
    "#,
);

testcase!(
    test_isinstance_any_literally,
    r#"
from typing import Any
def f(x: int | str):
    if isinstance(x, Any): # E: Expected class object, got type[Any]
        pass
    "#,
);

testcase!(
    test_isinstance_any_first,
    r#"
from typing import Any, assert_type
def f(x: Any):
    if isinstance(x, bool):
        assert_type(x, bool)
    else:
        assert_type(x, Any)
"#,
);

testcase!(
    test_unittest_assert,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_true(self) -> None:
        x = foo()
        self.assertTrue(x is not None)
        assert_type(x, int)
    
    def test_false(self) -> None:
        x = foo()
        self.assertFalse(x is None)
        assert_type(x, int)
"#,
);

testcase!(
    test_unittest_assert_none,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_is_none(self) -> None:
        x = foo()
        self.assertIsNone(x)
        assert_type(x, None)
    
    def test_is_not_none(self) -> None:
        x = foo()
        self.assertIsNotNone(x)
        assert_type(x, int)
"#,
);

testcase!(
    test_unittest_assert_isinstance,
    r#"
from typing import assert_type
from unittest import TestCase
def foo() -> int | None: ...
class MyTest(TestCase):
    def test_is_instance(self) -> None:
        x = foo()
        self.assertIsInstance(x, int)
        assert_type(x, int)
    
    def test_is_not_instance(self) -> None:
        x = foo()
        self.assertNotIsInstance(x, int)
        assert_type(x, None)
"#,
);
