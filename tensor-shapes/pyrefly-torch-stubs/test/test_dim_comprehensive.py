# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Comprehensive tests for SymInt type: parsing, arithmetic, and subtyping.

This file tests the SymInt/SymInt type system independent of Tensor shapes.
"""

from typing import Any, assert_type, Literal, TYPE_CHECKING

from shape_extensions import SymVar


if TYPE_CHECKING:
    from shape_extensions import SymInt


# ============================================================================
# Basic SymInt Parsing
# ============================================================================


def test_dim_literal_parsing() -> None:
    """SymInt[n] with literal integers"""
    x: SymInt[3] = 3
    assert_type(x, SymInt[3])

    y: SymInt[42] = 42
    assert_type(y, SymInt[42])


def test_dim_typevar_parsing[N: SymVar](n: SymInt[N]) -> None:
    """SymInt[N] with type variable"""
    assert_type(n, SymInt[N])


def test_dim_expression_parsing[N: SymVar](n: SymInt[N]) -> None:
    """SymInt[N+1] with expression"""
    x = n + 1
    assert_type(x, SymInt[N + 1])


# ============================================================================
# SymInt Arithmetic - All Operators
# ============================================================================


def test_dim_add_literal[N: SymVar](n: SymInt[N]) -> None:
    """Addition: SymInt + literal"""
    result = n + 2
    assert_type(result, SymInt[N + 2])


def test_dim_add_dim[A: SymVar, B: SymVar](a: SymInt[A], b: SymInt[B]) -> None:
    """Addition: SymInt + SymInt"""
    result = a + b
    assert_type(result, SymInt[A + B])


def test_dim_sub_literal[N: SymVar](n: SymInt[N]) -> None:
    """Subtraction: SymInt - literal"""
    result = n - 1
    assert_type(result, SymInt[N - 1])


def test_dim_sub_dim[A: SymVar, B: SymVar](a: SymInt[A], b: SymInt[B]) -> None:
    """Subtraction: SymInt - SymInt"""
    result = a - b
    assert_type(result, SymInt[A - B])


def test_dim_mul_literal[N: SymVar](n: SymInt[N]) -> None:
    """Multiplication: SymInt * literal"""
    result = n * 2
    assert_type(result, SymInt[N * 2])


def test_dim_mul_dim[A: SymVar, B: SymVar](a: SymInt[A], b: SymInt[B]) -> None:
    """Multiplication: SymInt * SymInt"""
    result = a * b
    assert_type(result, SymInt[A * B])


def test_dim_floordiv_literal[N: SymVar](n: SymInt[N]) -> None:
    """Floor division: SymInt // literal"""
    result = n // 2
    assert_type(result, SymInt[N // 2])


def test_dim_floordiv_dim[A: SymVar, B: SymVar](a: SymInt[A], b: SymInt[B]) -> None:
    """Floor division: SymInt // SymInt"""
    result = a // b
    assert_type(result, SymInt[A // B])


def test_dim_complex_expression[N: SymVar](n: SymInt[N]) -> None:
    """Complex arithmetic expression"""
    # (N + N) // 2 expression - simplification happens during subtyping
    double = n + n
    half_double = double // 2
    assert_type(half_double, SymInt[(N + N) // 2])


def test_dim_nested_expression[A: SymVar, B: SymVar](
    a: SymInt[A], b: SymInt[B]
) -> None:
    """Nested arithmetic"""
    result = (a + b) * 2
    assert_type(result, SymInt[(A + B) * 2])


# ============================================================================
# SymInt Subtyping - Using def f(x: T1) -> T2: return x pattern
# ============================================================================


def dim_to_dim_same[N: SymVar](x: SymInt[N]) -> SymInt[N]:
    """SymInt[N] <: SymInt[N]"""
    return x


def dim_literal_to_same(x: SymInt[3]) -> SymInt[3]:
    """SymInt[3] <: SymInt[3]"""
    return x


def dim_to_int[N: SymVar](x: SymInt[N]) -> int:
    """SymInt[N] <: int - SymInt values are subtypes of int"""
    return x


def dim_literal_to_int(x: SymInt[5]) -> int:
    """SymInt[5] <: int"""
    return x


def int_to_dim_any(x: int) -> SymInt[Any]:
    """int <: SymInt[Any] - plain int can be used where SymInt expected"""
    return x


def literal_to_dim(x: Literal[7]) -> SymInt[7]:
    """Literal[7] <: SymInt[7] - literal ints are subtypes of SymInt"""
    return x


def dim_expression_subtype[N: SymVar](x: SymInt[N + N]) -> SymInt[N * 2]:
    """SymInt[N + N] <: SymInt[N * 2] - after simplification these are equal"""
    return x


def dim_double_half[N: SymVar](x: SymInt[(N + N) // 2]) -> SymInt[N]:
    """SymInt[(N + N) // 2] <: SymInt[N] - simplifies to N"""
    return x


# ============================================================================
# SymInt Type Variable Binding
# ============================================================================


def identity_dim[X: SymVar](x: SymInt[X]) -> SymInt[X]:
    """Identity function for SymInt - X binds to the dimension"""
    return x


def test_dim_binding_literal() -> None:
    """Binding type var to literal"""
    result = identity_dim(4)
    assert_type(result, SymInt[4])


def test_dim_binding_typevar[N: SymVar](n: SymInt[N]) -> None:
    """Binding type var to another type var"""
    result = identity_dim(n)
    assert_type(result, SymInt[N])


def test_dim_binding_expression[A: SymVar, B: SymVar](
    a: SymInt[A], b: SymInt[B]
) -> None:
    """Binding type var to expression"""
    expr = a * b
    result = identity_dim(expr)
    assert_type(result, SymInt[A * B])


# ============================================================================
# SymInt Unification - Type Var in Result Position
# ============================================================================


def double_dim[X: SymVar](x: SymInt[X]) -> SymInt[X * 2]:
    """Return doubled dimension"""
    return x * 2


def test_double_dim_literal() -> None:
    """Double a literal dimension"""
    result = double_dim(5)
    assert_type(result, SymInt[10])


def test_double_dim_typevar[N: SymVar](n: SymInt[N]) -> None:
    """Double a symbolic dimension"""
    result = double_dim(n)
    assert_type(result, SymInt[N * 2])


def half_dim[X: SymVar](x: SymInt[X]) -> SymInt[X // 2]:
    """Return halved dimension"""
    return x // 2


def test_half_dim_literal() -> None:
    """Half a literal dimension"""
    result = half_dim(10)
    assert_type(result, SymInt[5])


def test_half_dim_typevar[N: SymVar](n: SymInt[N]) -> None:
    """Half a symbolic dimension"""
    result = half_dim(n)
    assert_type(result, SymInt[N // 2])


# ============================================================================
# Multi-Argument SymInt Functions
# ============================================================================


def add_dims[A: SymVar, B: SymVar](a: SymInt[A], b: SymInt[B]) -> SymInt[A + B]:
    """Add two dimensions"""
    return a + b


def test_add_dims_literals() -> None:
    """Add literal dimensions"""
    result = add_dims(3, 4)
    assert_type(result, SymInt[7])


def test_add_dims_typevars[X: SymVar, Y: SymVar](x: SymInt[X], y: SymInt[Y]) -> None:
    """Add symbolic dimensions"""
    result = add_dims(x, y)
    assert_type(result, SymInt[X + Y])


def test_add_dims_mixed[N: SymVar](n: SymInt[N]) -> None:
    """Add symbolic and literal"""
    result = add_dims(n, 5)
    assert_type(result, SymInt[N + 5])


# ============================================================================
# SymInt with Prior Binding
# ============================================================================


def two_dims_same_var[X: SymVar](first: SymInt[X], second: SymInt[X]) -> SymInt[X]:
    """Both arguments must have same dimension"""
    return first


def test_two_dims_same_literal() -> None:
    """Same literal for both"""
    result = two_dims_same_var(5, 5)
    assert_type(result, SymInt[5])


def test_two_dims_same_typevar[N: SymVar](n: SymInt[N]) -> None:
    """Same typevar for both"""
    result = two_dims_same_var(n, n)
    assert_type(result, SymInt[N])


def with_derived[X: SymVar](first: SymInt[X], second: SymInt[X // 2]) -> SymInt[X]:
    """Second arg uses derived dimension"""
    return first


def test_derived_binding[N: SymVar](n: SymInt[N]) -> None:
    """Bind X from first, check X // 2 in second"""
    half = n // 2
    result = with_derived(n, half)
    assert_type(result, SymInt[N])


def test_derived_with_simplification[A: SymVar](a: SymInt[A]) -> None:
    """Bind X = A + A, check X // 2 = A"""
    double_a = a + a  # SymInt[A + A]
    # X = A + A, X // 2 = (A + A) // 2 = A
    result = with_derived(double_a, a)
    assert_type(result, SymInt[A + A])
