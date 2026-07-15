# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test type variable unification in SymInt expressions"""

from typing import assert_type, TYPE_CHECKING

from shape_extensions import SymIntVar

if TYPE_CHECKING:
    from shape_extensions import SymInt


# Test 1: Top-level type var unification
# When passing SymInt[A * B] to a function expecting SymInt[X],
# X should be unified with A * B
def identity_symint[X: SymIntVar](x: SymInt[X]) -> SymInt[X]:
    return x


def test_top_level_unification[A: SymIntVar, B: SymIntVar](a: SymInt[A], b: SymInt[B]):
    expr = a * b  # SymInt[A * B]
    assert_type(expr, SymInt[A * B])
    result = identity_symint(expr)
    assert_type(result, SymInt[A * B])
    # X should be unified with A * B, so result should be SymInt[A * B]
    assert_type(result, SymInt[A * B])


# Test 2: Nested type var without prior binding
# When passing SymInt[(A * B) // 2] to a function expecting SymInt[X // 2],
# X cannot be inferred from a nested position - this is an error
def half_symint[X: SymIntVar](x: SymInt[X // 2]) -> SymInt[X]:
    return x * 2  # type: ignore


def test_nested_unification_fails[A: SymIntVar, B: SymIntVar](
    a: SymInt[A], b: SymInt[B]
):
    expr = (a * b) // 2  # SymInt[(A * B) // 2]
    # X is in a nested position and cannot be inferred.
    # E: Type variable cannot be inferred from a nested position
    half_symint(expr)


# Test 3: Nested type var with prior binding
# If X is bound from the first argument, then the second argument can use X in a nested position
def two_args[X: SymIntVar](first: SymInt[X], second: SymInt[X // 2]) -> SymInt[X]:
    return first


def test_nested_with_prior_binding[N: SymIntVar](n: SymInt[N]):
    half_n = n // 2  # SymInt[N // 2]
    # First arg binds X = N, second arg checks N // 2 = N // 2.
    result = two_args(n, half_n)
    assert_type(result, SymInt[N])


# Test 4: Nested type var with prior binding - complex expression
# X is bound to A + A from first arg, second arg uses X // 2 = (A + A) // 2 = A
def test_nested_with_simplification[A: SymIntVar](a: SymInt[A]):
    double_a = a + a  # SymInt[A + A]
    # X = A + A from first arg
    # Second arg: X // 2 = (A + A) // 2 = A (after simplification)
    result = two_args(double_a, a)
    assert_type(result, SymInt[A + A])
