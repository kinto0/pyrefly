# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test SymInt type behavior with Any and implicit parameters.

This test documents that int literals can be assigned to SymInt types,
including bare SymInt, SymInt[Any], or passed to functions with type parameters.
"""

from typing import Any, assert_type, TYPE_CHECKING

from shape_extensions import SymIntVar

if TYPE_CHECKING:
    from shape_extensions import SymInt

symint_implicit_any: SymInt = 4
assert_type(symint_implicit_any, SymInt)
symint_explicit_any: SymInt[Any] = 4
assert_type(symint_explicit_any, SymInt[Any])


def accept_and_return_symint[N: SymIntVar](s: SymInt[N]) -> SymInt[N]:
    return s


def test_accept_and_return_symint():
    s = accept_and_return_symint(4)
    assert_type(s, SymInt[4])
    n: int = 4
    s_n = accept_and_return_symint(n)
    assert_type(s_n, SymInt)
    s_implicit_any = accept_and_return_symint(symint_implicit_any)
    assert_type(s_implicit_any, SymInt)
    s_explicit_any = accept_and_return_symint(symint_explicit_any)
    assert_type(s_explicit_any, SymInt[Any])
