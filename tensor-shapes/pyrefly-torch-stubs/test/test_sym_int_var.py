# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test shape_extensions.SymIntVar for tensor shape dimensions.

shape_extensions.SymIntVar marks symbolic integer dimensions in pyrefly.
This test verifies that:
1. SymIntVar("N") works for shape annotations
2. SymIntTuple carriers work for variadic shapes
3. Generic works with shape_extensions.SymIntVar for class-level type parameters
4. Shape arithmetic (N+1, N*2) works in annotations
"""

from typing import assert_type, Generic, TYPE_CHECKING

from shape_extensions import Elements, SymIntTuple, SymIntVar

if TYPE_CHECKING:
    from torch import Tensor

N = SymIntVar("N")
M = SymIntVar("M")


# ============================================================================
# Basic SymIntVar usage in function signatures
# ============================================================================


def test_symintvar_identity(x: Tensor[[N, M]]) -> Tensor[[N, M]]:
    """SymIntVar in input and output: same shape"""
    return x


def test_symintvar_single(x: Tensor[[N]]) -> Tensor[[N]]:
    """Single SymIntVar dimension"""
    return x


def test_symintvar_inference():
    """SymIntVar binds to concrete dims via inference"""
    import torch

    t: Tensor[[3, 4]] = torch.randn(3, 4)
    result = test_symintvar_identity(t)
    assert_type(result, Tensor[[3, 4]])


# ============================================================================
# SymIntVar with arithmetic in shapes
# ============================================================================


def test_symintvar_add(x: Tensor[[N, M]]) -> Tensor[[N + 1, M]]:
    """N + 1 in return type"""
    return x  # type: ignore[bad-return]


def test_symintvar_mul(x: Tensor[[N, M]]) -> Tensor[[N * 2, M]]:
    """N * 2 in return type"""
    return x  # type: ignore[bad-return]


def test_symintvar_sub(x: Tensor[[N, M]]) -> Tensor[[N - 1, M]]:
    """N - 1 in return type"""
    return x  # type: ignore[bad-return]


def test_symintvar_two_vars(x: Tensor[[N, M]]) -> Tensor[[N + M, 3]]:
    """N + M in return type"""
    return x  # type: ignore[bad-return]


# ============================================================================
# Generic with SymIntVar for class-level type parameters
# ============================================================================


class SameShapeLayer(Generic[N]):
    """Class generic over single SymIntVar"""

    def forward(self, x: Tensor[[N]]) -> Tensor[[N]]:
        return x


def test_class_generic():
    """Generic class with method call — N binds from input"""
    layer = SameShapeLayer()
    import torch

    x: Tensor[[3]] = torch.randn(3)
    result = layer.forward(x)
    assert_type(result, Tensor[[3]])


# ============================================================================
# SymIntTuple carrier in function signatures
# ============================================================================


def test_syminttuple_identity[Ns: SymIntTuple](x: Tensor[Ns]) -> Tensor[Ns]:
    """SymIntTuple carrier preserves shape"""
    return x


def test_syminttuple_inference():
    """SymIntTuple carrier binds to concrete dims via inference"""
    import torch

    t: Tensor[[10, 20]] = torch.randn(10, 20)
    result = test_syminttuple_identity(t)
    assert_type(result, Tensor[[10, 20]])


def test_syminttuple_with_fixed_dim[Ns: SymIntTuple, N: SymIntVar](
    x: Tensor[[*Elements[Ns], N]],
) -> Tensor[[*Elements[Ns], N]]:
    """SymIntTuple carrier mixed with SymIntVar"""
    return x


def test_syminttuple_with_arithmetic[Ns: SymIntTuple, N: SymIntVar](
    x: Tensor[[*Elements[Ns], N]],
) -> Tensor[[*Elements[Ns], N + 1]]:
    """SymIntTuple carrier with SymIntVar arithmetic"""
    return x  # type: ignore[bad-return]


# ============================================================================
# SymIntTuple carrier with Generic for class-level shape parameters
# ============================================================================


class VariadicLayer:
    """Layer with a generic SymIntTuple carrier method"""

    def forward[Shape: SymIntTuple](self, x: Tensor[Shape]) -> Tensor[Shape]:
        return x


def test_class_syminttuple_carrier():
    """Generic class with SymIntTuple carrier — shape preserved"""
    layer = VariadicLayer()
    import torch

    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = layer.forward(x)
    assert_type(result, Tensor[[2, 3, 4]])
