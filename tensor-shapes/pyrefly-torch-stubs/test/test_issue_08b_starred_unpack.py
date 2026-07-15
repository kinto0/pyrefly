# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

from shape_extensions import SymIntVar


if TYPE_CHECKING:
    from shape_extensions import SymInt
    from torch import Tensor


def test_starred_unpack[
    B: SymIntVar,
    T: SymIntVar,
    NHeads: SymIntVar,
    HeadDim: SymIntVar,
](
    x: Tensor[[B, T, NHeads, HeadDim]],
) -> None:
    # First check that tuple slicing works
    sizes = x.size()
    assert_type(sizes, tuple[SymInt[B], SymInt[T], SymInt[NHeads], SymInt[HeadDim]])

    sliced = x.size()[:-1]
    assert_type(sliced, tuple[SymInt[B], SymInt[T], SymInt[NHeads]])

    # Starred unpacking now preserves element types
    result = x.float().reshape(*sliced, -1, 2)
    assert_type(result, Tensor[[B, T, NHeads, HeadDim // 2, 2]])
