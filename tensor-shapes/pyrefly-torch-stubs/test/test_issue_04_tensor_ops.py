# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import SymIntVar

if TYPE_CHECKING:
    from shape_extensions import SymInt
    from torch import Tensor


def test_view[B: SymIntVar, T: SymIntVar, D: SymIntVar, NHead: SymIntVar](
    x: Tensor[[B, T, D]],
    bsz: SymInt[B],
    seqlen: SymInt[T],
    n_head: SymInt[NHead],
    head_dim: SymInt[D // NHead],
) -> None:
    # Test view operation
    result = x.view(bsz, seqlen, n_head, head_dim)
    assert_type(result, Tensor[[B, T, NHead, (D // NHead)]])


def test_transpose[B: SymIntVar, T: SymIntVar, NHead: SymIntVar, HeadDim: SymIntVar](
    q: Tensor[[B, T, NHead, HeadDim]],
) -> None:
    # Test transpose operation
    result = q.transpose(1, 2)
    assert_type(result, Tensor[[B, NHead, T, HeadDim]])


def test_split[
    B: SymIntVar,
    T: SymIntVar,
    D: SymIntVar,
    NLocalHeads: SymIntVar,
    NHead: SymIntVar,
](
    x: Tensor[[B, T, (NHead + 2 * NLocalHeads) * (D // NHead)]],
    dim: SymInt[D],
    kv_size: SymInt[NLocalHeads * (D // NHead)],
) -> None:
    # Test split with tuple (required for meta-shape inference)
    q, k, v = x.split((dim, kv_size, kv_size), dim=-1)
    assert_type(q, Tensor[[B, T, D]])
    assert_type(k, Tensor[[B, T, (NLocalHeads * (D // NHead))]])


def test_flatten[B: SymIntVar, T: SymIntVar, NHeads: SymIntVar, HeadDim: SymIntVar](
    x: Tensor[[B, T, NHeads, HeadDim // 2, 2]],
) -> None:
    # Test flatten operation
    result = x.flatten(3)
    # Result is Tensor[[B, T, NHeads, ((HeadDim // 2) * 2)]]
    # Note: algebraic equivalence to HeadDim is Issue 7
    assert_type(result, Tensor[[B, T, NHeads, ((HeadDim // 2) * 2)]])


def test_stack[SeqLen: SymIntVar, HeadDim: SymIntVar](
    real: Tensor[[SeqLen, HeadDim // 2]],
    imag: Tensor[[SeqLen, HeadDim // 2]],
) -> None:
    # Test stack with tuple (required for meta-shape inference)
    result = torch.stack((real, imag), dim=-1)
    assert_type(result, Tensor[[SeqLen, HeadDim // 2, 2]])
