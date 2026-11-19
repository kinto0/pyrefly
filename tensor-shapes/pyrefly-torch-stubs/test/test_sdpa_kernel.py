# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Upstream: https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/attention/__init__.py

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

from shape_extensions import assert_raises
from torch.nn.attention import (
    _cur_sdpa_kernel_backends,
    sdpa_kernel,
    SDPBackend,
    WARN_FOR_UNFUSED_KERNELS,
)


def test_sdpa_kernel_context() -> None:
    with sdpa_kernel(SDPBackend.MATH) as state:
        assert state == {}
        assert _cur_sdpa_kernel_backends() == [SDPBackend.MATH]

    with sdpa_kernel([SDPBackend.MATH], set_priority=True) as state:
        assert state == {}
        assert _cur_sdpa_kernel_backends(with_priority=True) == [SDPBackend.MATH]

    with assert_raises(AssertionError):
        # E: Argument `Literal['math']` is not assignable to parameter `backends`
        with sdpa_kernel("math"):
            pass


if TYPE_CHECKING:
    assert_type(WARN_FOR_UNFUSED_KERNELS, bool)
    assert_type(_cur_sdpa_kernel_backends(), list[SDPBackend])

    with sdpa_kernel(SDPBackend.MATH) as state:
        assert_type(state, dict[object, object])
