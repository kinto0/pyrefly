# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape


def test_exponential_and_log_preserve_shape() -> None:
    a = np.full(4, 2.0)
    b = np.ones((3, 4))

    assert_shape(np.exp(a), (4,))
    assert_shape(np.log(a), (4,))
    assert_shape(np.log2(a), (4,))
    assert_shape(np.log10(a), (4,))
    assert_shape(np.sqrt(b), (3, 4))


def test_trig_preserves_shape() -> None:
    angles = np.ones((2, 3))

    assert_shape(np.sin(angles), (2, 3))
    assert_shape(np.cos(angles), (2, 3))
    assert_shape(np.tan(angles), (2, 3))
    assert_shape(np.arcsin(np.full((2, 3), 0.5)), (2, 3))


def test_rounding_preserves_shape() -> None:
    a = np.full(5, -1.7)

    assert_shape(np.floor(a), (5,))
    assert_shape(np.ceil(a), (5,))
    assert_shape(np.round(a), (5,))
    assert_shape(np.trunc(a), (5,))
