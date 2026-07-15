# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import overload

from shape_extensions import SymInt, SymIntVar

from .. import dtype, float64, ndarray

@overload
def randn[N: SymIntVar](d0: SymInt[N], /) -> ndarray[[N], dtype[float64]]: ...
@overload
def randn[N: SymIntVar, M: SymIntVar](
    d0: SymInt[N], d1: SymInt[M], /
) -> ndarray[[N, M], dtype[float64]]: ...
