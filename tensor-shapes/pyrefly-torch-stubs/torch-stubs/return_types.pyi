# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-generic named tuples returned by torch operations.

Mirrors ``torch.return_types``, which torch generates from the output names in
``native_functions.yaml`` as tuple subclasses with a property per field.
"""

from typing import Any

from shape_extensions import IntTuple
from torch import Tensor

class aminmax[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def min(self) -> Tensor[Shape]: ...
    @property
    def max(self) -> Tensor[Shape]: ...

class cummax[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class cummin[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class kthvalue[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class max[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class median[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class min[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class mode[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class sort[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class topk[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def values(self) -> Tensor[Shape]: ...
    @property
    def indices(self) -> Tensor[Shape]: ...

class linalg_cholesky_ex[Result: IntTuple, Info: IntTuple](
    tuple[Tensor[Result], Tensor[Info]]
):
    @property
    def L(self) -> Tensor[Result]: ...
    @property
    def info(self) -> Tensor[Info]: ...

class linalg_inv_ex[Result: IntTuple, Info: IntTuple](
    tuple[Tensor[Result], Tensor[Info]]
):
    @property
    def inverse(self) -> Tensor[Result]: ...
    @property
    def info(self) -> Tensor[Info]: ...

class linalg_ldl_factor[Factor: IntTuple, Pivots: IntTuple](
    tuple[Tensor[Factor], Tensor[Pivots]]
):
    @property
    def LD(self) -> Tensor[Factor]: ...
    @property
    def pivots(self) -> Tensor[Pivots]: ...

class linalg_ldl_factor_ex[Factor: IntTuple, Pivots: IntTuple, Info: IntTuple](
    tuple[Tensor[Factor], Tensor[Pivots], Tensor[Info]]
):
    @property
    def LD(self) -> Tensor[Factor]: ...
    @property
    def pivots(self) -> Tensor[Pivots]: ...
    @property
    def info(self) -> Tensor[Info]: ...

class linalg_lu_factor[Factor: IntTuple, Pivots: IntTuple](
    tuple[Tensor[Factor], Tensor[Pivots]]
):
    @property
    def LU(self) -> Tensor[Factor]: ...
    @property
    def pivots(self) -> Tensor[Pivots]: ...

class linalg_lu_factor_ex[Factor: IntTuple, Pivots: IntTuple, Info: IntTuple](
    tuple[Tensor[Factor], Tensor[Pivots], Tensor[Info]]
):
    @property
    def LU(self) -> Tensor[Factor]: ...
    @property
    def pivots(self) -> Tensor[Pivots]: ...
    @property
    def info(self) -> Tensor[Info]: ...

class linalg_lu[Permutation: IntTuple, Lower: IntTuple, Upper: IntTuple](
    tuple[Tensor[Permutation], Tensor[Lower], Tensor[Upper]]
):
    @property
    def P(self) -> Tensor[Permutation]: ...
    @property
    def L(self) -> Tensor[Lower]: ...
    @property
    def U(self) -> Tensor[Upper]: ...

class linalg_qr[Orthogonal: IntTuple, Triangular: IntTuple](
    tuple[Tensor[Orthogonal], Tensor[Triangular]]
):
    @property
    def Q(self) -> Tensor[Orthogonal]: ...
    @property
    def R(self) -> Tensor[Triangular]: ...

class linalg_svd[Left: IntTuple, Singular: IntTuple, Right: IntTuple](
    tuple[Tensor[Left], Tensor[Singular], Tensor[Right]]
):
    @property
    def U(self) -> Tensor[Left]: ...
    @property
    def S(self) -> Tensor[Singular]: ...
    @property
    def Vh(self) -> Tensor[Right]: ...

class linalg_lstsq[
    Solution: IntTuple,
    Residuals: IntTuple,
    Rank: IntTuple,
    SingularValues: IntTuple,
](tuple[Tensor[Solution], Tensor[Residuals], Tensor[Rank], Tensor[SingularValues]]):
    @property
    def solution(self) -> Tensor[Solution]: ...
    @property
    def residuals(self) -> Tensor[Residuals]: ...
    @property
    def rank(self) -> Tensor[Rank]: ...
    @property
    def singular_values(self) -> Tensor[SingularValues]: ...

class linalg_solve_ex[Result: IntTuple, Info: IntTuple](
    tuple[Tensor[Result], Tensor[Info]]
):
    @property
    def result(self) -> Tensor[Result]: ...
    @property
    def info(self) -> Tensor[Info]: ...

class linalg_slogdet[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def sign(self) -> Tensor[Shape]: ...
    @property
    def logabsdet(self) -> Tensor[Shape]: ...

class slogdet[Shape: IntTuple](tuple[Tensor[Shape], Tensor[Shape]]):
    @property
    def sign(self) -> Tensor[Shape]: ...
    @property
    def logabsdet(self) -> Tensor[Shape]: ...

def __getattr__(name: str) -> Any: ...
