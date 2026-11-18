# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from shape_extensions import assert_shape


def test_solve_triangular_unitriangular() -> None:
    # https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html
    matrix: torch.Tensor[[2, 2]] = torch.tensor(((0.0, 2.0), (0.0, 0.0)))
    rhs: torch.Tensor[[2, 1]] = torch.tensor(((5.0,), (2.0,)))
    solution = torch.linalg.solve_triangular(
        matrix,
        rhs,
        upper=True,
        unitriangular=True,
    )
    assert_shape(solution.shape, (2, 1))
    assert torch.equal(solution, torch.tensor(((1.0,), (2.0,))))


def test_solve_triangular_nonunit_diagonal() -> None:
    matrix: torch.Tensor[[2, 2]] = torch.tensor(((2.0, 1.0), (0.0, 4.0)))
    rhs: torch.Tensor[[2, 1]] = torch.tensor(((6.0,), (8.0,)))
    solution = torch.linalg.solve_triangular(
        matrix,
        rhs,
        upper=True,
        unitriangular=False,
    )
    assert_shape(solution.shape, (2, 1))
    assert torch.equal(solution, torch.tensor(((2.0,), (2.0,))))
