# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, Int, IntVar
from torch import nn, Tensor


def test_lstm_initial_state() -> None:
    lstm = nn.LSTM(6, 4, batch_first=True)
    inputs = torch.ones(2, 5, 6)
    hx = (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))
    output, (hidden, cell) = lstm(inputs, hx)
    assert_shape(output.shape, (2, 5, 4))
    assert_shape(hidden.shape, (1, 2, 4))
    assert_shape(cell.shape, (1, 2, 4))

    output, (hidden, cell) = lstm.forward(input=inputs, hx=(hidden, cell))
    assert_shape(output.shape, (2, 5, 4))
    assert_shape(hidden.shape, (1, 2, 4))
    assert_shape(cell.shape, (1, 2, 4))


def test_lstm_bidirectional_initial_state() -> None:
    lstm = nn.LSTM(6, 4, num_layers=3, bidirectional=True)
    inputs = torch.ones(5, 2, 6)
    hx = (torch.zeros(6, 2, 4), torch.zeros(6, 2, 4))
    output, (hidden, cell) = lstm(inputs, hx=hx)
    assert_shape(output.shape, (5, 2, 8))
    assert_shape(hidden.shape, (6, 2, 4))
    assert_shape(cell.shape, (6, 2, 4))


def test_lstm_explicit_none_state() -> None:
    lstm = nn.LSTM(6, 4, batch_first=True)
    inputs = torch.ones(2, 5, 6)
    output, _ = lstm.forward(inputs, None)
    assert_shape(output.shape, (2, 5, 4))
    output, _ = lstm(inputs, hx=None)
    assert_shape(output.shape, (2, 5, 4))


if TYPE_CHECKING:

    def check_symbolic_initial_state[B: IntVar, T: IntVar](
        batch: Int[B], steps: Int[T]
    ) -> None:
        lstm = nn.LSTM(6, 4, batch_first=True)
        inputs = torch.ones(batch, steps, 6)
        hx = (torch.zeros(1, batch, 4), torch.zeros(1, batch, 4))
        output, (hidden, cell) = lstm.forward(inputs, hx=hx)
        assert_type(output, Tensor[[B, T, 4]])
        assert_type(hidden, Tensor[[1, B, 4]])
        assert_type(cell, Tensor[[1, B, 4]])
