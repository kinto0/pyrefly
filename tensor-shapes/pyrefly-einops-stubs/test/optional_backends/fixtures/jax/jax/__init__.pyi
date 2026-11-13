# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from shape_extensions import IntTuple

# Only the nominal shaped type is needed; this is not a JAX API stub.
class Array[Shape: IntTuple = IntTuple]:
    @property
    def shape(self) -> Shape: ...
