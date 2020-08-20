import itertools

import torch

import pytest
import torchtrain as tt


@pytest.mark.parametrize(
    "cast,inputs",
    itertools.product(
        [
            (tt.cast.BFloat16(), torch.bfloat16),
            (tt.cast.Bool(), torch.bool),
            (tt.cast.Byte(), torch.uint8),
            (tt.cast.Char(), torch.int8),
            (tt.cast.Double(), torch.double),
            (tt.cast.Float(), torch.float),
            (tt.cast.Half(), torch.half),
            (tt.cast.Int(), torch.int),
            (tt.cast.Long(), torch.long),
            (tt.cast.Short(), torch.short),
        ],
        [torch.randn(2, 3, 4)],
    ),
)
def test_cast(cast, inputs):
    caster, desired = cast
    assert caster(inputs).dtype == desired
