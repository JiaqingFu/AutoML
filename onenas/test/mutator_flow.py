import oneflow.experimental as flow
import random
import numpy as np
import unittest
from collections import Counter

import onenas.nn as nn
from onenas.mutator import Sampler
from onenas.nn.mutator import process_inline_mutation, extract_mutation_from_pt_module
from onenas.serializer import model_wrapper
from onenas.utils import ContextStack

flow.enable_eager_execution()

input_arr = np.array(
    [
        [-0.94630778, -0.83378579, -0.87060891],
        [2.0289922, -0.28708987, -2.18369248],
        [0.35217619, -0.67095644, -1.58943879],
        [0.08086036, -1.81075924, 1.20752494],
        [0.8901075, -0.49976737, -1.07153746],
        [-0.44872912, -1.07275683, 0.06256855],
        [-0.22556897, 0.74798368, 0.90416439],
        [0.48339456, -2.32742195, -0.59321527],
    ],
    dtype=np.float32,
)

input_arr = flow.Tensor(input_arr)


def _unpack_if_only_one(ele):
    if len(ele) == 1:
        return ele[0]
    return ele


class RandomSampler(Sampler):
    def __init__(self):
        self.counter = 0

    def choice(self, candidates, *args, **kwargs):
        self.counter += 1
        return random.choice(candidates)


def _get_converted_pytorch_model(model_ir):
    mutation = {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model_ir.history}
    with ContextStack('fixed', mutation):
        model = model_ir.python_class(**model_ir.python_init_params)
        return model


@model_wrapper
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell1 = nn.LayerChoice([nn.Linear(3, 8), nn.Linear(3, 8, bias=False)])

    def forward(self, x):
        x = self.cell1(x)
        return x


raw_model, mutators = extract_mutation_from_pt_module(Net())
sampler = RandomSampler()
model = raw_model
for mutator in mutators:
    model = mutator.bind_sampler(sampler).apply(model)

converted_model = _get_converted_pytorch_model(model)

output_arr = converted_model(input_arr).numpy()

print(converted_model.cell1._parameters)
print(output_arr)
