import torch
import random
import unittest
from collections import Counter

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import Sampler, basic_unit
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.execution.python import _unpack_if_only_one
from nni.retiarii.nn.pytorch.mutator import process_inline_mutation, extract_mutation_from_pt_module
from nni.retiarii.serializer import model_wrapper
from nni.retiarii.utils import ContextStack


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
        self.cell1 = nn.LayerChoice([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)])
        self.cell2 = nn.LayerChoice([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)])

    def forward(self, x):
        x = self.cell1(x)
        x = self.cell2(x)
        return x


raw_model, mutators = extract_mutation_from_pt_module(Net())
sampler = RandomSampler()
model = raw_model
for mutator in mutators:
    model = mutator.bind_sampler(sampler).apply(model)

converted_model = _get_converted_pytorch_model(model)
print(converted_model.cell1)
print(converted_model.cell2)

print(converted_model(torch.randn(1, 16)))