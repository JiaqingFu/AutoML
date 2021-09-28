import unittest
from collections import OrderedDict

import numpy as np

import oneflow.experimental as flow
import oneflow.typing as tp

flow.enable_eager_execution()

input_arr = np.random.random((1, 1, 28, 28)).astype(np.float32)


class Net(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = flow.nn.Conv2d(1, 20, 5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = flow.nn.ReLU(x)
        return x


net = Net()

x = flow.Tensor(input_arr)
x = net(x)

print(x.numpy())
