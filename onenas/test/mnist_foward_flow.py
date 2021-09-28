import oneflow.experimental as flow
import random
import numpy as np
import unittest
from collections import Counter

flow.enable_eager_execution()

input_arr = np.random.random((1, 1, 28, 28)).astype(np.float32)

input_arr = flow.Tensor(input_arr)


class Net(flow.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 128
        self.conv1 = flow.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = flow.nn.Conv2d(20, 50, 3, 1)
        self.fc1 = flow.nn.Linear(5 * 5 * 50, hidden_size)
        self.fc2 = flow.nn.Linear(hidden_size, 10)
        self.relu = flow.nn.ReLU()
        self.relu1 = flow.nn.ReLU()
        self.relu2 = flow.nn.ReLU()
        self.max_pool2d = flow.nn.MaxPool2d(2, 2)
        self.max_pool2d1 = flow.nn.MaxPool2d(2, 2)
        self.logsoftmax = flow.nn.LogSoftmax(dim=1)
        self.view = flow.dynamic_reshape

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.relu1(self.conv2(x))
        x = self.max_pool2d1(x)
        x = x.reshape(shape=[x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)
        return x


net = Net()
output_arr = net(input_arr).numpy()

print(output_arr)
