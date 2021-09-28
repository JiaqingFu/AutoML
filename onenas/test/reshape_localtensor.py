import numpy as np
import oneflow.experimental as flow
import oneflow
flow.enable_eager_execution()

x = np.array(
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
).astype(np.float32)
tensor_in = flow.Tensor(x)
l_tensor_in = oneflow._oneflow_internal.LocalTensor(x)
of_shape = flow.reshape(tensor_in, shape=[2, 2, 2, -1]).numpy().shape
np_shape = (2, 2, 2, 2)

print(of_shape)
print(np_shape)
