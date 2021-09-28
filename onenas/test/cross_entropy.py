import oneflow.experimental as flow
import numpy as np

flow.enable_eager_execution()

g_test_samples = [
    {
        "input": np.array(
            [
                [-0.6980871, 0.4765042, -1.969919, 0.28965086, -0.53548324],
                [-0.26332688, 0.27541, 0.30080616, 0.09914763, 0.53522176],
                [0.7332028, 0.38375184, -0.2831992, -0.9833142, 0.387824],
            ]
        ),
        "target": np.array([3, 3, 4], dtype=np.int32),
        "out": np.array([1.1380, 1.7332, 1.4287], dtype=np.float32),
        "out_sum": np.array([4.2999], dtype=np.float32),
        "out_mean": np.array([1.4333], dtype=np.float32),
    },
    {
        "input": np.array(
            [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
        ),
        "target": np.array([[[1, 0], [0, 1]]], dtype=np.int32),
        "out": np.array([[[0.6882, 0.6832], [0.8544, 1.8006]]], dtype=np.float32),
        "out_sum": np.array([4.0263], dtype=np.float32),
        "out_mean": np.array([1.0066], dtype=np.float32),
    },
    {
        "input": np.array(
            [
                [[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]],
                [[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]],
            ]
        ),
        "target": np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=np.int32),
        "out": np.array(
            [
                [[0.6882, 0.6832], [0.8544, 1.8006]],
                [[0.6882, 0.6832], [0.8544, 1.8006]],
            ],
            dtype=np.float32,
        ),
        "out_sum": np.array([8.0526], dtype=np.float32),
        "out_mean": np.array([1.0066], dtype=np.float32),
    },
    {
        "input": np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]),
        "target": np.array([[1, 0, 0, 1]], dtype=np.int32),
        "out": np.array([[0.6882, 0.6832, 0.8544, 1.8006]], dtype=np.float32, ),
        "out_sum": np.array([4.0263], dtype=np.float32),
        "out_mean": np.array([1.0066], dtype=np.float32),
    },
]

for sample in g_test_samples:
    loss = flow.nn.CrossEntropyLoss(reduction=None)
    input = flow.Tensor(sample["input"], dtype=flow.float32)
    target = flow.Tensor(sample["target"], dtype=flow.int32)
    of_out = loss(input, target)
    assert np.allclose(of_out.numpy(), sample["out"], 1e-4, 1e-4)

    loss_sum = flow.nn.CrossEntropyLoss(reduction="sum")
    of_out_sum = loss_sum(input, target)
    assert np.allclose(of_out_sum.numpy(), sample["out_sum"], 1e-4, 1e-4)

    loss_mean = flow.nn.CrossEntropyLoss(reduction="mean")
    of_out_mean = loss_mean(input, target)
    assert np.allclose(of_out_mean.numpy(), sample["out_mean"], 1e-4, 1e-4)
