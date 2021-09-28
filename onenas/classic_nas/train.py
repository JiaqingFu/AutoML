import random
import oneflow.experimental as flow
import onenas.nn as nn
from onenas.mutator import Sampler
from onenas.nn.mutator import extract_mutation_from_pt_module
from onenas.serializer import model_wrapper
from onenas.utils import ContextStack
from onenas.strategy.utils import dry_run_for_search_space
from onenas.strategy.bruteforce import random_generator, get_targeted_model

DEVICE = 'cuda'
MAX_EPOCHS = 2
flow.enable_eager_execution()


def unpack_if_only_one(ele):
    if len(ele) == 1:
        return ele[0]
    return ele


def get_converted_pytorch_model(model_ir):
    mutation = {mut.mutator.label: unpack_if_only_one(mut.samples) for mut in model_ir.history}
    with ContextStack('fixed', mutation):
        model = model_ir.python_class(**model_ir.python_init_params)
        model.to(DEVICE)
        return model


class RandomSampler(Sampler):
    def __init__(self):
        self.counter = 0

    def choice(self, candidates, *args, **kwargs):
        self.counter += 1
        return random.choice(candidates)


@model_wrapper
class Net(nn.Module):
    def __init__(self, hidden_size):
        # 不能写成 super(Net, self).__init__()
        super().__init__()

        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1 = nn.LayerChoice([
            nn.Conv2d(1, 20, 5, 1),
            nn.Conv2d(1, 20, 3, 1)
        ])  # nn.Conv2d(1, 50, 5, 1)
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(20, 50, 5, 1, padding=2),
            nn.Conv2d(20, 50, 3, 1, padding=1)
        ])  # nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.LayerChoice([
            nn.Linear(1800, hidden_size),
            nn.Linear(1800, hidden_size, bias=False)
        ])
        self.fc2 = nn.Linear(hidden_size, 10)

        self.relu = flow.F.relu
        self.max_pool2d = flow.nn.MaxPool2d(kernel_size=2, stride=2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2d(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


if __name__ == '__main__':

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist()

    base_model, applied_mutators = extract_mutation_from_pt_module(Net(128))
    # base_model_ir.evaluator = trainer  ???

    # TODO: Replace with strategy/bruteforce
    search_space = dry_run_for_search_space(base_model, applied_mutators)

    models = []
    for sample in random_generator(search_space):
        model_ir = get_targeted_model(base_model, applied_mutators, sample)
        models.append((sample, get_converted_pytorch_model(model_ir)))

    for sample, model in models:
        print("===================================")
        model.train()
        corss_entropy = flow.nn.CrossEntropyLoss()
        sgd = flow.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

        for epoch in range(MAX_EPOCHS):
            for i, (images, labels) in enumerate(zip(train_images, train_labels)):
                images = flow.Tensor(images).to(DEVICE)
                labels = flow.Tensor(labels, dtype=flow.int).to(DEVICE)
                logits = flow.Tensor(model(images))
                loss = corss_entropy(logits, labels)
                loss.backward()
                sgd.step()
                sgd.zero_grad()

                if i % 100 == 0:
                    print("sample: ", list(sample.values()),
                          "| epoch: ", epoch,
                          "| i: ", i,
                          "| loss: ", loss.numpy().mean()
                          )
