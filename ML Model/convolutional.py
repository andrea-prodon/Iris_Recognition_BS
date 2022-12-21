import torch
import torch.optim as optim
from torch import nn 
import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Union
from tqdm.notebook import tqdm, trange
from dataset import IrisDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
torch.backends.cudnn.benchmark = False



class CNN(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_feature: int, output_size: int
    ) -> None:
        """
        Simple model that uses convolutions

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_feature: size of the hidden dimensions to use
        :param output_size: expected size of the output
        """
        super().__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=n_feature, kernel_size=3
        )
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=3)
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=3)

        self.fc1 = nn.Linear(n_feature * 5 * 5, 10)
        self.fc2 = nn.Linear(10, output_size)

    def forward(self, 
                x: torch.Tensor, 
                return_conv1: bool = False, 
                return_conv2: bool = False, 
                return_conv3: bool = False
        ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :param return_conv1: if True return the feature maps of the first convolution
        :param return_conv2: if True return the feature maps of the second convolution
        :param return_conv3: if True return the feature maps of the third convolution

        :returns: predictions with size [batch, output_size]
        """
        print(f'pre convolution 1: {x.shape}')
        x = self.conv1(x)
        if return_conv1:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        print(f'pre convolution 2: {x.shape}')
        x = self.conv2(x)
        if return_conv2:
            return x
        x = F.relu(x)

        # Not so easy to keep track of shapes... right?
        # An useful trick while debugging is to feed the model a fixed sample batch
        # and print the shape at each step, just to be sure that they match your expectations.

        # print(x.shape)
        print(f'pre convolution 3: {x.shape}')
        x = self.conv3(x)
        if return_conv3:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)
        print(f'pre linear 1: {x.shape}')
        x = self.fc1(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2)
        print(f'pre linear 2: {x.shape}')
        x = self.fc2(x)
        return x

# A fixed sample batch
# x, _ = next(iter(train_loader))
# model = CNN(input_size, n_channels, 9, 10)
# _ = model(x)

def permute_pixels(images: torch.Tensor, perm: Optional[torch.Tensor]) -> torch.Tensor:
    """ Permutes the pixel in each image in the batch

    :param images: a batch of images with shape [batch, channels, w, h]
    :param perm: a permutation with shape [w * h]

    :returns: the batch of images permuted according to perm
    """
    if perm is None:
        return images

    batch_size = images.shape[0]
    n_channels = images.shape[1]
    w = images.shape[2]
    h = images.shape[3]
    images = images.view(batch_size, n_channels, -1)
    images = images[..., perm]
    images = images.view(batch_size, n_channels, w, h)
    return images


def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average

    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager

        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager

def test_model(
    test_dl: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    perm: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Dict[str, Union[float, Callable[[Optional[float]], float]]]:
    """Compute model accuracy on the test set

    :param test_dl: the test dataloader
    :param model: the model to train
    :param perm: if not None, permute the pixel in each image according to perm

    :returns: computed accuracy
    """
    model.eval()
    test_loss_averager = make_averager()  # mantain a running average of the loss
    correct = 0
    for data, target in test_dl:
        # send to device
        data, target = data.to(device), target.to(device)

        if perm is not None:
            data = permute_pixels(data, perm)

        output = model(data)

        test_loss_averager(F.cross_entropy(output, target))

        # get the index of the max probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum().item()

    return {
        "accuracy": 100.0 * correct / len(test_dl.dataset),
        "loss_averager": test_loss_averager,
        "correct": correct,
    }

def fit(
    epochs: int,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    tag: str,
    perm: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> float:
    """Train the model and computes metrics on the test_loader at each epoch

    :param epochs: number of epochs
    :param train_dl: the train dataloader
    :param test_dl: the test dataloader
    :param model: the model to train
    :param opt: the optimizer to use to train the model
    :param tag: description of the current model
    :param perm: if not None, permute the pixel in each image according to perm

    :returns: accucary on the test set in the last epoch
    """
    for epoch in trange(epochs, desc="train epoch"):
        model.train()
        train_loss_averager = make_averager()  # mantain a running average of the loss

        # TRAIN
        tqdm_iterator = tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            desc=f"batch [loss: None]",
            leave=False,
        )
        for batch_idx, elem in tqdm_iterator:

            data, target = elem['image'], elem['label']
            print(data.shape)
            # send to device
            data, target = data.to(device).float(), target.to(device)

            if perm is not None:
                data = permute_pixels(data, perm)

            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss_averager(loss.item())

            tqdm_iterator.set_description(
                f"train batch [avg loss: {train_loss_averager(None):.3f}]"
            )
            tqdm_iterator.refresh()

        # TEST
        test_out = test_model(test_dl, model, perm, device)

        print(
            f"Epoch: {epoch}\n"
            f"Train set: Average loss: {train_loss_averager(None):.4f}\n"
            f"Test set: Average loss: {test_out['loss_averager'](None):.4f}, "
            f"Accuracy: {test_out['correct']}/{len(test_dl.dataset)} "
            f"({test_out['accuracy']:.0f}%)\n"
        )
    models_accuracy[tag] = test_out['accuracy']
    return test_out['accuracy']



def get_model_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Encapsulate the creation of the model's optimizer, to ensure that we use the
    same optimizer everywhere

    :param model: the model that contains the parameter to optimize

    :returns: the model's optimizer
    """
    return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # return optim.SGD(model.parameters(), lr=0.01, momentum=0.1, weight_decay=1e-5)


# Define the number of the epochs
epochs = 4

# Number of the feature maps in the CNN
n_features = 6

# Define a dictionary that will contain the performance of the different models
models_accuracy = {}

rootpath = "Dataset/CASIA_Iris_interval_norm/"
hyper_param_batch = 10
samples = 108
train_data_set = IrisDataset(data_set_path=rootpath, transforms=None, n_samples = samples)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = IrisDataset(data_set_path=rootpath, transforms=None, train=False, n_samples=samples)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

input_size = test_data_set[0]['image'].shape[0] * test_data_set[0]['image'].shape[1]
n_channels =1
output_size = 108
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_cnn = CNN(input_size, n_channels, n_features, output_size)
model_cnn.to(device)
optimizer = get_model_optimizer(model_cnn)

#print(f'Number of parameters: {count_parameters(model_cnn)}')

fit(epochs=epochs, 
    train_dl=train_loader,
    test_dl=test_loader,
    model=model_cnn,
    opt=optimizer, 
    tag='cnn',
    device=device)