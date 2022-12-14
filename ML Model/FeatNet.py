from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, load
from torch.nn import functional as F
import cv2
import numpy as np
from PIL import Image
from skimage.filters.rank import equalize
from skimage.morphology import disk
from torchvision import datasets, transforms
import os
from dataset import IrisDataset
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.enabled = False


# da far funzionare
def ImageEnhancement(normalized_iris: str):
    normalized_iris = cv2.imread(normalized_iris)
    print(normalized_iris.shape)
    normalized_iris = normalized_iris.astype(np.uint8)
    print(normalized_iris.shape)
    enhanced_image=normalized_iris    
    enhanced_image = equalize(enhanced_image, disk(16))
    print(enhanced_image.shape)
    roi = enhanced_image[0:48,:]
    return roi

rootpath = "Dataset/CASIA_Iris_interval_norm/"

class FeatNet(nn.Module):
    def __init__(self, n_classes, pretrainedName=""):
        super(FeatNet, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_a', nn.Conv2d(1, 16, kernel_size=(3, 7), stride=1, padding=(1, 3), bias=False)),
            ('tan1_a', nn.Tanh())
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('pool1_a', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv2_a', nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False)),
            ('tan2_a', nn.Tanh())
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('pool2_a', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv3_a', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('tan3_a', nn.Tanh())
        ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('pool2_a', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv3_a', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('tan3_a', nn.Tanh())
        ]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('pool2_a', nn.AvgPool2d(kernel_size=1, stride=1)),
            ('conv3_a', nn.Conv2d(128, n_classes, kernel_size=3, stride=1, padding=1, bias=False)),
            ('tan3_a', nn.Tanh())
        ]))

        
        self.fuse_a = nn.Conv2d(348, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.flatten = nn.Flatten()

        if pretrainedName != '':
            modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pretrained", pretrainedName)
            self.load_state_dict(load(modelPath))
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x2 = F.interpolate(x2, size=(360, 80), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(360, 80), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=(360, 80), mode='bilinear', align_corners=False)
        x5 = F.interpolate(x5, size=(360, 80), mode='bilinear', align_corners=False)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)

        out = self.fuse_a(x6)
        out = self.flatten(out)
        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
        nn.AdaptiveAvgPool2d((1, 1)))


if __name__ == '__main__':
    hyper_param_epoch = 50
    hyper_param_batch = 1
    hyper_param_learning_rate = 0.001

    transforms_train = transforms.Compose([transforms.Resize((360, 80)),
                                        transforms.RandomRotation(10.),
                                        transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((360, 80)),
                                        transforms.ToTensor()])

  
    samples = 108
    train_data_set = IrisDataset(data_set_path=rootpath, transforms=transforms_train, n_samples = samples)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

    test_data_set = IrisDataset(data_set_path=rootpath, transforms=transforms_test, train=False, n_samples=samples)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

    if not (train_data_set.num_classes == test_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_classes = train_data_set.num_classes
    #custom_model = CustomConvNet(num_classes=num_classes).to(device)
    custom_model = FeatNet(n_classes = num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)

    #Training
    custom_model.train()
    for e in range(hyper_param_epoch):
        train_loss = 0
        for i_batch, item in enumerate(train_loader):

            images = item['image'].float().to(device)
            labels = item['label'].to(device)
            # Forward pass
            outputs = custom_model(images)

            loss = criterion(outputs, labels)
            train_loss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, hyper_param_epoch, train_loss/len(train_loader)))

    # Test the model
    custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    with torch.no_grad():
        correct = 0
        total = 0
        for item in test_loader:
            images = item['image'].float().to(device)
            labels = item['label'].to(device)

            outputs = custom_model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            print('predicted : ',predicted, '\nlabels : ',labels)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
