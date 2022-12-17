import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import cv2
import numpy as np
from PIL import Image
from skimage.filters.rank import equalize
from skimage.morphology import disk
from torchvision import datasets, transforms
import os
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

def estrazione_dataset(rootpath,train=True):
    train_features = np.zeros((60,360,80))
    train_classes = np.zeros(60, dtype = np.uint8)
    test_features = np.zeros((80,360,80))
    test_classes = np.zeros(80, dtype = np.uint8)

    for i in range(1,20):
        filespath = rootpath + str(i) + "/"
        for j in range(1,4):
            irispath = filespath + str(i).zfill(3) + "_1_" + str(j) + ".jpg"
            ROI = cv2.imread(irispath, cv2.IMREAD_GRAYSCALE)
            #ROI = ImageEnhancement(irispath)
            train_features[(i-1)*3+j-1, :, :] = ROI
            train_classes[(i-1)*3+j-1] = i
        for k in range(1,5):
            irispath = filespath + str(i).zfill(3) + "_2_" + str(k) + ".jpg"
            ROI = cv2.imread(irispath,  cv2.IMREAD_GRAYSCALE)
            #ROI = ImageEnhancement(irispath)
            test_features[(i-1)*4+k-1, :,:] = ROI
            test_classes[(i-1)*4+k-1] = i
    if train:
        return train_features, train_classes
    else:
        return test_features, test_classes  
    



def transform_to_np(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    return image


class CustomImageDataset(Dataset):
    def read_data_set(self):
        images, labels = estrazione_dataset(self.data_set_path, self.train)
        classes = set(labels)
        return images, labels, len(images), len(classes)

    def __init__(self, data_set_path, transforms, train=True):
            self.data_set_path = data_set_path
            self.train = train
            self.transforms = transforms
            self.images, self.labels, self.length, self.num_classes = self.read_data_set()       
        
    def __getitem__(self, index):
            image= self.images[index]
            if self.transforms is not None:
                image = self.transforms(Image.fromarray(np.uint8(image)))
            return {'image': image, 'label': self.labels[index]}

    def __len__(self):
            return self.length


class CustomConvNet(nn.Module): 
  def __init__(self, num_classes):
    super(CustomConvNet, self).__init__()
      
    self.layer1 = self.conv_module(1, 16)
    self.layer2 = self.conv_module(16, 32)
    self.layer3 = self.conv_module(32, 64)
    self.layer4 = self.conv_module(64, 128)
    self.layer5 = self.conv_module(128, 256)
    self.layer6 = self.global_avg_pool(256, num_classes)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = out.view(-1, num_classes)
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


hyper_param_epoch = 50
hyper_param_batch = 1
hyper_param_learning_rate = 0.001

transforms_train = transforms.Compose([transforms.Resize((360, 80)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((360, 80)),
                                      transforms.ToTensor()])

print('Creazione training set')
train_data_set = CustomImageDataset(data_set_path=rootpath, transforms=transforms_train)
print(f'Training set creato, shape: {len(train_data_set)}')
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path=rootpath, transforms=transforms_test, train=False)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(f'device: {device}')
num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
custom_model.train()
for e in range(hyper_param_epoch):
  for i_batch, item in enumerate(train_loader):
    images = item['image'].to(device)
    labels = item['label'].to(device)
    
    # Forward pass
    outputs = custom_model(images)

    loss = criterion(outputs, labels)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, hyper_param_epoch, loss.item()))

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
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]