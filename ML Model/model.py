import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import  transforms
import os
from dataset import IrisDataset
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.enabled = False


class CustomConvNet(nn.Module): 
  def __init__(self, num_classes):
    super(CustomConvNet, self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc1 = nn.Sequential(nn.Linear(11*2*512, 4096),  nn.ReLU(), nn.Dropout(), nn.BatchNorm1d(4096))
    self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(), nn.BatchNorm1d(4096))
    self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.conv6(out)
    out = self.conv7(out)
    out = self.conv8(out)
    out = self.conv9(out)
    out = self.conv10(out)
    out = self.conv11(out)
    out = self.conv12(out)
    out = self.conv13(out)
    out = out.view(-1, 11*2*512)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out


if __name__ == '__main__':
    rootpath = "Dataset/CASIA_Iris_interval_norm/"

    hyper_param_epoch = 50
    hyper_param_batch = 10
    hyper_param_learning_rate = 0.001

    transforms_train = transforms.Compose([transforms.Resize((360, 80)),
                                        transforms.RandomRotation(10.),
                                        transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((360, 80)),
                                        transforms.ToTensor()])

  
    samples = 50
    train_data_set = IrisDataset(data_set_path=rootpath, transforms=transforms_train, n_samples = samples)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

    test_data_set = IrisDataset(data_set_path=rootpath, transforms=transforms_test, train=False, n_samples=samples)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

    if not (train_data_set.num_classes == test_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device='cpu'
    num_classes = train_data_set.num_classes
    custom_model = CustomConvNet(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    print('Start training')
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
