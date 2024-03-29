import os
from PIL import Image
from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import numpy as np
import ctypes
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.svm import SVC

#rete feed-forward. Prende l'input, lo alimenta attraverso diversi livelli uno dopo l'altro e infine fornisce l'output.
#accuracy 95.6%
class CustomImageDataset(Dataset):
  def read_data_set(self):
    all_img_files = []
    all_labels = []
    
    class_names = os.walk(self.data_set_path).__next__()[1]
    class_names.sort()
    
    for index, class_name in enumerate(class_names):
            
      label = index
      img_dir = os.path.join(self.data_set_path, class_name)
      img_files = os.walk(img_dir).__next__()[2]
      
      for img_file in img_files:
        img_file = os.path.join(img_dir, img_file)
        img = Image.open(img_file)
        if img is not None:
          all_img_files.append(img_file)
          all_labels.append(label)

    return all_img_files, all_labels, len(all_img_files), len(class_names)

  def __init__(self, data_set_path, transforms=None):
    self.data_set_path = data_set_path
    self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
    self.transforms = transforms
  
  def __getitem__(self, index):
    image = Image.open(self.image_files_path[index])
    image = image.convert("L")

    if self.transforms is not None:
      image = self.transforms(image)

    return {'image': image, 'label': self.labels[index]}

  def __len__(self):
    return self.length


class CustomConvNet(nn.Module): 
  def __init__(self, num_classes):
    self.num_classes = num_classes
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
    out = out.view(-1, self.num_classes)
    
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


hyper_param_epoch = 100
hyper_param_batch = 100
hyper_param_learning_rate = 0.001

transforms_train = transforms.Compose([transforms.Resize((360, 80)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((360, 80)),
                                      transforms.ToTensor()])

train_data_set = CustomImageDataset(data_set_path="Dataset/train", transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path="Dataset/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
  
np.set_printoptions(threshold=np.inf)

summary(custom_model, (1, 360, 80))
with torch.no_grad():
  correct = 0
  total = 0
  y_true = [] 
  y_pred = []
  for item in test_loader:
    images = item['image'].to(device)
    labels = item['label'].to(device)
    #train_label = np.array([sample['label'] for sample in train_loader], dtype="object")
    #train_data = np.array([sample['image'].reshape(-1) for sample in train_loader])
    #test_data = np.array([sample['image'].reshape(-1) for sample in test_loader], dtype="object")
    #test_label = np.array([sample['label'] for sample in test_loader])
    
    outputs = custom_model(images)
    _, predicted = torch.max(outputs.data, 1)
    print('predicted : ',predicted, '\nlabels : ',labels)
    labels = list(labels.numpy())
    prediceted = list(predicted.numpy())
    y_true += (labels)
    y_pred += (predicted)

  print('Accuracy score: ', accuracy_score(y_true, y_pred, sample_weight=None))
  print('Precision score: ', precision_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None, zero_division=1))
  print('Recall score: ', recall_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None, zero_division=1))
  print('F score: ', f1_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None, zero_division=1))
  print(classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=1))
  confusion_matrix = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=None)
  disp.plot()
  plt.show()
  fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
  roc_auc = auc(fpr, tpr)
  display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
  display.plot()
  plt.show()

torch.save(custom_model.state_dict(), "model_cnn.pth")
