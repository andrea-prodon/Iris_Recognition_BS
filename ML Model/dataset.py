from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image






class IrisDataset(Dataset):
    def read_data_set(self):
        images, labels = self.estrazione_dataset()
        classes = set(labels)
        return images, labels, len(images), len(classes)

    def __init__(self, data_set_path, transforms, train=True, n_samples=108):
            self.data_set_path = data_set_path
            self.train = train
            self.n_samples = n_samples
            self.transforms = transforms
            self.images, self.labels, self.length, self.num_classes = self.read_data_set()       
        
    def __getitem__(self, index):
            image= self.images[index]
            if self.transforms is not None:
                image = self.transforms(Image.fromarray(np.uint8(image)))
            return {'image': image, 'label': self.labels[index]}

    def __len__(self):
            return self.length
    
    def estrazione_dataset(self):
        tipo = '_1_' if not self.train else '_2_'
        offset = 3 if not self.train else 4
        features = np.zeros((self.n_samples*offset,360,80))
        classes = np.zeros(self.n_samples*offset, dtype = np.uint8)

        for i in range(1,self.n_samples+1):
            filespath = self.data_set_path + str(i) + "/"
            for j in range(1,offset+1):
                irispath = filespath + str(i).zfill(3) + tipo + str(j) + ".jpg"
                ROI = cv2.imread(irispath, cv2.IMREAD_GRAYSCALE)
                #ROI = ImageEnhancement(irispath)
                features[(i-1)*offset+j-1, :, :] = ROI
                classes[(i-1)*offset+j-1] = i - 1

        return features, classes 


 

def transform_to_np(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    return image


