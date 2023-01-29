from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision
import warnings
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from torchvision import transforms
from Final.IrisLocalization import IrisLocalization
from Final.IrisNormalization import IrisNormalization
from Final.ImageEnhancement import ImageEnhancement
from Final.FeatureExtraction import FeatureExtraction
from cnn_feature.cnn_normal import CustomConvNet
from preprocessing.normalization_cnn import cnn_normalization
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding

warnings.filterwarnings("ignore")

app = FastAPI()

app.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class IrisNotFoundException(Exception):
    pass


def model_initialization(train_features: np.ndarray, train_classes: np.ndarray, n: int) -> KNeighborsClassifier:
    train_redfeatures = train_features.copy()
    lda = LinearDiscriminantAnalysis(n_components = n)
    lle = LocallyLinearEmbedding(n_neighbors=n+1,n_components=n)
    if n < 108:
        lda.fit(train_features,train_classes)
        train_redfeatures = lda.transform(train_features)
        
    if n >= 108 and n < 323:
        lle.fit(train_features)
        train_redfeatures = lle.transform(train_features)
    
    knn = KNeighborsClassifier(n_neighbors = 1, metric = 'cosine')
    knn.fit(train_redfeatures, train_classes)
    return knn, lda, lle


@app.on_event("startup")
def init_model() -> None:
    print('Server startup.')
    #cnn = CustomConvNet(108)
    #cnn.load_state_dict(torch.load('model_cnn.pth'))
    #app.model = cnn.eval()
    rootpath = "Dataset/CASIA1/"
    train_data, train_classes = estrazione_dataset(rootpath)
    app.knn_model, app.lda, app.lle = model_initialization(train_data, train_classes, 200)
    print('Start up is over. Server is running')


@app.get('/')
def ping() -> str:
    return "ok"


@app.post('/recognition')
def recognition(file: UploadFile) -> Response:

    # load image
    img = Image.open(file.file)
    img = np.array(img)

    #predicted = 3
    try:
        '''
        normalized = cnn_normalization(np_image)
        normalized = normalized.astype('float32',casting='same_kind')
        normalized = torch.tensor(normalized).reshape(1,1,360,80)
        #normalized = torch.repeat_interleave(normalized,3, dim=3)
        print(normalized.shape)
        
        print(type(normalized), normalized.shape)
        subject = 0
        with torch.no_grad():
            prediction = app.model(normalized)
            _, predicted = torch.max(prediction.data, 1)
        '''
        img = preprocessing(img)
        img = img.reshape(1,-1)
        img = app.lle.transform(img)
        predicted = app.knn_model.predict(img) 
        if predicted == 3:
            raise IrisNotFoundException
        return JSONResponse(content={"subject": str(predicted)}, status_code=200)
    
    except IrisNotFoundException:
        return Response(content="Access denied!", status_code=422)


def preprocessing(image: np.ndarray) -> np.ndarray:
    iris, pupil = IrisLocalization(image)
    normalized = IrisNormalization(image, pupil, iris)
    ROI = ImageEnhancement(normalized)
    image = FeatureExtraction(ROI)
    return image

def estrazione_dataset(data_set_path: str, train: bool = True, n_samples: int = 108) -> tuple[np.ndarray, np.ndarray]:
        tipo = '_1_' if  train else '_2_'
        offset = 3 if train else 4
        features = np.zeros((n_samples*offset,1536))
        classes = np.zeros(n_samples*offset, dtype = np.uint8)

        for i in range(1,n_samples+1):
            filespath = data_set_path + str(i) + "/"
            for j in range(1,offset+1):
                irispath = filespath + str(i).zfill(3) + tipo + str(j) + ".jpg"
                img = cv2.imread(irispath, 0)
                features[(i-1)*offset+j-1, :] = preprocessing(img)
                classes[(i-1)*offset+j-1] = i 
        return features, classes 