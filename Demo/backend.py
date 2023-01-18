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
import joblib
from cnn_feature.cnn_normal import CustomConvNet
from preprocessing.normalization_cnn import cnn_normalization
import cv2

warnings.filterwarnings("ignore")

app = FastAPI()

app.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class IrisNotFoundException(Exception):
    pass

@app.on_event("startup")
def init_model() -> None:
    cnn = CustomConvNet(108)
    cnn.load_state_dict(torch.load('model_cnn.pth'))
    app.model = cnn.eval()


@app.get('/')
def ping() -> str:
    return "ok"


@app.post('/recognition')
def recognition(file: UploadFile):

    # load image
    img = Image.open(file.file)
    np_image = np.array(img)
    try:
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

        return JSONResponse(content={"subject": str(predicted)}, status_code=200)
    
    except IrisNotFoundException:
        return Response(content="No face detected!", status_code=422)


