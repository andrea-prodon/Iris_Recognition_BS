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

warnings.filterwarnings("ignore")

app = FastAPI()

app.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class IrisNotFoundException(Exception):
    pass

@app.on_event("startup")
def init_model() -> None:
    app.model = torch.load("model_cnn.pth")



@app.get('/')
def ping() -> str:
    return "ok"


@app.post('/recognition')
def recognition(file: UploadFile):

    # load image
    img = Image.open(file.file)
    np_image = np.array(img)
    try:
        iris, pupil = IrisLocalization(img)
        normalized = IrisNormalization(img, pupil, iris)
        subject = app.model(normalized)
        return JSONResponse(content={"subject": subject}, status_code=200)
    except IrisNotFoundException:
        return Response(content="No face detected!", status_code=422)


