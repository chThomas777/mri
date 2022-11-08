import flask
import io
import string
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
from PIL import Image
import albumentations as A
from flask import Flask, jsonify, request

params = {
    'image_size': 256,
    'batch_size': 32,
    'model': 'resnet50d',
    'output_dim':4,
    'num_workers': 4,
    'epochs': 5,
    "device": 'cuda'
}

class MRIModel(nn.Module):
  def __init__(self, model_name=params['model'], out_dim=params['output_dim'], pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        for param in self.model.parameters():
            param.require_grad = False
        n_features = self.model.fc.in_features 
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Linear(n_features, out_dim)
        self.softmax = nn.Softmax()

  def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return self.softmax(output)

model = torch.load('mri.pth', map_location ='cpu')
model.eval()

target_label = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}

transform = A.Compose(
        [A.Resize(300,300),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def prepare_img(img):
    im = Image.open(io.BytesIO(img))
    im = np.array(im)
    im = transform(image=im)['image']
    im = torch.tensor(im/255).to('cpu').float()
    im = im.unsqueeze(0).permute(0, 3, 1, 2)
    return im


def preds(x, model):
    with torch.no_grad():
        model.eval()
        prediction = model(x)
    prediction = target_label[int(torch.argmax(prediction, 1))]
    return prediction


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_img(img_bytes)

    return jsonify(prediction=preds(img, model))


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
