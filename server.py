from __future__ import print_function
from __future__ import division
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import urllib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

app = Flask(__name__)
from PIL import Image
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/status', methods=['GET'])
def returnStatus():
    return jsonify({"status":"ready to detect covid!"})

@app.route('/api/predict', methods=['POST'])
def predictCov():
    data = request.get_json()
    url, filename = (data['url'], "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 

    # if torch.cuda.is_available():
        # input_batch = input_batch.to('cuda')
        # model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)
    index = probabilities.argmax().item()
    return jsonify({'result':index})

if __name__ == '__main__':
    modelfile = '../SavedModels/covBinarySqueezeNet.pth'
    model = torch.load(modelfile, map_location=torch.device('cpu'))
    app.run(debug=True, host='0.0.0.0')