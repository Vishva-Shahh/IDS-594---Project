# -*- coding: utf-8 -*-
"""IDS 594 Flask.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kKZ3s-b72JctWHewifChNgk_gC_QoFb1
"""

# import os
# os.listdir('../input/')
from google.colab import drive
drive.mount('/content/drive')

import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/content/drive/My Drive/IDS 576/Project/Final Project (1)')

from python_utils import *
import time
import matplotlib.pyplot as plt
import cv2 as cv
from math import sqrt 
import pandas as pd
import numpy as np
from torchvision import transforms as tfs
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
import pickle
import random
import argparse
import sys
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import ion, show
import io

import pandas as pd
import flask
import tensorflow as tf

global graph

graph = tf.compat.v1.get_default_graph()

model = torch.load('fer2013_resnet18_model.pkl', map_location=torch.device('cpu'))
model.eval()
app = flask.Flask(__name__)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neural']

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    outputs = np.argmax(outputs.data.numpy())
    return outputs


@app.route("/", methods=["GET","POST"])
def predict():
  if request.method == 'GET':
      return render_template('index.html')

  if request.method == 'POST':
      if 'file' not in request.files:
          flash('No file')
          return redirect(request.url)

      file = request.files['file']
      if file.filename == '':
          flash('No file')
          return redirect(request.url)

      if file and allowed_file(file.filename):
          image = file.read()

          class_name = get_prediction(image)
          return jsonify({'class_name': class_name})
          
if __name__ == '__main__':
  app.run(host='0.0.0.0')