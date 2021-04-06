import cv2
import keras
from keras.layers import *
from keras.models import load_model
import torch
import cv2
from PIL import Image

#phase 1
model_segmentation = load_model("")
input_1  = cv2.imread("")
output_1 = model_segmentation.predict(input_1)

#phase 2 
  
