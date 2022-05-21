#encoding:utf-8
'''
detect and segement potential nuclei in miscropic images (H&E stained)
@author: Kemeng Chen 
'''
import os
from unittest import result
import numpy as np 
from skimage.transform import resize
import cv2
from time import time
from util import*
from tensorflow import keras
import matplotlib.pyplot as plt


temp_image=cv2.imread("data/sample_2/sample_2.png")


X_test = np.zeros(
    (
        1,
        128,
        128,
        3
    ),
    dtype=np.uint8
)
temp_image = resize(
    temp_image,
    (128, 128),
    mode='constant',
    preserve_range=True
)
X_test[0] = temp_image
model = keras.models.load_model("models/nuclear_model.h5")
print("preds_test",X_test)
preds_test = model.predict(X_test, verbose=1)
skimage.io.imshow(preds_test[0])
skimage.io.show()

