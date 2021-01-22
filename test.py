#region basic imports
import pandas as pd
import numpy as np
#endregion
#region ml imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
import cv2
#endregion
#region plotting
import matplotlib.pyplot as plt
from matplotlib import colors
#endregion
#region custom imports
import imageClass as ic
#endregion

image = ic.image(shape = "image", link = "download.jpg")

print(image.canvas)

for i in image.hierarchy[0][:,1]:
    a = np.zeros_like(image.img)
    b = np.zeros_like(image.img) + image.img
    c = np.zeros_like(image.img)
    cv2.drawContours(a, image.contours, i+1, 255, thickness = -1)
    cv2.drawContours(b, image.contours, i+1, 255, thickness = 1)
    c[a == 255] = image.img[a == 255]

    (y, x) = np.where(a == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    c = c[topy:bottomy+1, topx:bottomx+1]

    ic.plotImage(np.array([a, b, c]))