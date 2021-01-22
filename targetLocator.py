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
class image:
    
    def __init__(self, size, shape = "circle", **kwargs):
        self.size = size
        self.canvas = np.zeros(size)
        if shape == "circle":
            s = min(self.size)
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    if (i-self.size[0]/2)**2+(j-self.size[1]/2)**2<s**2/4:
                        self.canvas[i,j] = 1
        if shape == "image":
            img = cv2.imread(kwargs["link"],0)
            ret,thresh = cv2.threshold(img,127,255,0)
            self.contours,self.hierarchy = cv2.findContours(thresh, 1, 2)

#region basic ml settings
model = keras.Sequential([
    layers.InputLayer(input_shape = [128,128,3]), #unknown input_shape for now
    
    # Data Augmentation
    preprocessing.RandomRotation(factor = 0.1),

    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Two
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
#endregion
pic = image((100,100))

#region Define Plot in Matlibplot
rows,cols = pic.canvas.shape
plt.subplot(2, 2, 1)
plt.imshow(pic.canvas, interpolation='nearest', 
                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows],
                 cmap='bwr')
plt.subplot(2, 2, 2)
#endregion

kernel = np.array([
                    [-1, -1, -1],
                    [-1, 6, -1],
                    [-1, -1, -1],
                    
                ])
#region Matlibplot Stuff again
rows,cols = kernel.shape

plt.imshow(kernel, interpolation='nearest', 
                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows],
                 cmap='bwr')
#endregion


#region old

im = tf.cast(pic.canvas, dtype=tf.float32)
im = tf.reshape(im, [1, *im.shape, 1])
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)
image_filter = tf.nn.conv2d(
    input=im,
    filters=kernel,
    strides=1,
    padding='VALID',
)
image_detect = tf.nn.relu(image_filter)
plt.subplot(2, 2, 3)
im = tf.squeeze(image_filter).numpy()
rows,cols = im.shape
plt.imshow(im, interpolation='nearest', 
                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows],
                 cmap='bwr')

plt.subplot(2, 2, 4)

im = tf.squeeze(image_detect).numpy()
rows,cols = im.shape

plt.imshow(im, interpolation='nearest', 
                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows],
                 cmap='bwr')
plt.show()

#endregion