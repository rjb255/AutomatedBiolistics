#region basic imports
import pandas as pd
import numpy as np
import math
#endregion
#region ml imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2
#endregion
#region plotting
import matplotlib.pyplot as plt
from matplotlib import colors
#endregion


class image:    
    def __init__(self, size = 100, shape = "circle", **kwargs):
        self.size = size
        self.canvas = np.zeros(size)
        if shape == "circle":
            s = min(self.size)
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    if (i-self.size[0]/2)**2+(j-self.size[1]/2)**2<s**2/4:
                        self.canvas[i,j] = 1
        if shape == "image":
            self.img = cv2.imread(kwargs["link"])
            kernel = np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ])
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            editted = cv2.filter2D(self.gray, -1, kernel)
            self.ret, self.thresh = cv2.threshold(editted,10,255,0)
            self.contours, self.hierarchy = cv2.findContours(255 - self.thresh, 1, 2)
            self.im2 = np.zeros_like(self.gray)
            cv2.drawContours(self.im2, self.contours, -1, 255, thickness = 1)

class model:
    inputShape = [128, 128, 3] #Unknown atm
    model = keras.Sequential([
        #Input
        layers.InputLayer(input_shape = inputShape),
        #Augment
        preprocessing.RandomRotation(factor = 0.1),
        #Layer 1
        layers.BatchNormalization(renorm = True),
        layers.Conv2D(filters=10, kernel_size=3, activation='relu', padding='same'),
        #Layer 2
        layers.BatchNormalization(renorm = True),
        layers.Conv2D(filters=20, kernel_size=3, activation='relu', padding='same'),
        #Head
        layers.BatchNormalization(renorm = True),
        layers.Flatten(),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    optimiser = keras.optimizers.SGD(lr=0.01, nesterov=True)
    model.compile(
        optimizer=optimiser,
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )
    early_stopping = keras.callbacks.EarlyStopping(
                        patience=10,
                        min_delta=0.001,
                        restore_best_weights=True,
                     )
    def __init__(self, **kwargs):
        overallDir = kwargs["dir"]
        ds_train_ = image_dataset_from_directory(
            overallDir +'/train',
            labels='inferred',
            label_mode='binary',
            image_size=[128, 128],
            interpolation='nearest',
            batch_size=64,
            shuffle=True,
        )
        ds_valid_ = image_dataset_from_directory(
            overallDir +'/valid',
            labels='inferred',
            label_mode='binary',
            image_size=[128, 128],
            interpolation='nearest',
            batch_size=64,
            shuffle=False,
        )
        def convert_to_float(image, label):
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image, label

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds_train = (
            ds_train_
            .map(convert_to_float)
            .cache()
            .prefetch(buffer_size=AUTOTUNE)
        )
        ds_valid = (
            ds_valid_
            .map(convert_to_float)
            .cache()
            .prefetch(buffer_size=AUTOTUNE)
        )
        self.his = self.model.fit(
                    ds_train,
                    validation_data=ds_valid,
                    epochs = 50,
                    callbacks=[self.early_stopping],
                   )

def plotImage(image):
    if image.ndim == 2:
        plt.imshow(image, interpolation='nearest')
    else:
        phi = (1 + 5 ** 0.5) / 2
        n = image.shape[0]
        x = (n / phi) ** 0.5
        pdim = [math.ceil(x), math.floor(x * phi)]
        if pdim[0] * pdim[1] < n:
            pdim[1] += 1
        for i in range(n):
            plt.subplot(pdim[0], pdim[1], i + 1)
            plt.imshow(image[i], interpolation='nearest')
    plt.show()