import keras
from keras.models import load_model
from skimage.color import lab2rgb, rgb2lab
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from math import ceil

class ImageColorization:
    def __init__(self):
        self.SIZE = 256

    def visualize(self, imgs):
        _, axs = plt.subplots(ceil(len(imgs)/2), 2, figsize=(12, 12))
        axs = axs.flatten()

        for img, ax in zip(imgs, axs):
            ax.imshow(img)

    def generateData(self, folder="ImageProcessing\mirflickr25k", 
                     batch=100, size=256):

        dataset = ImageDataGenerator(rescale=1./255)
        dataset = dataset.flow_from_directory(folder, target_size=(size, size),
                                              batch_size=batch, class_mode=None)
        trainX = []
        trainY = []

        for img in dataset[0]:
            npImg = np.array(img)
            LABimg = rgb2lab(npImg)
            
            trainX.append(LABimg[:, :, 0])
            trainY.append(LABimg[:, :, 1:]/128.0)

        trainX, trainY = np.array(trainX), np.array(trainY)
        trainX = trainX.reshape(trainX.shape+(1,))
        print("Added {} images".format(len(trainX)))
        return trainX, trainY

    def addModel(self):
        print("Adding Model...")
        # make sequential model
        model = Sequential([
                Conv2D(64, (3, 3), activation="relu", padding="same", strides=2),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                Conv2D(128, (3, 3), activation="relu", padding="same", strides=2),
                Conv2D(256, (3, 3), activation="relu", padding="same"),
                Conv2D(256, (3, 3), activation="relu", padding="same", strides=2),
                Conv2D(512, (3, 3), activation="relu", padding="same"),
                Conv2D(512, (3, 3), activation="relu", padding="same"),
                Conv2D(256, (3, 3), activation="relu", padding="same"),

                Conv2D(128, (3, 3), activation="relu", padding="same"),
                UpSampling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                UpSampling2D((2, 2)),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                Conv2D(16, (3, 3), activation="relu", padding="same"),
                Conv2D(2, (3, 3), activation="tanh", padding="same"),
                UpSampling2D((2, 2))
            ])

        model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        return model

    def train(self, x: np.array, y: np.array, batch=2, epochs=10):
        modelFile = "Model/ImageColorization"

        model = load_model(modelFile) if os.path.exists(modelFile) \
                                      else self.addModel()
        
        model.fit(x, y, batch_size=batch, epochs=epochs)
        model.save(modelFile)
        return model

    def test(self, img):
        img = self.readFile(img) if type(img) == str \
                                 else img
        img = cv2.resize(img, (self.SIZE, self.SIZE))

        gray = rgb2lab(img)[:, :, 0]
        grayTest = np.array([gray.shape + (1,)])

        model = load_model("Model/ImageColorization")
        prediction = model.predict(grayTest)
        prediction = prediction*128

        LABimg = cv2.merge((grayTest[0], prediction[0]))
        RGBimg = lab2rgb(LABimg)

        self.visualize([img, RGBimg])
    
    def readFile(self, filePath: str) -> np.array: 
        img = cv2.imread(filePath)
        return np.array(img)
