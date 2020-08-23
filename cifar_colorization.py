from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float) 
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def onehot_labels(labels):
    return np.eye(100)[labels]

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='bytes')
    fo.close()
    return dict
  
X_train = get_proper_images(unpickle('drive/My Drive/cifar-100-python/train')[b'data'])
Y_train = onehot_labels(unpickle('drive/My Drive/cifar-100-python/train')[b'fine_labels'])
X_test = get_proper_images(unpickle('drive/My Drive/cifar-100-python/test')[b'data'])
Y_test = onehot_labels(unpickle('drive/My Drive/cifar-100-python/test')[b'fine_labels'])

X=[]
Y=[]
for image in X_train:
  X.append(rgb2lab(1.0/255*image)[:,:,0])
  Y.append(rgb2lab(1.0/255*image)[:,:,1:])
  
X=np.array(X)
Y=np.array(Y)
Y /= 128
X=X.reshape(len(X),32,32,1)
Y=Y.reshape(len(Y),32,32,2)

# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.summary()

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X, 
    y=Y,
    batch_size=1000,
    epochs=10)

X_T=[]
Y_T=[]
for image in X_test:
  X_T.append(rgb2lab(1.0/255*image)[:,:,0])
  Y_T.append(rgb2lab(1.0/255*image)[:,:,1:])

X_T=np.array(X_T)
Y_T=np.array(Y_T)
Y_test /= 128
X_T=X_T.reshape(len(X_T),32,32,1)
Y_T=Y_T.reshape(len(Y_T),32,32,2)

result = model.evaluate(X_T, Y_T, batch_size=1)
print(result)
output = model.predict(X_T)
output *= 128
# Output colorizations
cur = np.zeros((32, 32, 3))
cur[:,:,0] = X_T[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("img_result.png", lab2rgb(cur))
imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))