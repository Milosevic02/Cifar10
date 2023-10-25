import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Conv2D,Dense,Flatten,Dropout,GlobalMaxPooling2D,MaxPooling2D,BatchNormalization
from keras.models import Model

cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train,x_test = x_train/255.0 , x_test /255.0
y_train,y_test = y_train.flatten(),y_test.flatten()
print("x_train.shape: " , x_train.shape)
print("y_train.shape",y_train.shape)