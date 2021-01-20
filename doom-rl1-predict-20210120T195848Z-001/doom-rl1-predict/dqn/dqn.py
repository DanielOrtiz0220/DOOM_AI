from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import tensorflow as tf
#tf.python.control_flow_ops = tf

class Network(object):
    @staticmethod    
    def dqn(input_shape, action_size, learning_rate):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(input_shape)))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=512, activation='relu'))
        model.add(Dense(output_dim=action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model