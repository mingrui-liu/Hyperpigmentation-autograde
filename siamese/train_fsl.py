
import cv2
import math
import shutil
import random

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model, load_model,save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation,GlobalAveragePooling2D,Lambda,Concatenate,Input,BatchNormalization
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import SGD,Adam,RMSprop,Nadam
from siamese import SiameseNetwork
import matplotlib.pyplot as plt

import tensorflow_addons as tfa


import constants
import generator_fsl

def create_base_model():
    conv_base = ResNet50(include_top = False, weights = 'imagenet',
                          input_shape = (224, 224, 3))


    #conv_base.trainable = False
    x = conv_base.output
    x = tf.keras.layers.Dropout(0.5)(x)
    embedding = GlobalAveragePooling2D()(x)
    embedding = Dense(128)(embedding)    
    return Model(conv_base.input, embedding)

def SiameseNetwork(base_model):
    """
    Create the siamese model structure using the supplied base and head model.
    """
    input_a = Input(shape=(224, 224, 3),name = "image1")
    input_b = Input(shape=(224, 224, 3),name = "image2")

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    head = Concatenate()([processed_a, processed_b])
    head = Dense(1)(head)
    head = Activation(activation='sigmoid')(head)
    return Model([input_a, input_b], head)

base_model = create_base_model()



siamese_network = SiameseNetwork(base_model)

siamese_network.save("test.h5")
lr_schedule  = tfa.optimizers.ExponentialCyclicalLearningRate(
                              initial_learning_rate=1e-8,
                              maximal_learning_rate=1e-4,
                              step_size=240,
                              )
opt = Adam(learning_rate=lr_schedule)

siamese_network.compile(optimizer= opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_ds,test_ds = generator_fsl.create_generators()



history = siamese_network.fit(train_ds,
      epochs = 100,
      steps_per_epoch = 100,
      validation_data = test_ds,
      validation_steps = 20)
      
      
def plot(history):
  
    plt.title('Training and validation accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(history.history['loss'][7:], label='loss')
    plt.plot(history.history['val_loss'][7:], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()
    
plot(history)

    
