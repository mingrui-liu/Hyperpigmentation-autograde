
import cv2
import math
import shutil
import random

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model, load_model,save_model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import SGD,Adam,RMSprop,Nadam

import tensorflow_addons as tfa


import constants
import generator


conv_base = ResNet50(include_top = False, weights = 'imagenet',
                      input_shape = (224, 224, 3))


#conv_base.trainable = False
x = conv_base.output
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
predictions = tf.keras.layers.Dense(7, activation='softmax')(x)


model = Model(inputs = conv_base.input, outputs=predictions)

lr_schedule  = tfa.optimizers.ExponentialCyclicalLearningRate(
                              initial_learning_rate=1e-8,
                              maximal_learning_rate=1e-4,
                              step_size=240,
                              )
opt = Adam(learning_rate=lr_schedule)

model.compile(optimizer= opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_ds,test_ds = generator.create_generators()
model.fit(train_ds,
      epochs = 200,
      steps_per_epoch = 40,
      validation_data = test_ds)



    
