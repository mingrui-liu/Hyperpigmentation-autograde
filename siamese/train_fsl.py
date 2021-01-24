
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

def train_model():

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


    train_ds,val_ds, test_ds,test_labels = generator_fsl.create_generators()

    base_model = create_base_model()
    siamese_network = SiameseNetwork(base_model)

    #siamese_network.save("test.h5")
    lr_schedule  = tfa.optimizers.ExponentialCyclicalLearningRate(
                                  initial_learning_rate=1e-8,
                                  maximal_learning_rate=1e-6,
                                  step_size=240,
                                  )
    opt = Adam(learning_rate=1e-8)

    siamese_network.compile(optimizer= opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy','RootMeanSquaredError'])



    history = siamese_network.fit(train_ds,
          epochs = 100,
          steps_per_epoch = 50,
          validation_data = val_ds,
          validation_steps = 20)

    prediction = siamese_network.predict_classes(test_ds)
    evaluate = siamese_network.evaluate(test_ds,steps= 32)

    return history,evaluate,prediction,test_labels

          
     
def calc_similarity(test_paths,test_labels,df):
  path_test = test_paths[0:1]*100
  label_test = test_labels[0:1]*100

  res = []
  for i in set(df.label):
    paths_train = df[df.label == i].path[0:100]
    labels_train = df[df.label == i].label[0:100]

    temp_ds = create_ds_prediction(path_test,label_test,paths_train,labels_train)
    temp_ds = temp_ds.map(lambda image_label1,image_label2: 
        (image_label1[0],image_label2[0],image_label1[1],image_label2[1]),num_parallel_calls=AUTOTUNE)
    temp_ds = temp_ds.map(lambda image1,image2,label1,label2 :
        (normalize(tf.cast(image1,tf.float32)),normalize(tf.cast(image2,tf.float32)),label1,label2),num_parallel_calls=AUTOTUNE)
    temp_ds = temp_ds.map(lambda image1,image2,label1,label2 :((image1,image2),detect(label1,label2)),num_parallel_calls=AUTOTUNE)
    temp_ds = temp_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    prediction = new_model.predict(temp_ds)
    avg_similarity = sum(prediction)/len(prediction)
    res.append([i,avg_similarity])

  All_res = pd.DataFrame(res, columns = ['class', 'similarity']) 
  return All_res

def prediction(similarity_table):
  return similarity_table.max()[0]

        
