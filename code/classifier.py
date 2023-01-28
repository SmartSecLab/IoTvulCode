import json
import os
import re
import warnings
from pathlib import Path
from string import printable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import (ELU, LSTM, BatchNormalization, Bidirectional,
                          Convolution1D, Convolution2D, Embedding, Input,
                          MaxPooling1D, MaxPooling2D, SimpleRNN, concatenate)
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn import model_selection

warnings.filterwarnings("ignore")


# <guru> Newly added API calls
from keras.utils import pad_sequences
from models import CNN

## Deep Learning: Original Requirements on iDetect:
# - tensorflow 1.4.1
# - keras 2.1.2
# <Guru> but it works with tf2.x 


DATA_CSV = '../data/iDetect_refine/DNN_Multi_Class.csv'
CLASS_MODEL = 'CNN'  # <guru> recheck CNN model, not running.

## Load data code_snippet

Al_Boghdady_Binary_Dataset = pd.read_csv(DATA_CSV, encoding= 'unicode_escape')
Al_Boghdady_Binary_Dataset.sample(n=10).head(5) 


# Checking for duplicate rows or null values
Al_Boghdady_Binary_Dataset.dropna(inplace=True)
Al_Boghdady_Binary_Dataset.drop_duplicates(inplace=True)
Al_Boghdady_Binary_Dataset.sample(n=5).head(5) 
# (4810, 2)


#Dataset tokenization
code_snippet_int_tokens = [[printable.index(x) + 1 for x in code_snippet if x in printable] 
                           for code_snippet in Al_Boghdady_Binary_Dataset.code]
max_len = 150
# X = sequence.pad_sequences(code_snippet_int_tokens, maxlen=max_len) # original
X = pad_sequences(code_snippet_int_tokens, maxlen=max_len)
target = np.array (Al_Boghdady_Binary_Dataset.isMalicious)
print('Matrix dimensions of X: ', X.shape, 'Vector dimension of target: ', target.shape)


#Split the data set into training and test data
X_train, X_test, target_train, target_test = model_selection.train_test_split(X, target, test_size=0.30, random_state=30)



# Layer dimensions
def print_layers_dims(model):
    l_layers = model.layers
    # Note None is ALWAYS batch_size
    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ', l_layers[i].input_shape, 'Output Shape: ', l_layers[i].output_shape)

# Save model to disk
def save_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    #have h5py installed
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)
    

# Load model from disk 
def load_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    with open(fileModelJSON, 'r') as f:
         model_json = json.load(f)
         model = model_from_json(model_json)
    
    model.load_weights(fileWeights)
    return model

if CLASS_MODEL=='RNN':
    #RNN Model for Binary Classification
    # <guru> both Binary and Multi-Class codes are identical, I don't know why author making confusion making 
    # multiple replication without their distinct applications.

    # Main Input
    main_input = Input(shape=(max_len,),dtype='int32')

    # Embedding Layers
    # Emb_Layer = Embedding(input_dim=150, output_dim=32, input_length=150, W_regularizer=regularizers.l2(1e-4))(main_input)  # original license
    Emb_Layer = Embedding(input_dim=150, output_dim=32, input_length=150)(main_input) 
    Emb_Layer = Bidirectional(SimpleRNN(150, return_sequences=False, dropout=0.0, recurrent_dropout=0.0))(Emb_Layer)
    Emb_Layer = Dense(55, activation='softmax')(Emb_Layer)  # <guru> I think the activation fuction should be 'sigmoid' here for binary classification???

    # RNN Model Settings
    RNN_model = Model(inputs=main_input, outputs=Emb_Layer)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    RNN_model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    RNN_model.summary() 


    # Fit model and Cross-Validation
    RNN_history = RNN_model.fit(X_train, target_train, epochs=800, batch_size=64)
    loss, accuracy = RNN_model.evaluate(X_test, target_test, verbose=1)
    print('\nTesting Accuracy =', accuracy, '\n')


    print(RNN_history.history.keys())
    plt.plot(RNN_history.history['acc'])
    #plt.plot(history.history['loss'])
    plt.title('The RNN model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy'], loc='lower right')
    plt.show()

    print('\nFinal Cross-Validation Accuracy of RNN training model', accuracy, '\n')

elif CLASS_MODEL=='CNN':
    model = CNN()
    print(model.summary())
    
    # Fit model and Cross-Validation, Training Model 2 CONV + FULLY CONNECTED
    CNN_model = CNN()
    history = CNN_model.fit(X_train, target_train, epochs=800, batch_size=64)
    loss, accuracy = CNN_model.evaluate(X_test, target_test, verbose=1)

    print(history.history.keys())
    plt.plot(history.history['acc'])
    #plt.plot(history.history['loss'])
    plt.title('The CNN model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy'], loc='lower right')
    plt.show()

    print('\nFinal Cross-Validation Accuracy of CNN training model', accuracy, '\n')

else:
    print('Invalid Model! Please select the valid model!')
    