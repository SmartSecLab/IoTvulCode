"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT

Project: ENViSEC - Artificial Intelligence-enabled Cybersecurity for Future Smart Environments 
(funded from the European Union’s Horizon 2020, NGI-POINTER under grant agreement No 871528).
@Developer: Guru Bhandari
"""

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
import keras
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
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")

## Training Model 2 - 1D Convolutions and Fully Connected Layers

# <Guru> Under Construction: implementation of class to share common variables in the functions.
# class MyClass(object):
#     def __init__(self):
#         self.a = ['A','X','R','N','L']  # Shared instance member :D


# class ApplyDNN:
#     def __init__(self, config):
#         # function arguments:
#         self.max_len = config['preprocess']['max_len'] 
#         self.input_length = config['dnn']['input_length']
#         self.input_dim = config['dnn']['input_dim']
#         self.output_dim = config['dnn']['output_dim']
#         self.emb_dim = config['dnn']['output_dim']
#         self.max_vocab_len = config['preprocess']['max_vocab_len']
#         self.dropout = config['dnn']['dropout']
#         self.recur_dropout = config['dnn']['recur_dropout']

#         # Optimizer arguments:
#         self.learn_rate = float(config['dnn']['lr'])
#         self.beta_1 = config['dnn']['beta_1']
#         self.beta_2 = config['dnn']['beta_2']
#         self.epsilon = float(config['dnn']['epsilon'])
#         self.decay = config['dnn']['decay']
#         self.loss = config['dnn']['loss']
    
#         # Metrics
#         self.metrics = ['acc']
#         print(self.loss)
        
    def RNN(config):
        """
        RNN Model for Binary Classification
        <guru> both Binary and Multi-Class codes are identical, I don't know why author making confusion making 
        multiple replications without their distinct applications.
        """

        # function arguments:
        max_len = config['preprocess']['max_len'] 
        input_length = config['dnn']['input_length']
        input_dim = config['dnn']['input_dim']
        output_dim = config['dnn']['output_dim']
        emb_dim = config['dnn']['output_dim']
        max_vocab_len = config['preprocess']['max_vocab_len']
        dropout = config['dnn']['dropout']
        recur_dropout = config['dnn']['recur_dropout']
        
        # Optimizer arguments:
        learn_rate = float(config['dnn']['lr'])
        beta_1 = config['dnn']['beta_1']
        beta_2 = config['dnn']['beta_2']
        epsilon = float(config['dnn']['epsilon'])
        decay = config['dnn']['decay']
        loss = config['dnn']['loss']
        
        # Metrics
        metrics = ['acc']
        
        # Main Input
        main_input = Input(shape=(max_len,), dtype='int32')

        # Embedding Layers
        # Emb_Layer = Embedding(input_dim=150, output_dim=32, input_length=150, 
        # W_regularizer=regularizers.l2(1e-4))(main_input)  # original license
        Emb_Layer = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length)(main_input) 
        Emb_Layer = Bidirectional(SimpleRNN(input_dim, 
                                            return_sequences=False, 
                                            dropout=dropout, 
                                            recurrent_dropout=recur_dropout))(Emb_Layer)
        # <guru> I think the activation function should be 'sigmoid' here for binary classification???
        Emb_Layer = Dense(55, activation='softmax')(Emb_Layer)  
        # RNN Model Settings
        model = Model(inputs=main_input, outputs=Emb_Layer)
        adam = Adam(lr=learn_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
        model.compile(optimizer=adam, loss=loss, metrics=metrics)
        print(model.summary())
        return model
    
    
def CNN(config):
    """
    CNN Model for Binary Classification
    """
    # function arguments:
    max_len = config['preprocess']['max_len'] 
    input_length = config['dnn']['input_length']
    input_dim = config['dnn']['input_dim']
    output_dim = config['dnn']['output_dim']
    emb_dim = config['dnn']['output_dim']
    max_vocab_len = config['preprocess']['max_vocab_len']
    dropout = config['dnn']['dropout']
    recur_dropout = config['dnn']['recur_dropout']
    
    # Optimizer arguments:
    learn_rate = float(config['dnn']['lr'])
    beta_1 = config['dnn']['beta_1']
    beta_2 = config['dnn']['beta_2']
    epsilon = float(config['dnn']['epsilon'])
    decay = config['dnn']['decay']
    loss = config['dnn']['loss']
    l2_reg = float(config['dnn']['l2_reg'])  #<Guru> TBD: apply this on tf2/keras2 version.
    
    # Metrics
    metrics = ['acc']
    
    # Input
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    # Embedding layer
    # emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
    #             W_regularizer=W_reg)(main_input)  # original with W_regularizer
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len)(main_input) # 
    emb = Dropout(0.25)(emb)

    
    def sum_1d(X):
        return K.sum(X, axis=1)
    
    def get_conv_layer(emb, kernel_size=5, filters=150):
        # Conv layer
        # conv = Convolution1D(kernel_size=kernel_size, filters=filters, \
        #              border_mode='same')(emb)  # Original with tf1.5 now 'boarder_mode' is 'padding'
        conv = Convolution1D(kernel_size=kernel_size, filters=filters, padding='same')(emb)
        conv = ELU()(conv)

        conv = Lambda(sum_1d, output_shape=(filters,))(conv)
        #conv = BatchNormalization(mode=0)(conv)
        conv = Dropout(0.5)(conv)
        return conv
    
    # Multiple Conv Layers
    # calling custom conv function from above
    conv1 = get_conv_layer(emb, kernel_size=2, filters=150)
    conv2 = get_conv_layer(emb, kernel_size=3, filters=150)
    conv3 = get_conv_layer(emb, kernel_size=4, filters=150)
    conv4 = get_conv_layer(emb, kernel_size=5, filters=150)

    # Fully Connected Layers
    merged = concatenate([conv1,conv2,conv3,conv4], axis=1)

    hidden1 = Dense(1024)(merged)
    hidden1 = ELU()(hidden1)
    if int(keras.__version__.split('.')[0])<2:
        hidden1 = BatchNormalization(mode=0)(hidden1)
    else:
        hidden1 = BatchNormalization()(hidden1)
    hidden1 = Dropout(0.5)(hidden1)

    hidden2 = Dense(1024)(hidden1)
    hidden2 = ELU()(hidden2)
    # hidden2 = BatchNormalization(mode=0)(hidden2)
    if int(keras.__version__.split('.')[0])<2:
        hidden2 = BatchNormalization(mode=0)(hidden2)
    else:
        hidden2 = BatchNormalization()(hidden2)
    hidden2 = Dropout(0.5)(hidden2)
      
    # Output layer (last fully connected layer)
    output = Dense(55, activation='softmax', name='output')(hidden2)
    
    # Compile model 
    if int(keras.__version__.split('.')[0])<2:
        model = Model(input=[main_input], output=[output]) 
    else:
        model = Model(inputs=[main_input], outputs=[output])
        
    # CNN Model Settings and define optimizer
    model = Model(inputs=main_input, outputs=[output])
    adam = Adam(lr=learn_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(optimizer=adam, loss=loss, metrics=metrics)
    print(model.summary())
    return model



#Defining the Training Model Classifier for Binary Classification
def RF_model():
    clf = RandomForestClassifier(n_jobs=4)

    pipe_RF = Pipeline([
    ('preprocessing', transformer),
        ('vectorizer', vectorizer),
        ('clf', clf)]
    )

    ##Setting of the best parameters
    best_params = {
        'clf__criterion': 'gini',
        'clf__max_features': 'sqrt',
        'clf__min_samples_split': 3,
        'clf__n_estimators': 300
    }
    pipe_RF.set_params(**best_params)

    ## Fitting
    pipe_RF.fit(Code_Snippet_train, Code_Tag_train)

    ##Evaluation of the training model
    print(f'Accuracy: {pipe_RF.score(Code_Snippet_test, Code_Tag_test)}')