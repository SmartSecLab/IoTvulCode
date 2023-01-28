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

#CNN Model for Binary Classification

def CNN(max_len=150, emb_dim=32, max_vocab_len=150, W_reg=regularizers.l2(1e-4)):
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
    
    # Compile model and define optimizer
    if int(keras.__version__.split('.')[0])<2:
        model = Model(input=[main_input], output=[output]) 
    else:
        model = Model(inputs=[main_input], outputs=[output])
        
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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