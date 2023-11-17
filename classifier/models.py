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
import pickle
import re
import warnings
from pathlib import Path
from string import printable

import joblib
# import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skops.io as sio
import tensorflow as tf
from matplotlib import pyplot
# import tensorflow_addons as tfa
from nltk.tokenize.regexp import WhitespaceTokenizer
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.layers import (ELU, LSTM, Activation, BatchNormalization,
                                     Bidirectional, Convolution1D,
                                     Convolution2D, Dense, Dropout, Embedding,
                                     Flatten, Input, Lambda, MaxPooling1D,
                                     MaxPooling2D, SimpleRNN, concatenate)
from tensorflow.keras.models import (Model, Sequential, load_model,
                                     model_from_json)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing import sequence


# from keras_metrics import metrics as km


warnings.filterwarnings("ignore")


class ModelArchs:
    def __init__(self, config):
        """Initialize the class with available settings
        Args:
            config (_dict_): configuration settings
        """
        self.config = config

        # function arguments:
        self.max_len = self.config["preprocess"]["max_len"]
        self.classify_type = str(self.config['model']['type']).lower()

        # DNN arguments:
        self.input_length = self.config["dnn"]["input_length"]
        self.input_dim = self.config["dnn"]["input_dim"]
        self.emb_dim = self.config["dnn"]["output_dim"]
        self.max_vocab_len = self.config["preprocess"]["max_vocab_len"]
        self.dropout = self.config["dnn"]["dropout"]
        self.recur_dropout = self.config["dnn"]["recur_dropout"]

        # Optimizer arguments:
        self.optimizer = self.config["dnn"]["optimizer"]  # adam
        self.learn_rate = self.config["dnn"]["lr"]
        self.beta_1 = self.config["dnn"]["beta_1"]
        self.beta_2 = self.config["dnn"]["beta_2"]
        self.epsilon = float(self.config["dnn"]["epsilon"])
        self.decay = self.config["dnn"]["decay"]

        if self.classify_type == 'binary':
            self.loss = self.config["dnn"]["loss_binary"]
            self.activ_last_layer = 'sigmoid'

        elif self.classify_type == 'multiclass':
            self.loss = self.config["dnn"]["loss_multiclass"]
            self.activ_last_layer = 'softmax'
        else:
            raise ValueError(
                f"Invalid classification type: {self.classify_type}")

        self.output_dim = self.config["dnn"]["output_dim"]

        print(f'\n\nConfigurations: {self.config}')
        # Metrics
        self.metrics = [
            "acc",
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC(),
        ]
        print(f'\nPerformance metrics: {self.metrics}')
        print("-" * 50)

    def optimize_model(self, model):
        """apply optimizer"""
        # model = Model(inputs=main_input, outputs=emb_layer)
        optim = Adam(
            learning_rate=1e-4,
            # beta_1=self.beta_1,
            # beta_2=self.beta_2,
            # epsilon=self.epsilon,
            # decay=self.decay,  # deprecated from Keras 2.3
        )
        # model.compile(optimizer=optim, loss=self.loss, metrics=self.metrics)
        model.compile(optimizer=optim, loss=self.loss, metrics=self.metrics)
        print(f"\n {model.summary()}")
        return model

    def apply_RNN(self):
        """
        RNN Model for Binary and Multiclass Classification
        """
        # Main Input
        # main_input = Input(shape=(self.max_len,), dtype="int32")

        main_input = Input(shape=(self.max_len,))

        # Embedding Layers
        # emb_layer = Embedding(input_dim=150, output_dim=32, input_length=150,
        # W_regularizer=regularizers.l2(1e-4))(main_input)  # original license
        emb_layer = Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_length=self.input_length,
        )(main_input)

        emb_layer = Bidirectional(
            SimpleRNN(
                self.input_dim,
                return_sequences=False,
                dropout=self.dropout,
                recurrent_dropout=self.recur_dropout,
            )
        )(emb_layer)
        # <guru> I think the activation function should be 'sigmoid' here
        # for binary classification???
        # emb_layer = Dense(55, activation="softmax")(
        #     emb_layer)  # iDetech original - static

        # initializer = tf.keras.initializers.RandomNormal(
        #     mean=0.0, stddev=0.05, seed=None)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        emb_layer = Dense(
            units=self.input_dim/2,
            activation="relu",
            kernel_initializer=initializer)(emb_layer, )

        emb_layer = Dense(self.input_dim/4, activation="tanh")(emb_layer)

        # output layer
        emb_layer = Dense(
            self.output_dim,
            activation=self.activ_last_layer,
        )(emb_layer)

        # apply RNN model settings
        model = Model(inputs=main_input, outputs=emb_layer)

        # apply optimizer
        model = self.optimize_model(model)
        return model

    # define apply_funRNN function
    def apply_funRNN(self, vocab_size: int, embedding_matrix, MAX_LEN: int):
        """Define the RNN model"""
        model = Sequential()
        model.add(Embedding(
            vocab_size, MAX_LEN,
            weights=[embedding_matrix],
            input_length=MAX_LEN,
            trainable=False))
        model.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.output_dim,
                  activation=self.activ_last_layer,))
        model = self.optimize_model(model)
        return model

    # baseline
    def apply_DNN(self):
        """
        DNN Model for Binary and Multiclass Classification
        """
        # create model
        model = Sequential()
        model.add(Dense(self.max_len, input_shape=(
            self.max_len,), activation='sigmoid'))
        model.add(Dense(self.max_len/4,  activation='sigmoid'))
        model.add(Dropout(0.0002))

        # output layer
        print(f'Output dim from model: {self.output_dim}')
        model.add(
            Dense(self.output_dim,
                  activation=self.activ_last_layer))

        # Compile model
        sgd = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(loss=self.loss,
                      optimizer=sgd,
                      metrics=['acc']
                      )
        return model

    def apply_CNN(self):
        """
        CNN Model for Binary/Multi_Class Classification
        Training Model 2 - 1D Convolutions and Fully Connected Layers
        """
        # Input
        main_input = Input(shape=(self.max_len,),
                           dtype="int32", name="main_input")

        # Embedding layer
        # emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
        #             W_regularizer=W_reg)(main_input)  # original with W_regularizer
        emb = Embedding(
            input_dim=self.max_vocab_len,
            output_dim=self.emb_dim,
            input_length=self.max_len,
        )(main_input)
        emb = Dropout(0.25)(emb)

        def sum_1d(X):
            return K.sum(X, axis=1)

        def get_conv_layer(emb, kernel_size=5, filters=150):
            # Conv layer
            # conv = Convolution1D(kernel_size=kernel_size, filters=filters, \
            #              border_mode='same')(emb)  # Original with tf1.5 now 'boarder_mode' is 'padding'
            conv = Convolution1D(
                kernel_size=kernel_size, filters=filters, padding="same"
            )(emb)
            conv = ELU()(conv)

            conv = Lambda(sum_1d, output_shape=(filters,))(conv)
            # conv = BatchNormalization(mode=0)(conv)
            conv = Dropout(0.5)(conv)
            return conv

        # Multiple Conv Layers
        # calling custom conv function from above
        conv1 = get_conv_layer(emb, kernel_size=2, filters=150)
        conv2 = get_conv_layer(emb, kernel_size=3, filters=150)
        conv3 = get_conv_layer(emb, kernel_size=4, filters=150)
        conv4 = get_conv_layer(emb, kernel_size=5, filters=150)

        # Fully Connected Layers
        merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        hidden1 = Dense(self.input_dim)(merged)
        hidden1 = ELU()(hidden1)

        if int(keras.__version__.split(".")[0]) < 2:
            hidden1 = BatchNormalization(mode=0)(hidden1)
        else:
            hidden1 = BatchNormalization()(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(self.input_dim)(hidden1)
        hidden2 = ELU()(hidden2)
        # hidden2 = BatchNormalization(mode=0)(hidden2)
        if int(keras.__version__.split(".")[0]) < 2:
            hidden2 = BatchNormalization(mode=0)(hidden2)
        else:
            hidden2 = BatchNormalization()(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        # Output layer (last fully connected layer)
        hidden2 = Dense(self.input_dim/4, activation="softmax",
                        name="output")(hidden2)

        # output layer

        output = Dense(self.output_dim,
                       activation=self.activ_last_layer)(hidden2)

        # Compile model
        model = Model(inputs=[main_input], outputs=[output])

        # apply optimizer
        model = self.optimize_model(model)
        return model

    def apply_funCNN(self, vocab_size: int, embedding_matrix, MAX_LEN: int):
        """Define the CNN model"""
        model = Sequential()
        model.add(Embedding(
            vocab_size, MAX_LEN,
            weights=[embedding_matrix],
            input_length=MAX_LEN,
            trainable=False))
        model.add(Convolution1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Convolution1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.output_dim,
                  activation=self.activ_last_layer,))
        model = self.optimize_model(model)
        return model

    def apply_LSTM(self):
        """
        multi-layer DNN model for the training
        """
        model = Sequential()
        # First LSTM layer defining the input sequence length
        model.add(
            LSTM(input_shape=(self.input_dim, 1),
                 units=32, return_sequences=True)
        )
        model.add(Dropout(self.dropout))

        # Second LSTM layer with 128 units
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(self.dropout))

        # Third LSTM layer with 100 units
        model.add(LSTM(units=150, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.output_dim, activation="softmax"))
        # output layer
        model.add(Dense(self.output_dim,
                  activation=self.activ_last_layer,))
        # apply optimizer
        model = self.optimize_model(model)
        return model

    def apply_funLSTM(self, vocab_size: int, embedding_matrix, MAX_LEN: int):
        """Define the LSTM model"""
        model = Sequential()
        model.add(Embedding(
            vocab_size, MAX_LEN,
            weights=[embedding_matrix],
            input_length=MAX_LEN,
            trainable=False))
        model.add(LSTM(128, dropout=0.2, recurrent_dropouts=0.2))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.output_dim,
                  activation=self.activ_last_layer,))
        model = self.optimize_model(model)
        return model

    def apply_multiDNN(self):
        """multi-layer DNN model for the training"""
        model = Sequential()
        model.add(Dense(2000, activation="relu", input_dim=self.input_dim))
        model.add(Dense(1500, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(800, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(400, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(150, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(self.output_dim, activation="softmax"))
        # output layer
        model.add(Dense(self.output_dim,
                        activation=self.activ_last_layer,))

        # apply optimizer
        model = self.optimize_model(model)
        return model

    def apply_RF(self, input_data):
        """Defining the Training Model Classifier for Binary Classification"""
        def preprocess4RF(input_data):
            """Cleaning-up"""
            return (
                pd.Series(input_data)
                .replace(r"\b([A-Za-z])\1+\b", "", regex=True)
                .replace(r"\b[A-Za-z]\b", "", regex=True)
            )
        transformer = FunctionTransformer(preprocess4RF)

        token_pattern = r"""([A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]"'`])"""
        vectorizer = TfidfVectorizer(
            token_pattern=token_pattern, max_features=3000)

        # Training Model Classifier for Multi-Class Classification
        clf = RandomForestClassifier(n_jobs=4)

        model = Pipeline(
            [("preprocessing", transformer),
             ("vectorizer", vectorizer), ("clf", clf)]
        )

        # Setting of the best parameters
        best_params = {
            "clf__criterion": "gini",
            "clf__max_features": "sqrt",
            "clf__min_samples_split": 3,
            "clf__n_estimators": 300,
        }
        model.set_params(**best_params)
        return model
