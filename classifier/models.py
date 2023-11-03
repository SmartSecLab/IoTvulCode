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

        # function arguments:
        self.max_len = config["preprocess"]["max_len"]
        self.classify_type = str(config['model']['type']).lower()

        # DNN arguments:
        self.input_length = config["dnn"]["input_length"]
        self.input_dim = config["dnn"]["input_dim"]
        self.output_dim = config["dnn"]["output_dim"]
        self.emb_dim = config["dnn"]["output_dim"]
        self.max_vocab_len = config["preprocess"]["max_vocab_len"]
        self.dropout = config["dnn"]["dropout"]
        self.recur_dropout = config["dnn"]["recur_dropout"]

        # Optimizer arguments:
        self.optimizer = config["dnn"]["optimizer"]  # adam
        self.learn_rate = config["dnn"]["lr"]
        self.beta_1 = config["dnn"]["beta_1"]
        self.beta_2 = config["dnn"]["beta_2"]
        self.epsilon = float(config["dnn"]["epsilon"])
        self.decay = config["dnn"]["decay"]

        if self.classify_type == 'binary':
            self.loss = config["dnn"]["loss_binary"]
        elif self.classify_type == 'multiclass':
            self.loss = config["dnn"]["loss_multiclass"]
            self.classes_len = 10

        # Metrics
        self.metrics = [
            "acc",
            # tf.keras.metrics.Recall(),
            # tf.keras.metrics.Precision(),
            # tf.keras.metrics.AUC(),
        ]
        print(self.metrics)
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
        RNN Model for Binary Classification
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
        if self.classify_type == 'binary':
            emb_layer = Dense(1, activation="sigmoid",
                              name='RNN-network')(emb_layer)
        else:
            emb_layer = Dense(1, activation="sigmoid",
                              name='RNN-network')(emb_layer)

        # apply RNN model settings
        model = Model(inputs=main_input, outputs=emb_layer)

        # apply optimizer
        model = self.optimize_model(model)
        return model

    # def apply_RNN(self):
    #     inputs = Input(shape=(self.max_len,))
    #     sharable_embedding = Embedding(self.input_dim,
    #                                    output_dim=32,
    #                                    #    weights=[embedding_matrix],
    #                                    input_length=self.max_len,
    #                                    #    trainable=self.embedding_trainable
    #                                    )(inputs)
    #     dense = Flatten()(sharable_embedding)
    #     dense = Dense(self.input_dim,
    #                   activation='tanh')(dense)

    #     # dense = Dense(self.input_dim,
    #     #               activation='tanh')(dense)

    #     # dense = Dense(int(self.input_dim / 2),
    #     #               activation='tanh')(dense)
    #     rnn_out = LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)

    #     dense = Dense(int(self.input_dim / 4))(rnn_out)
    #     dense = Dense(1, activation='sigmoid')(dense)

    #     model = Model(inputs=inputs, outputs=dense, name='RNN_network')
    #     #     # apply optimizer
    #     model = self.optimize_model(model)
    #     return model

    # baseline
    def apply_DNN(self):
        # create model
        model = Sequential()
        model.add(Dense(self.max_len, input_shape=(
            self.max_len,), activation='sigmoid'))
        model.add(Dense(self.max_len/4,  activation='sigmoid'))
        model.add(Dropout(0.0002))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        sgd = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(loss='binary_crossentropy',
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
        if self.classify_type == 'binary':
            output = Dense(1, activation="sigmoid",
                           name='CNN-network')(hidden2)
        else:
            output = Dense(1, activation="softmax",
                           name='CNN-network')(hidden2)

        # Compile model
        model = Model(inputs=[main_input], outputs=[output])

        # # CNN Model Settings and define optimizer #TODO check this
        # model = Model(inputs=main_input, outputs=[emb_layer])

        # apply optimizer
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
        model.add(Dense(1, activation="sigmoid"))
        # apply optimizer
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
        model.add(Dense(1, activation="sigmoid"))

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
