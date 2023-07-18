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

# import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (ELU, LSTM, BatchNormalization, Bidirectional,
                                     Convolution1D, Convolution2D, Embedding, Input,
                                     MaxPooling1D, MaxPooling2D, SimpleRNN, concatenate)
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model, Sequential, load_model, model_from_json
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing import sequence


# from keras_metrics import metrics as km

from matplotlib import pyplot
from nltk.tokenize.regexp import WhitespaceTokenizer
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")


class ModelArchs:
    def __init__(self, config):

        # function arguments:
        self.max_len = config["preprocess"]["max_len"]
        self.input_length = config["dnn"]["input_length"]
        self.input_dim = config["dnn"]["input_dim"]
        self.output_dim = config["dnn"]["output_dim"]

        self.emb_dim = config["dnn"]["output_dim"]
        self.max_vocab_len = config["preprocess"]["max_vocab_len"]
        self.dropout = config["dnn"]["dropout"]
        self.recur_dropout = config["dnn"]["recur_dropout"]

        # Optimizer arguments:
        self.optimizer = config["dnn"]["optimizer"]  # adam
        self.learn_rate = float(config["dnn"]["lr"])
        self.beta_1 = config["dnn"]["beta_1"]
        self.beta_2 = config["dnn"]["beta_2"]
        self.epsilon = float(config["dnn"]["epsilon"])
        self.decay = config["dnn"]["decay"]
        self.loss = config["dnn"]["loss"]

        # Metrics
        self.metrics = [
            "acc",
            # "val_acc",
            # tf.keras.metrics.BinaryAccuracy(),
            # tf.keras.metrics.AUC(),
            # km.recall(),
            # km.f1_score(),
            # km.average_recall(),
        ]
        print(self.metrics)
        print("-" * 50)

    def optimize_model(self, model):
        """apply optimizer"""
        # model = Model(inputs=main_input, outputs=Emb_Layer)
        optim = Adam(
            lr=self.learn_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            decay=self.decay,
        )
        model.compile(optimizer=optim, loss=self.loss, metrics=self.metrics)
        print(f"\n {model.summary()}")
        return model

    def apply_RNN(self):
        """
        RNN Model for Binary Classification
        <guru> both Binary and Multi-Class codes in iDetect are identical,
        I don't know why author is making confusion by making
        multiple replications without their distinct applications.
        """
        # Main Input
        main_input = Input(shape=(self.max_len,), dtype="int32")

        # Embedding Layers
        # Emb_Layer = Embedding(input_dim=150, output_dim=32, input_length=150,
        # W_regularizer=regularizers.l2(1e-4))(main_input)  # original license
        Emb_Layer = Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_length=self.input_length,
        )(main_input)

        Emb_Layer = Bidirectional(
            SimpleRNN(
                self.input_dim,
                return_sequences=False,
                dropout=self.dropout,
                recurrent_dropout=self.recur_dropout,
            )
        )(Emb_Layer)
        # <guru> I think the activation function should be 'sigmoid' here
        # for binary classification???
        Emb_Layer = Dense(55, activation="softmax")(
            Emb_Layer)  # iDetech original
        # Emb_Layer = Dense(2, activation="sigmoid")(Emb_Layer)

        # apply RNN model settings
        model = Model(inputs=main_input, outputs=Emb_Layer)

        # apply optimizer
        model = self.optimize_model(model)
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

        hidden1 = Dense(1024)(merged)
        hidden1 = ELU()(hidden1)
        if int(keras.__version__.split(".")[0]) < 2:
            hidden1 = BatchNormalization(mode=0)(hidden1)
        else:
            hidden1 = BatchNormalization()(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(1024)(hidden1)
        hidden2 = ELU()(hidden2)
        # hidden2 = BatchNormalization(mode=0)(hidden2)
        if int(keras.__version__.split(".")[0]) < 2:
            hidden2 = BatchNormalization(mode=0)(hidden2)
        else:
            hidden2 = BatchNormalization()(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        # Output layer (last fully connected layer)
        output = Dense(55, activation="softmax", name="output")(hidden2)

        # Compile model
        if int(keras.__version__.split(".")[0]) < 2:
            model = Model(input=[main_input], output=[output])
        else:
            model = Model(inputs=[main_input], outputs=[output])

        # # CNN Model Settings and define optimizer #TODO check this
        # model = Model(inputs=main_input, outputs=[Emb_Layer])

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

        # apply optimizer
        model = self.optimize_model(self, model)
        return model

    def apply_RF(self, df):
        """Defining the Training Model Classifier for Binary Classification"""

        code_col = df.code

        def preprocess4RF(code_col):
            """Cleaning-up"""
            return (
                pd.Series(code_col)
                .replace(r"\b([A-Za-z])\1+\b", "", regex=True)
                .replace(r"\b[A-Za-z]\b", "", regex=True)
            )

        transformer = FunctionTransformer(preprocess4RF)
        token_pattern = r"""([A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]"'`])"""
        vectorizer = TfidfVectorizer(
            token_pattern=token_pattern, max_features=3000)

        # Dataset split for training and testing.
        code_train, code_test, tag_train, tag_test = train_test_split(
            df.code,
            df.isMalicious,
            test_size=0.15,
            shuffle=True,  # TODO apply random_state instead shuffle=True for reproducibility
        )

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

        # Fitting
        model.fit(code_train, tag_train)
        acc = model.score(code_test, tag_test)
        print(f"Accuracy: {acc}")
        return model
