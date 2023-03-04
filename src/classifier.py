"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT

Project: ENViSEC - Artificial Intelligence-enabled Cybersecurity for Future Smart Environments 
(funded from the European Union’s Horizon 2020, NGI-POINTER under grant agreement No 871528).
@Programmer: Guru Bhandari
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path
from string import printable
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from keras.utils import np_utils, pad_sequences

# from keras_preprocessing.sequence import pad_sequences

from sklearn import model_selection

from src.models import Classifier
from src.plot import plot_metrics

warnings.filterwarnings("ignore")


def load_config(yaml_file):
    """
    load the yaml file and return a dictionary
    """
    with open(yaml_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return exc


def load_data(data_csv):
    """Load data code_snippet"""
    df = pd.read_csv(data_csv, encoding="unicode_escape")
    # Checking for duplicate rows or null values
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"\nShape of the input data: {df.shape}")
    print("Samples:")
    print("-" * 50)
    print(df.head(3))
    print("-" * 50)
    return df.reset_index(drop=True)


def tokenize_data(df):
    """Dataset tokenization"""
    code_snippet_int_tokens = [
        [printable.index(x) + 1 for x in code_snippet if x in printable]
        for code_snippet in df.code
    ]
    # X = sequence.pad_sequences(code_snippet_int_tokens, maxlen=max_len) # original
    X = pad_sequences(code_snippet_int_tokens, maxlen=max_len)
    target = np.array(df.isMalicious)
    print(f"Matrix dimensions of X: {X.shape},\nVector dimension of y:{target.shape}")
    return X, target


# Save model to disk
def save_model(model_JSON, file_weights):
    """Saving model to disk"""
    print("Saving model to disk: ", model_JSON, "and", file_weights)
    # have h5py installed
    if Path(model_JSON).is_file():
        os.remove(model_JSON)
    json_string = model.to_json()
    with open(model_JSON, "w") as f:
        json.dump(json_string, f)
    if Path(file_weights).is_file():
        os.remove(file_weights)
    model.save_weights(file_weights)


# Layer dimensions
def print_layers_dims(model):
    l_layers = model.layers
    # Note None is ALWAYS batch_size
    for i in range(len(l_layers)):
        print(l_layers[i])
        print(
            "Input Shape: ",
            l_layers[i].input_shape,
            "Output Shape: ",
            l_layers[i].output_shape,
        )


# Load model from disk
def load_model(model_JSON, file_weights):
    with open(model_JSON, "r") as f:
        model_json = json.load(f)
        model = model_from_json(model_json)

    model.load_weights(file_weights)
    return model


if __name__ == "__main__":
    # Command Line Arguments:
    parser = ArgumentParser(
        description="AI-enabled IoT Cybersecurity Approach for Vulnerability Detection..."
    )
    parser.add_argument("--model", type=str, help="Name of the ML model to train/test.")
    parser.add_argument("--data", type=str, help="Data file for train/test.")
    paras = parser.parse_args()

    # Config File Arguments:
    config = load_config("config.yaml")
    data_csv = paras.data if paras.data else config["data_file"]
    test_size = config["model"]["split_ratio"]
    seed = config["model"]["seed"]
    data_file = config["data_file"]
    epochs = config["dnn"]["epochs"]
    batch_size = config["dnn"]["batch"]
    CLASS_MODEL = paras.model if paras.model else config["model"]["name"]
    max_len = config["preprocess"]["max_len"]  # for pad_sequences

    # Display settings
    print("\n\n" + "=" * 25 + " " + CLASS_MODEL + " Model Training " + "=" * 25)
    print(f"Configurations: \n {config}")
    print("-" * 50)

    # Load input data
    df = load_data(data_csv=data_csv)
    X, target = tokenize_data(df)

    # Split the data set into training and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, target, test_size=test_size, random_state=seed
    )

    # Choose ML model
    mdl_obj = Classifier(config)
    if CLASS_MODEL == "RNN":
        model = mdl_obj.apply_RNN()
    elif CLASS_MODEL == "CNN":
        model = mdl_obj.apply_CNN()
    elif CLASS_MODEL == "LSTM":
        model = mdl_obj.apply_LSTM()
    elif CLASS_MODEL == "RF":
        model = mdl_obj.apply_RF(df)
    elif CLASS_MODEL == "multiDNN":
        model = mdl_obj.apply_multiDNN()
    else:
        print("Invalid Model! Please select any valid model!")
        exit(1)

    # Fitting model and Cross-Validation
    if CLASS_MODEL == "RF":
        # Evaluation of the training model
        print("-" * 50)
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
        )
        # print(history)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        # print('\nTesting Accuracy =', accuracy, '\n')
        plot_metrics(history)
        print("\nFinal Cross-Validation Accuracy of RNN training model", accuracy, "\n")
