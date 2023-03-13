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
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
from string import printable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dvclive.keras import DVCLiveCallback
from keras.utils import np_utils, pad_sequences
from sklearn import model_selection

from src.models import Classifier
from src.plot import plot_history, plot_metrics

# from keras.preprocessing.sequence import pad_sequences


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
    df = pd.read_csv(data_csv, encoding="utf-8")  # og: encoding="unicode_escape"
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
def save_model_idetect(model_JSON, file_weights):
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


def save_model(model_file, config):
    """save the trained model as a pickle file"""
    if config["model"]["save"]:
        if config["model"]["name"] != "RF":
            model.save(model_file)
        else:
            pickle.dump(model, open(model_file, "wb"))

    print("The final trained model is saved at: ", model_file)
    print("\n" + "-" * 35 + "Training Completed" + "-" * 35 + "\n")


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


def init_neptune(classfr, epochs, data_file):
    """return neptune init object if neptune is enabled"""
    import neptune

    # from neptune.integrations.tensorflow_keras import NeptuneCallback

    print("\n" + "-" * 30 + "Neptune" + "-" * 30 + "\n")
    exp_tags = [classfr, f"epochs:{epochs}", data_file]

    nt_config = ConfigParser()
    neptune_file = ".neptune.ini"
    print("Reading neptune config file: ", neptune_file)

    nt_config.read(neptune_file)
    project = nt_config["neptune_access"]["project"]
    api_token = nt_config["neptune_access"]["api_token"]

    # put your neptune credentials
    nt_run = neptune.init_run(
        project=project, api_token=api_token, name="IoTvulCode", tags=exp_tags
    )
    # save configuration and module file to neptune.
    nt_run["configurations"].upload("config.yaml")
    nt_run["model_archs"].upload("src/models.py")
    nt_run["code"].upload("src/classifier.py")
    return nt_run


def apply_checkpoints(model, cp_path, patience):
    """apply tf callbacks to store the best model checkpoint and apply early stopping."""
    log_dir = cp_path + "logs/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    model.save_weights(cp_path + "pre-fit.weights")
    tf_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=patience,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_path + "checkpoint_model.h5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]
    return tf_callbacks


def train_model(classfr):
    """Choose ML model"""
    mdl_obj = Classifier(config)

    if classfr == "RNN":
        model = mdl_obj.apply_RNN()
    elif classfr == "CNN":
        model = mdl_obj.apply_CNN()
    elif classfr == "LSTM":
        model = mdl_obj.apply_LSTM()
    elif classfr == "RF":
        model = mdl_obj.apply_RF(df)
    elif classfr == "multiDNN":
        model = mdl_obj.apply_multiDNN()
    else:
        print("Invalid model! Please select any valid model!")
        exit(1)
    return model


def gen_model_dir(config, clr, epochs):
    """save our current model state so we can reload it later"""
    mdir = clr + "-" + epochs + "-" + Path(data_file).stem + "/"
    config["model"]["path"] = config["model"]["path"] + mdir

    if config["debug"]:
        config["model"]["path"] = config["model"]["path"].rsplit("/", 1)[0] + "-debug/"
        config["model"]["use_neptune"] = (
            False if config["debug"] == True else config["model"]["use_neptune"]
        )
        config["dnn"]["epochs"] = 2

    if config["train"]:
        Path(config["model"]["path"]).mkdir(parents=True, exist_ok=True)
    print(f"\n\nModel path: {config['model']['path']}")
    return config


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
    epochs = config["dnn"]["epochs"] if config["debug"] != True else 2
    batch_size = config["dnn"]["batch"]
    classfr = paras.model if paras.model else config["model"]["name"]
    max_len = config["preprocess"]["max_len"]  # for pad_sequences

    # create a dynamic path to save the trained model
    config = gen_model_dir(config, clr=classfr, epochs=str(epochs))

    # Display settings
    print("\n\n" + "=" * 25 + " " + classfr + " Model Training " + "=" * 25)
    print(f"Configurations: \n {config}")
    print("-" * 50)

    # Load input data
    df = load_data(data_csv=data_csv)
    X, target = tokenize_data(df)

    # Split the data set into training and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, target, test_size=test_size, random_state=seed
    )

    # get the train model
    model = train_model(classfr)

    # store metadata to neptune.ai
    if config["model"]["use_neptune"]:
        from neptune.integrations.tensorflow_keras import NeptuneCallback

        nt_run = init_neptune(classfr, epochs, data_csv)

    # applying callbacks
    if classfr != "RF":
        tf_callbacks = apply_checkpoints(
            model=model,
            cp_path=config["model"]["path"],
            patience=config["dnn"]["patience"],
        )
        if config["model"]["use_neptune"]:
            neptune_cbk = NeptuneCallback(run=nt_run, base_namespace="metrics")
            tf_callbacks.append(neptune_cbk)

        # Fitting model and Cross-Validation
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            # callbacks=[tf_callbacks],
            callbacks=[DVCLiveCallback(save_dvc_exp=True)],
        )
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        # plot_metrics(history)
        fig_name = config["model"]["path"] + classfr
        plot_history(history, fig_name)
        print(f"\nAccuracy of the model: {accuracy}\n")

        # save tracked files
        if config["model"]["use_neptune"]:
            nt_run["learning_curves"].track_files(fig_name + ".pdf")
            nt_run["loss_curve"].track_files(fig_name + "_loss.pdf")

        model_file = config["model"]["path"] + "model-final.h5"
        if config["train"]:
            save_model(model_file, config)
    else:
        # TODO: log non-DNN models output to neptune
        # nt_run["acc"] = ?? or params=dict
        print(f"Trained with non-DNN model: {classfr}")

    # under construction
    if config["test"]:
        _, _, X_test, y_train, _, y_test = split_data(df, config)
        output_size = len(set(list(y_train)))

        if config["model"]["name"] != "RF":
            test_model(model_file, X_test, y_test, output_size)
        else:
            loaded_model = pickle.load(open(model_file, "RF"))
            result = loaded_model.score(X_test, y_test)
            print("Result: ", result)
        print("\n" + "-" * 35 + "Testing Completed" + "-" * 35 + "\n")
