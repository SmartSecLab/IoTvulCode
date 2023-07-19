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
# from keras.utils import np_utils, pad_sequences
from sklearn import model_selection
from tensorflow.keras.preprocessing.sequence import pad_sequences

# custom modules
from src.models import ModelArchs
from src.plot import Plotter
from src.utility import Utility

# from keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings("ignore")


class Classifier:
    def __init__(self):
        self.util = Utility()
        # self.config = self.util.config
        self.model = None
        self.history = None

    def load_config(self, yaml_file):
        """
        load the yaml file and return a dictionary
        """
        with open(yaml_file, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                return exc

    def update_config_args(self, config, paras):
        """create model dir to store it
         set the config args from command line if provided """
        config["model"]["name"] = paras.model if paras.model else config["model"]["name"]
        config["data_file"] = paras.data if paras.data else config["data_file"]
        config['dnn']['epochs'] = config['dnn']['debug_epochs'] if config['debug'] else config['dnn']['epochs']

        mdir = config["model"]["name"] + "-" + str(config["dnn"]["epochs"]) + \
            "-" + Path(config["data_file"]).stem + "/"

        config["model"]["path"] = config["model"]["path"] + mdir

        if config["debug"]:
            config["model"]["path"] = config["model"]["path"].rsplit(
                "/", 1)[0] + "-debug/"
            config["model"]["use_neptune"] = (
                False if config["debug"] == True else config["model"]["use_neptune"]
            )

        if config["train"]:
            Path(config["model"]["path"]).mkdir(parents=True, exist_ok=True)
        print(f"\n\nModel path: {config['model']['path']}")
        return config

    def load_data(self, data_csv):
        """Load data code snippets"""
        df = pd.read_csv(
            data_csv, encoding="utf-8")  # og: encoding="unicode_escape"
        # Checking for duplicate rows or null values
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        print(f"\nShape of the input data: {df.shape}")
        print("Samples:")
        print("-" * 50)
        print(df.head(3))
        print("-" * 50)
        return df.reset_index(drop=True)

    def tokenize_data(self, df, max_len):
        """Dataset tokenization"""
        code_snippet_int_tokens = [
            [printable.index(x) + 1 for x in code_snippet if x in printable]
            for code_snippet in df.code
        ]
        # X = sequence.pad_sequences(code_snippet_int_tokens, maxlen=max_len) # original
        X = pad_sequences(code_snippet_int_tokens, maxlen=max_len)
        target = np.array(df.isMalicious)
        print(
            f"Matrix dimensions of X: {X.shape},\nVector dimension of y:{target.shape}")
        return X, target

    def split_data(self, config, df):
        """Split data into train and test sets"""
        X, y = classfr.tokenize_data(df, config["preprocess"]["max_len"])
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y,
            test_size=config["model"]["split_ratio"],
            random_state=config["model"]["seed"],
        )
        print(
            f"Train data: {X_train.shape}, {y_train.shape}\n \
                Test data: {X_test.shape}, {y_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    def save_model_idetect(self, model_JSON, file_weights):
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

    def save_model(self, config, model, model_file):
        """save the trained model as a pickle file"""
        if config["model"]["save"]:
            if config["model"]["name"] != "RF":
                model.save(model_file)
            else:
                pickle.dump(model, open(model_file, "wb"))

        print("The final trained model is saved at: ", model_file)
        print("\n" + "-" * 35 + "Training Completed" + "-" * 35 + "\n")

    def print_layers_dims(self, model):
        """ Layer dimensions """
        l_layers = model.layers
        # Note None is ALWAYS batch_size
        for layer in l_layers:
            print(layer)
            print(
                "Input Shape: ",
                layer.input_shape,
                "Output Shape: ",
                layer.output_shape)

    def load_model(self, model_JSON, file_weights):
        """Load model from disk"""
        with open(model_JSON, "r") as f:
            model_json = json.load(f)
            model = model_from_json(model_json)

        model.load_weights(file_weights)
        return model

    def init_neptune(self, model_name, epochs, data_file):
        """return neptune init object if neptune is enabled"""
        import neptune

        # from neptune.integrations.tensorflow_keras import NeptuneCallback

        print("\n" + "-" * 30 + "Neptune" + "-" * 30 + "\n")
        exp_tags = [model_name, f"epochs:{epochs}", data_file]

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
        # save the configuration and module file to Neptune.
        nt_run["configurations"].upload("config.yaml")
        nt_run["model_archs"].upload("src/models.py")
        nt_run["code"].upload("src/classifier.py")
        return nt_run

    def apply_checkpoints(self, model, cp_path, patience):
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

    def select_model_arch(self, config):
        """Choose ML model"""
        # Display the settings first
        model_name = config["model"]["name"]

        print("\n\n" + "=" * 25 + " " + model_name +
              " Model Training " + "=" * 25)
        print(f"Configurations: \n {config}")
        print("-" * 50)

        arch = ModelArchs(config)

        if model_name == "RNN":
            model = arch.apply_RNN()
        elif model_name == "CNN":
            model = arch.apply_CNN()
        elif model_name == "LSTM":
            model = arch.apply_LSTM()
        elif model_name == "RF":
            model = arch.apply_RF(df)
        elif model_name == "multiDNN":
            model = arch.apply_multiDNN()
        else:
            print("Invalid model! Please select any valid model!")
            exit(1)
        return model

    def train_model(self, config, model_file, X_train, y_train, X_test, y_test):
        """train the selected model"""
        model_name = config["model"]["name"]
        epochs = config["dnn"]["epochs"]

        # Select the model architecture
        model = self.select_model_arch(config)

        # store metadata to neptune.ai
        if config["model"]["use_neptune"]:
            from neptune.integrations.tensorflow_keras import NeptuneCallback
            nt_run = self.init_neptune(model_name, epochs, config["data_file"])

        # Apply callbacks for training to store the best model checkpoint
        # and apply early stopping.
        if model_name != "RF":
            tf_callbacks = self.apply_checkpoints(
                model=model,
                cp_path=config["model"]["path"],
                patience=config["dnn"]["patience"],
            )
            if config["model"]["use_neptune"]:
                neptune_cbk = NeptuneCallback(
                    run=nt_run, base_namespace="metrics")
                tf_callbacks.append(neptune_cbk)

            # Fitting model and Cross-Validation
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=config["dnn"]["batch"],
                validation_data=(X_test, y_test),
                # callbacks=[tf_callbacks],
                callbacks=[DVCLiveCallback(save_dvc_exp=True)],
            )
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            # plot_metrics(history)
            fig_name = config["model"]["path"] + model_name

            plot = Plotter(config)
            plot.plot_history(history, fig_name)
            print(f"\nAccuracy of the model: {accuracy}\n")

            # save tracked files
            if config["model"]["use_neptune"]:
                nt_run["learning_curves"].track_files(fig_name + ".pdf")
                nt_run["loss_curve"].track_files(fig_name + "_loss.pdf")

            classfr.save_model(config, model, model_file)

        else:
            # TODO: log non-DNN models output to Neptune
            # nt_run["acc"] = ?? or params=dict
            print(f"Trained with non-DNN model: {model_name}")
        return model

    def evaluate_model(self, config, model_file, X_eval, y_eval):
        """evaluate the trained model
        """
        if config["model"]["name"] != "RF":
            if Path(model_file).is_file():
                print("Loading the trained model from: ", model_file)
                model = self.load_model(model_file)
                print(f"Model loaded successfully! \nEvaluating the model...")
                print("Model Summary: \n", model.summary())

                y_pred = model.predict(X_eval)
                y_pred = np.argmax(y_pred, axis=1)
                y_eval = np.argmax(y_eval, axis=1)
                print("Classification Report: \n",
                      classification_report(y_test, y_pred))
            else:
                print(f"Model file: {model_file} not found!")
                print("Please train the model first!")
        else:
            train_model = pickle.load(open(model_file, "RF"))
            result = self.train_model.score(X_eval, y_eval)
            print("Result: ", result)

        print("\n" + "-" * 35 + "Testing Completed" + "-" * 35 + "\n")

    def parse_args(self):
        """Parse command line arguments."""
        parser = ArgumentParser(
            description="AI-enabled IoT Cybersecurity Approach for Vulnerability Detection..."
        )
        parser.add_argument("--model", type=str,
                            help="Name of the ML model to train/test.")
        parser.add_argument("--data", type=str,
                            help="Data file for train/test.")
        return parser.parse_args()


if __name__ == "__main__":
    classfr = Classifier()
    paras = classfr.parse_args()
    config = classfr.load_config("config.yaml")

    # create a specific dir to save the trained model
    config = classfr.update_config_args(
        config=config,
        paras=paras)

    model_file = config["model"]["path"] + "model-final.h5"

    # Load input data
    df = classfr.load_data(data_csv=config["data_file"])

    # Split the dataset
    X_train, X_test, y_train, y_test = classfr.split_data(config, df)

    # Train the model
    if config['train']:
        model = classfr.train_model(
            config, model_file, X_train, y_train, X_test, y_test)

    # TODO: Evaluation of the trained model
    if config["test"]:
        output_size = len(set(list(y_train)))
        classfr.evaluate_model(config, model_file, X_test, y_test)
