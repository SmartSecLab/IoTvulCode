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
import warnings
from argparse import ArgumentParser
from configparser import ConfigParser

import pandas as pd
import yaml
from sklearn import model_selection
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Preprocessor():
    """This class does several preprocessing functions:
        - loading the data
        - tokenizing the data
        - saving the model
        - loading the model, etc.
    """

    def __init__(self, config):
        """Initialize the class with available settings
        Args:
            config (_dict_): configuration settings
        """
        self.config = config

    def load_data(self, data_file):
        """Load data code snippets"""
        df = pd.read_csv(
            data_file, encoding="utf-8")  # og: encoding="unicode_escape"

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
            for code_snippet in df.code]

        X = pad_sequences(code_snippet_int_tokens, maxlen=max_len)
        target = np.array(df.isMalicious)
        print(f"Matrix dimensions of X: {X.shape},\
                \nVector dimension of y:{target.shape}")
        return X, target

    def split_data(self, df):
        """Split data into train and test sets"""
        X, y = self.tokenize_data(df, self.config["preprocess"]["max_len"])

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y,
            test_size=self.config["model"]["split_ratio"],
            random_state=self.config["model"]["seed"],
        )
        print(f"Train data: {X_train.shape}, {y_train.shape}\n \
                Test data: {X_test.shape}, {y_test.shape}"
              )
        return X_train, X_test, y_train, y_test

    def save_model_idetect(self, model_JSON, file_weights):
        """Saving model to disk"""
        print(f"Saving model to disk:{model_JSON} and {file_weights}")
        # have h5py installed
        if Path(model_JSON).is_file():
            os.remove(model_JSON)
        json_string = model.to_json()
        with open(model_JSON, "w") as f:
            json.dump(json_string, f)
        if Path(file_weights).is_file():
            os.remove(file_weights)
        model.save_weights(file_weights)

    def save_model(self, model, model_file):
        """Save the trained model as a pickle file"""
        if self.config["model"]["save"]:
            if self.config["model"]["name"] != "RF":
                model.save(model_file)
            else:
                pickle.dump(model, open(model_file, "wb"))

        print(f"The final trained model is saved at: {model_file}")
        print("\n" + "-" * 35 + "Training Completed" + "-" * 35 + "\n")

    def load_model(self, model_JSON, file_weights):
        """Load model from disk"""
        with open(model_JSON, "r") as f:
            model_json = json.load(f)
            model = model_from_json(model_json)

        model.load_weights(file_weights)
        return model
