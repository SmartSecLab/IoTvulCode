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
from pathlib import Path
from string import printable
import pickle
import _pickle as cPickle
import skops.io as sio
from tabulate import tabulate
import os
import errno

import numpy as np
import pandas as pd
import yaml
from sklearn import model_selection
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# import RandomOverSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss


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
        # og: encoding="unicode_escape"
        if self.config['debug'] is True:
            df = pd.read_csv(data_file,
                             encoding="utf-8",
                             nrows=int(self.config['debug_rows'])
                             )
        else:
            df = pd.read_csv(data_file, encoding='utf-8')

        # Checking for duplicate rows or null values
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        print(f"\nShape of the input data: {df.shape}")
        print("\nSamples:")
        print("-" * 50)
        print(tabulate(df.head(3), df.columns, tablefmt="simple_grid"))
        print("-" * 50)
        return df

    def tokenize_data(self, df, max_len):
        """Dataset tokenization"""
        code_snippet_int_tokens = [
            [printable.index(x) + 1 for x in code_snippet if x in printable]
            for code_snippet in df.code]

        # Pad the sequences (left padded with zeros)
        # to the max length of the code snippet
        X = pad_sequences(
            code_snippet_int_tokens,
            maxlen=max_len,
            padding='post'
        )
        target = np.array(df.label)
        print(f"Shape of X: {X.shape}, Shape of y:{target.shape}")
        return X, target

    def silent_remove(self, filename):
        """remove file if exists"""
        try:
            os.remove(filename)
        except OSError as e:  # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                raise  # re-raise exception if a different error occurred

    def encode_multiclass(self, y):
        """encode multiclass target """
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)

        classes_pkl = 'data/classes.pkl'
        self.silent_remove(filename=classes_pkl)

        with open(classes_pkl, 'wb') as f:
            pickle.dump(encoder, f)
        return encoded_y

    def decode_multiclass(self, encoded_y):
        """decode multiclass target """
        with open('data/classes.pkl', 'rb') as f:
            encoder = pickle.load(f)

        decoded_y = encoder.inverse_transform(encoded_y)
        # decoded_y = [x[0] for x in decoded_y]
        return decoded_y

    def show_y(self, y):
        """Count the number of labels in the dataset"""
        weights = list(np.bincount(y))
        weights = {i: weights[i] for i in range(len(weights))}
        print('+'*30)
        print(f'Weight of labels:\n{weights}')
        print('+'*30)

    # def show_x(self, X):
    #     """Show x values"""
    #     print('+'*40)
    #     print(f'x_type single: {type(X[0][0])}')
    #     print(f'X type matrix: {type(X)}')
    #     # np.save('X_train_part.txt', X_train)
    #     # with open('X_train_part.txt', 'w') as f:
    #     #     for item in X:
    #     #         f.write("%s\n" % item)
    #     # print(f'x_value sample: {X}')
    #     print('+'*40)

    # define function to apply RandomOverSampler
    def apply_over_sampling(self, X, y):
        # define oversampling strategy
        oversample = RandomOverSampler(sampling_strategy='minority')
        # fit and apply the transform
        X_over, y_over = oversample.fit_resample(X, y)
        return X_over, y_over

    # define function to apply under_sampling
    def apply_under_sampling(self, X, y):
        # define undersampling strategy
        undersample = NearMiss(version=1, n_neighbors=3)
        # fit and apply the transform
        X_under, y_under = undersample.fit_resample(X, y)
        return X_under, y_under

    def split_data(self, df):
        """Split data into train and test sets"""
        if self.config['model']['type'].lower() == 'multiclass':
            # TODO: do we need this filterization?
            # filter out labels with less than 200 samples
            df = df.groupby('label').filter(
                lambda v: len(v) > 200).reset_index(drop=True)

        elif self.config['model']['type'].lower() == 'binary':
            # target representation for binary classification
            df['label'] = df['label'].apply(
                lambda x: x if x == 'Benign' else 'Vulnerable')
            # y = [v if v == 'Benign' else 'Vulnerable' for v in y]
        else:
            raise ValueError(
                f"Invalid model type: {self.config['model']['type']}."
                f"Please choose either binary or multiclass.")

        print(f'Distribution of labels: \n{df.label.value_counts()}')

        if self.config['model']['name'] == 'RF':
            X, y = df.code, df.label
        else:
            X, y = self.tokenize_data(df, self.config["preprocess"]["max_len"])

        # convert list to numpy array for training
        y = self.encode_multiclass(y)

        # X, y = self.apply_over_sampling(X, y)
        X, y = self.apply_under_sampling(X, y)

        # X = X.astype('float32')
        # self.show_x(X)
        self.show_y(y)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y,
            test_size=self.config["model"]["split_ratio"],
            # random_state=self.config["model"]["seed"],
        )
        print(f"\nTrain data; X: {X_train.shape}, y{y_train.shape}")
        print(f"Test data; X: {X_test.shape}, y: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    def save_model(self, model, model_file):
        """Save the trained model as a pickle file"""
        if self.config["model"]["save"]:
            if self.config["model"]["name"] != "RF":
                model.save(model_file)
            else:
                sio.dumps(model_file)

    def save_model_idetect(self, model, model_json, file_weights):
        """Saving model to disk"""
        print(f"Saving model to disk:{model_json} and {file_weights}")
        # have h5py installed
        if Path(model_json).is_file():
            os.remove(model_json)
        json_string = model.to_json()

        with open(model_json, "w", encoding='utf-8') as f:
            json.dump(json_string, f)
        if Path(file_weights).is_file():
            os.remove(file_weights)
        model.save_weights(file_weights)

        print(f"The final trained model is saved at: {model_json}")
        print("\n" + "-" * 35 + "Training Completed" + "-" * 35 + "\n")
