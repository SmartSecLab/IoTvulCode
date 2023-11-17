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
import tensorflow as tf
import os
import errno

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
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
        # np.savetxt(r'../VulBERTa/data/dummy/tinyvul.txt', df.code.values, fmt='%s', delimiter='\t')
        print("-" * 50)
        return df

    def save_vectorized_data(self, X, y, file):
        """Save the vectorized data to a file."""
        print('Saving the vectorized data...')
        df = pd.DataFrame()
        df['X'] = X.tolist()
        df['y'] = y.tolist()
        df.to_pickle(file)
        print(f'Dataframe shape: {df.shape}')
        print(f'Saved the vectorized data to {file}')

    def load_vectorized_data(self, file):
        """Load the vectorized data from a file."""
        print(f'Loading the vectorized data from {file}...')
        df = pd.read_pickle(file)
        X = np.array(df['X'].tolist())
        y = np.array(df['y'].tolist())
        print(f'X shape: {X.shape} and y shape: {y.shape}')
        print('Vectorized data loaded!')
        return X, y

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
        except OSError as e:
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or dir
                raise  # re-raise exception if a different error occurred

    def encode_multiclass(self, y):
        """encode multiclass target """
        encoder = LabelEncoder()
        encoder.fit(y)
        output_dim = len(encoder.classes_)
        print(f'\nNumber of target classes: {output_dim}')
        onehot_y = encoder.transform(y)

        if self.config['model']['type'] == 'multiclass':
            onehot_y = tf.keras.utils.to_categorical(onehot_y)
            self.config['dnn']['output_dim'] = output_dim

        elif self.config['model']['type'] == 'binary':
            onehot_y = np.reshape(onehot_y, (-1, 1))
            self.config['dnn']['output_dim'] = 1
        else:
            raise ValueError('Invalid model type!')

        classes_pkl = 'data/classes.pkl'
        self.silent_remove(filename=classes_pkl)

        with open(classes_pkl, 'wb') as f:
            pickle.dump(encoder, f)
        return onehot_y

    def decode_multiclass(self, onehot_y):
        """decode multiclass target """
        with open('data/classes.pkl', 'rb') as f:
            encoder = pickle.load(f)

        if self.config['model']['type'] == 'multiclass':
            decoded_y = [encoder.inverse_transform(
                [np.argmax(v)])[0] for v in onehot_y]

        elif self.config['model']['type'] == 'binary':
            decoded_y = encoder.inverse_transform(onehot_y)
        else:
            raise ValueError("Please choose either binary or multiclass.")
        # decoded_y = [x[0] for x in decoded_y]
        return decoded_y

    def show_y(self, y):
        """Count the number of labels in the dataset"""
        # weights = list(np.bincount(y))
        # weights = {i: weights[i] for i in range(len(weights))}
        print('\n\n' + '+'*30)
        # print(f'Weight of labels:\n{weights}')
        labels_dist = pd.Series(y).value_counts()
        print(f'\nDistribution of targets: \n{labels_dist}')
        print(f'\n#classes: {len(labels_dist)}')

        # if len(labels_dist) < 2:
        #     raise ValueError('Only one class found in the dataset.')
        # exit(0)
        print('+'*30)

    # define function to apply RandomOverSampler
    def apply_over_sampling(self, X, y):
        """define oversampling strategy"""
        print('Applying data over sampling...')
        oversample = RandomOverSampler(sampling_strategy='minority')
        # fit and apply the transform
        X_over, y_over = oversample.fit_resample(X, y)
        return X_over, y_over

    # define function to apply under_sampling

    def apply_under_sampling(self, X, y):
        """define undersampling strategy"""
        print('Applying data under sampling (NearMiss)...')
        undersample = NearMiss(version=1, n_neighbors=3)
        # fit and apply the transform
        X_under, y_under = undersample.fit_resample(X, y)
        return X_under, y_under

    def process_data(self, df):
        """Split data into train and test sets"""
        if self.config['model']['type'].lower() == 'multiclass':
            # TODO: do we need this filterization?
            # filter out labels with less than 200 samples
            # df = df.groupby('label').filter(
            #     lambda v: len(v) > 100).reset_index(drop=True)
            pass

        elif self.config['model']['type'].lower() == 'binary':
            # target representation for binary classification
            df['label'] = df['label'].apply(
                lambda x: x if x == 'Benign' else 'Vulnerable')
            # y = [v if v == 'Benign' else 'Vulnerable' for v in y]
        else:
            raise ValueError(
                f"Invalid model type: {self.config['model']['type']}."
                f"Please choose either binary or multiclass.")

        if self.config['model']['name'] == 'RF':
            X, y = df.code, df.label
        else:
            if self.config['granular'] == 'statement':
                X, y = self.tokenize_data(
                    df, self.config["preprocess"]["max_len"])
            else:
                X, y = df.code, df.label

        self.show_y(y)

        # convert list to numpy array for training
        y = self.encode_multiclass(y)

        if self.config['apply_balancer'] == True:
            # X, y = self.apply_over_sampling(X, y)
            X, y = self.apply_under_sampling(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["model"]["split_ratio"],
            random_state=self.config["model"]["seed"],
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
        print('='*40)
        print(f'Model saved to {model_file}')
        print('='*40)

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
