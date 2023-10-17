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

import pickle
import warnings
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
from datetime import datetime
from sklearn import model_selection

# from string import printable
import pandas as pd
import tensorflow as tf
import yaml
from dvclive.keras import DVCLiveCallback
from sklearn.metrics import classification_report

# custom modules
from classifier.models import ModelArchs
from classifier.plot import Plotter
from classifier.preprocess import Preprocessor
from classifier.utility import Utility


class Classifier:
    """ This class is responsible for the following:
        - loading the data
        - tokenizing the data
        - training the model
        - evaluating the model
    """

    def __init__(self):
        self.util = Utility()
        self.config = self.util.load_config("config/classifier.yaml")
        self.model = None
        self.history = None
        self.arch = ModelArchs(self.config)

    def update_config_args(self, paras):
        """create model dir to store it
         set the config args from the command line if provided """
        self.config["model"]["name"] = paras.model if paras.model else self.config["model"]["name"]
        self.config["data_file"] = paras.data if paras.data else self.config["data_file"]

        if self.config['debug']:
            self.config['dnn']['epochs'] = self.config['dnn']['debug_epochs']
            time_now = ''
        else:
            time_now = datetime.now().strftime('%Y-%m-%d_%H.%Mm')

        mdir = self.config["model"]["name"] + "-" + str(self.config["dnn"]["epochs"]) + \
            "-" + Path(self.config["data_file"]).stem + "-" + time_now + "/"

        self.config["model"]["path"] = self.config["model"]["path"] + mdir

        if self.config["debug"] == True:
            self.config["model"]["path"] = self.config["model"]["path"].rsplit(
                "/", 1)[0] + "-debug/"
            self.config["model"]["use_neptune"] = False

        if self.config["train"]:
            Path(self.config["model"]["path"]).mkdir(
                parents=True, exist_ok=True)
        print(f"\n\nModel path: {self.config['model']['path']}")
        return self.config

    def init_neptune(self, model_name, data_file):
        """Return neptune init object if it is enabled"""
        import neptune

        nt_config = configParser()
        neptune_file = ".neptune.ini"
        nt_config.read(neptune_file)
        project = nt_config["neptune_access"]["project"]
        api_token = nt_config["neptune_access"]["api_token"]
        epochs = str(self.config['dnn']['epochs'])
        exprem_tags = [model_name, f"epochs:{epochs}", data_file]

        print("\n" + "-" * 30 + "Neptune" + "-" * 30 + "\n")
        print("Reading neptune config file: ", neptune_file)

        # put your neptune credentials here
        nt_run = neptune.init_run(
            project=project,
            api_token=api_token,
            name="IoTvulCode",
            tags=exprem_tags
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

    def select_model_arch(self):
        """Choose ML model"""

        model_name = self.config["model"]["name"]

        print("\n\n" + "=" * 25 + " " + model_name +
              " Model Training " + "=" * 25)
        print(f"Configurations: \n {config}")
        print("-" * 50)

        if model_name == "RNN":
            model = self.arch.apply_RNN()
        elif model_name == "CNN":
            model = self.arch.apply_CNN()
        elif model_name == "LSTM":
            model = self.arch.apply_LSTM()
        elif model_name == "RF":
            model = self.arch.apply_RF(df)
        elif model_name == "multiDNN":
            model = self.arch.apply_multiDNN()
        else:
            print("Invalid model! Please select a valid model!")
            exit(1)
        return model

    def train_model(self, model_file, X_train, y_train, X_test, y_test):
        """train the selected model"""
        model_name = self.config["model"]["name"]
        epochs = self.config["dnn"]["epochs"]

        # Apply callbacks for training to store the best model checkpoint
        # and apply early stopping.
        if model_name != "RF":
            # Select the model architecture
            model = self.select_model_arch()

            # store metadata to neptune.ai
            if self.config["model"]["use_neptune"]:
                from neptune.integrations.tensorflow_keras import NeptuneCallback
                nt_run = self.init_neptune(
                    model_name, epochs, self.config["data_file"])

            tf_callbacks = self.apply_checkpoints(
                model=model,
                cp_path=self.config["model"]["path"],
                patience=self.config["dnn"]["patience"],
            )
            if self.config["model"]["use_neptune"]:
                neptune_cbk = NeptuneCallback(
                    run=nt_run,
                    base_namespace="metrics"
                )
                tf_callbacks.append(neptune_cbk)

            # Fitting model and cross-validation
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=self.config["dnn"]["batch"],
                validation_data=(X_test, y_test),
                # callbacks=[tf_callbacks],
                callbacks=[DVCLiveCallback(save_dvc_exp=True)],
            )
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            fig_name = self.config["model"]["path"] + model_name

            plot = Plotter(self.config)
            plot.plot_history(history, fig_name)
            print(f"\nAccuracy of the model: {accuracy}\n")

            # save the tracked files
            if self.config["model"]["use_neptune"]:
                nt_run["learning_curves"].track_files(fig_name + ".pdf")
                nt_run["loss_curve"].track_files(fig_name + "_loss.pdf")
        else:
            # TODO: log non-DNN models output to Neptune
            # nt_run["acc"] = ?? or params=dict
            # Fitting
            model = self.arch.apply_RF(df.code)
            # model = self.select_model_arch()
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f"Accuracy: {acc}")
            print(f"Trained with non-DNN model: {model_name}")
        return model

    def load_tf_model(self, model_file):
        """ 
        Load model from disk
        Args:
            model_file (_type_): trained tensorflow model (h5)
            file_weights (_type_): file weights of the trained model
        """
        print(f"\nLoading the trained model from: \n{model_file}")
        model = tf.keras.models.load_model(model_file)
        # model.load_weights(file_weights)
        print('-'*20)
        print(model.summary())
        print("\nModel loaded successfully!\n")
        print('-'*20)
        return model

    def evaluate_model(self, model_file, X_eval, y_eval):
        """Evaluate the trained model
        """
        if self.config["model"]["name"] != "RF":
            if Path(model_file).is_file():
                model = self.load_tf_model(model_file)
                print("\nEvaluating the model...\n")
                # evaluate the model
                loss, acc = model.evaluate(X_eval, y_eval, verbose=1)

                # y_pred = model.predict(X_eval)
                # print(f'y_pred: {y_pred}')
                # print(f'y_eval: {y_eval}')
                # print(f'\ny_pred.shape: {y_pred.shape}')
                # print(f'y_eval.shape: {y_eval.shape}')

                # # y_pred = np.argmax(y_pred, axis=1)
                # # y_eval = np.argmax(y_eval, axis=1)

                # print(f'\ny_pred: {y_pred}')
                # print(f'y_eval: {y_eval}')
                # print(f'\ny_pred.shape: {y_pred.shape}')
                # print(f'y_eval.shape: {y_eval.shape}')

                # cls_report = classification_report(y_eval, y_pred)
                # print(f"Classification Report: \n{cls_report}")
                print('loss: ', loss)
                print('acc: ', acc)
            else:
                print(f"\n\nModel file: {model_file} not found!")
                print("Please train the model first!")
        else:
            result = model_file.score(X_eval, y_eval)
            print("Result: ", result)
        print("\n" + "-" * 35 + "Testing Completed" + "-" * 35 + "\n")

    def parse_args(self):
        """Parse command line arguments."""
        parser = ArgumentParser(description="AI-enabled IoT \
            Cybersecurity Approach for Vulnerability Detection...")
        parser.add_argument("--model", type=str,
                            help="Name of the ML model to train/test.")
        parser.add_argument("--data", type=str,
                            help="Data file for train/test.")
        return parser.parse_args()


if __name__ == "__main__":
    classfr = Classifier()
    paras = classfr.parse_args()
    config = classfr.config

    # update config args
    config = classfr.update_config_args(paras=paras)
    model_file = config["model"]["path"] + "model-final.h5"

    preprocess = Preprocessor(config)
    # Load input data
    df = preprocess.load_data(data_file=config["data_file"])

    # Split the dataset
    X_train, X_test, y_train, y_test = preprocess.split_data(df)

    # Train the model
    if config['train']:
        model = classfr.train_model(
            model_file, X_train, y_train, X_test, y_test)

        # save the trained model
        preprocess.save_model(model, model_file)

    # TODO: Evaluation of the trained model
    df_eval = preprocess.load_data(data_file=config["eval_data"])

    if config["test"]:
        if config["model"]["name"] != "RF":
            X_eval, y_eval = preprocess.tokenize_data(
                df=df_eval, max_len=config["preprocess"]["max_len"])
            # output_size = len(set(list(y_train)))
            classfr.evaluate_model(model_file, X_eval, y_eval)
        else:
            classfr.evaluate_model(model, X_eval, y_eval)
