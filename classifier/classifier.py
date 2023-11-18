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

import warnings
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import numpy as np

# from string import printable
import pandas as pd
import tensorflow as tf
import os
# from dvclive.keras import DVCLiveCallback
from sklearn.metrics import classification_report


# custom modules
from classifier.models import ModelArchs
from classifier.plot import Plotter
from classifier.preprocess import Preprocessor
from classifier.utility import Utility
from classifier.embeddings import MyEmbeddings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

        self.prepro = Preprocessor(self.config)

    def update_config_args(self, paras):
        """create model dir to store it
         set the config args from the command line if provided """
        self.config["model"]["name"] = paras.model if paras.model else self.config["model"]["name"]
        self.config['model']['type'] = paras.type if paras.type else self.config["model"]["type"]
        self.config["data_file"] = paras.data if paras.data else self.config["data_file"]
        self.config['granular'] = paras.granular if paras.granular else self.config['granular']

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
            # TODO: enable this line.
            # self.config["model"]["use_neptune"] = True

        if self.config["train"]:
            Path(self.config["model"]["path"]).mkdir(
                parents=True, exist_ok=True)
        print(f"\n\nModel path: {self.config['model']['path']}")
        return self.config

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

    def select_model_arch(self, params=None):
        """Choose ML model"""
        model_name = self.config["model"]["name"]
        print("\n\n" + "=" * 25 + " " + model_name +
              " Model Training " + "=" * 25)
        print("-" * 50)

        self.arch = ModelArchs(self.config)

        # ## statement-level granularity training
        if self.config['granular'] == 'statement':
            if model_name == "RNN":
                model = self.arch.apply_RNN()
            elif model_name == "CNN":
                model = self.arch.apply_CNN()
            elif model_name == "LSTM":
                model = self.arch.apply_LSTM()
            # elif model_name == "RF":
            #     model = self.arch.apply_RF(input_data)
            elif model_name == "multiDNN":
                model = self.arch.apply_multiDNN()
            elif model_name == "DNN":
                model = self.arch.apply_DNN()
            else:
                print("Invalid model! Please select a valid model!")
                exit(1)

        elif self.config['granular'] == 'function':
            # ## function-level granularity training
            vocab_size = params['vocab_size']
            embedding_matrix = params['embedding_matrix']
            max_len = params['max_len']

            if model_name == 'CNN':
                model = self.arch.apply_funCNN(
                    vocab_size=vocab_size,
                    embedding_matrix=embedding_matrix,
                    MAX_LEN=max_len,
                )
            elif model_name == 'RNN':
                model = self.arch.apply_funRNN(
                    vocab_size=vocab_size,
                    embedding_matrix=embedding_matrix,
                    MAX_LEN=max_len,
                )
            elif model_name == 'LSTM':
                model = self.arch.apply_funLSTM(
                    vocab_size=vocab_size,
                    embedding_matrix=embedding_matrix,
                    MAX_LEN=max_len,
                )
            else:
                print("Invalid model! Please select a valid model!\n" +
                      "Select one of CNN, RNN, or LSTM model")
                exit(1)
        return model

    def train_model(self, model_file, X_train, y_train, X_test, y_test, params=None):
        """train the selected model"""
        model_name = self.config["model"]["name"]
        epochs = self.config["dnn"]["epochs"]

        if model_name != "RF":
            # Select the model architecture
            model = self.select_model_arch(params)
            # store metadata to neptune.ai
            if self.config["model"]["use_neptune"] is True:
                from neptune.integrations.tensorflow_keras import NeptuneCallback
                nt_run = self.util.init_neptune(
                    model_name=model_name,
                    data_file=self.config["data_file"],
                    epochs=str(self.config['dnn']['epochs'])
                )
            tf_callbacks = self.apply_checkpoints(
                model=model,
                cp_path=self.config["model"]["path"],
                patience=self.config["dnn"]["patience"],
            )
            if self.config["model"]["use_neptune"]:
                neptune_cbk = NeptuneCallback(
                    run=nt_run,
                    base_namespace="metrics",
                    log_model_diagram=True,
                    log_on_batch=True,
                )
                tf_callbacks.append(neptune_cbk)

            # class_weights = class_weight.compute_class_weight(
            #     class_weight='balanced',
            #     classes=np.unique(y_train),
            #     y=y_train)

            # class_weights = dict(enumerate(class_weights))
            # # try: hardcoded blancer
            # # class_weights = {0: 0.5, 1: 0.5}
            # print(f'Class_weights: {class_weights}')

            # Fitting model and cross-validation
            # Apply callbacks for training to store the best model checkpoint
            # and apply early stopping.
            history = model.fit(
                x=X_train.tolist(),
                y=y_train.tolist(),
                epochs=epochs,
                batch_size=self.config["dnn"]["batch"],
                validation_data=(X_test.tolist(), y_test.tolist()),
                verbose=1,
                callbacks=[tf_callbacks],
                # use_multiprocessing=True,
                # workers=8,
                # callbacks=[DVCLiveCallback(save_dvc_exp=True)],
                # class_weight=class_weights,
            )

            fig_name = self.config["model"]["path"] + model_name
            plot = Plotter(self.config)
            plot.plot_history(history, fig_name)

            # save the tracked files
            if self.config["model"]["use_neptune"]:
                nt_run["learning_curves"].track_files(fig_name + ".pdf")
                nt_run["loss_curve"].track_files(fig_name + "_loss.pdf")
        else:
            # TODO: log non-DNN models output to Neptune
            # Fitting
            self.arch = ModelArchs(self.config)
            model = self.arch.apply_RF(input_data=X_train)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            print(f"Accuracy: {acc}")
            print(f"Trained with non-DNN model: {model_name}")
        return model

    # reconstruct labels from y_pred
    def reconstruct_labels(self, y_eval, y_pred):
        """Reconstruct labels from y_pred"""

        if self.config['model']['type'] == 'binary':
            y_pred = (y_pred > 0.5).astype('int32')

        y_eval = self.prepro.decode_multiclass(y_eval)
        y_pred = self.prepro.decode_multiclass(y_pred)

        print('+'*40)
        print(f'y_eval targets: \n{pd.value_counts(list(y_eval))}')
        print(f'\ny_pred targets: \n{pd.value_counts(list(y_pred))}')
        print("Evaluation labels shape", np.shape(y_eval))
        print('+'*40)
        print(classification_report(y_eval, y_pred))
        return y_pred

    def evaluate_model(self, model, X_eval, y_eval):
        """Evaluate the trained model
        """
        print(f'Model Type: {type(model)}')
        if self.config["model"]["name"] != "RF":
            # if type(model) == 'keras.src.engine.functional.Functional':
            #     print("Model is a functional model")
            #     model.summary()
            #     model = self.util.load_tf_model(model)
            # elif Path(model).is_file():

            print("\nEvaluating the model...\n")
            # evaluate the model
            eval_result = model.evaluate(X_eval, y_eval, verbose=1)
            print(f'Evaluation Result: {eval_result}')
            print('\nDone evaluation!\n')
            print('='*40)

            # # predict the model
            # y_pred = model.predict(X_eval)
            # self.reconstruct_labels(y_eval, y_pred)
        else:
            result = model.score(X_eval, y_eval)
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
        parser.add_argument("--type", type=str,
                            help="Classification type for train/test.")
        # get granular param
        parser.add_argument("--granular", type=str,
                            help="Granularity level for training.")
        return parser.parse_args()

    def run(self):
        """Run the training and testing process"""
        import json
        paras = self.parse_args()
        print(f'\n\nCommand arguments: {paras}')

        if self.config["model"]["name"] == "RF":
            assert self.config['train'] is True, \
                'The model should be trained first for non-DNN models!'

        # update config args
        self.config = self.update_config_args(paras=paras)
        print(f"\nUpdated Config:\n{json.dumps(self.config, indent=2)}\n")
        fun_params = {}  # for fun-level training

        print('-'*40)
        # load the preprocessor
        model_file = self.config["model"]["path"] + "model-final.h5"

        df = self.prepro.load_data(data_file=self.config["data_file"])

        if self.config['granular'] == 'function':
            # apply embeddings if granular level is function
            emb = MyEmbeddings(self.config)

            X, y, vocab_size, embedding_matrix = emb.vectorize_and_load_fun_data(
                df)
            # create dataframe from X, y
            df = pd.DataFrame({'code': X.tolist(), 'label': y.tolist()})

            # fill other_params
            fun_params['vocab_size'] = vocab_size
            fun_params['embedding_matrix'] = embedding_matrix
            fun_params['max_len'] = self.config['embedding']['max_len']

        # split data
        X_train, X_test, y_train, y_test = self.prepro.process_data(df)

        # Train the model
        if self.config['train']:
            model = self.train_model(
                model_file, X_train, y_train, X_test, y_test,
                params=fun_params)

            # save the trained model
            self.prepro.save_model(model, model_file)

        # Evaluation of the trained model
        if self.config["eval"]:
            # load the trained model for evaluation
            df_eval = self.prepro.load_data(
                data_file=self.config["eval_data"])

            if self.config["model"]["name"] != "RF":
                if self.config['granular'] == 'function':
                    # apply embeddings if granular level is function
                    emb = MyEmbeddings(self.config)

                    X_eval, y_eval, _, _ = emb.vectorize_and_load_fun_data(df)
                if self.config['granular'] == 'statement':
                    X_eval, y_eval = self.prepro.tokenize_data(
                        df=df_eval, max_len=self.config["preprocess"]["max_len"])
                else:
                    print('Select a valid granularity level: statement or function!')

                # self.evaluate_model(model_file, X_eval, y_eval)
                self.evaluate_model(model, X_eval.tolist(), y_eval.tolist())


if __name__ == "__main__":
    classfr = Classifier()
    classfr.run()
