"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, config):
        self.config = config

    def plot_metrics(self, history):
        print(history.history.keys())
        plt.plot(history.history["acc"])
        # plt.plot(history.history['loss'])
        plt.title("The CNN model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy"], loc="lower right")
        # plt.show()

    def plot_history(self, history, fig_name=None):
        """
        plot ML performance metrics obtained from the training/testing history
        reference: https://github.com/SmartSecLab/ENViSEC/blob/main/src/utility.py
        """
        plt.figure(figsize=(12, 10))
        plt.rcParams["font.size"] = "16"
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.plot(history.epoch, np.array(
            history.history["acc"]), label="train_acc")
        plt.plot(history.epoch, np.array(
            history.history["val_acc"]), label="val_acc")

        if "precision" in history.history:
            plt.plot(
                history.epoch, np.array(history.history["precision"]), label="precision"
            )
            plt.plot(
                history.epoch,
                np.array(history.history["val_precision"]),
                label="val_precision",
            )

        if "recall" in history.history:
            plt.plot(history.epoch, np.array(
                history.history["recall"]), label="recall")
            plt.plot(
                history.epoch, np.array(history.history["val_recall"]), label="val_recall"
            )
        plt.legend()
        # plt.ylim([0, 1])
        if fig_name:
            plt.savefig(fig_name + ".pdf")

        # plotting loss curve separately
        plt.figure(figsize=(12, 10))
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.plot(history.epoch, np.array(
            history.history["loss"]), label="train_loss")
        plt.plot(history.epoch, np.array(
            history.history["val_loss"]), label="val_loss")
        plt.legend()
        # plt.ylim([0, 1])
        if fig_name:
            plt.savefig(fig_name + "_loss.pdf")

    def plot_curves(self, fig_name, kfold, trained_model, X, y, cv=3, return_times=True):
        """
        plot different curves of ML measures.
        reference: https://github.com/SmartSecLab/ENViSEC/blob/main/src/utility.py
        """
        # fig, ax1 = plt.subplots(1, 2, figsize=(10, 15))
        title = "Learning Curves"
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator=trained_model,
            X=X,
            y=y,
            cv=kfold,  # cross validation default folds = 5 from sklearn 0.22
            return_times=return_times,
        )
        plt.rcParams["font.size"] = "16"
        plt.plot(train_sizes, np.mean(train_scores, axis=1))
        plt.plot(train_sizes, np.mean(test_scores, axis=1))
        plt.legend(["Training", "Testing"], loc="lower right")
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.savefig(fig_name)


# print(model.summary())
# model_fig = 'model_plot.png'
# plot_model(model, to_file=model_fig, show_shapes=True, show_layer_names=True)
# print(f'Model architecture is saved at {model_fig}')

# token_sizes = [len(x) for x in X_train]
# token_sizes_pd = pd.Series(token_sizes)
# print("Token distribution:")
# print(token_sizes_pd.describe())

# max_length = MAX_LEN
# token_sizes_pd = token_sizes_pd[token_sizes_pd <= MAX_LEN]
# # token_sizes_pd.plot.box()
