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

import glob
import os
import pickle
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


class Plotter:
    def __init__(self, config):
        self.config = config

    def plot_metrics(history):
        print(history.history.keys())
        plt.plot(history.history["acc"])
        # plt.plot(history.history['loss'])
        plt.title("The CNN model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy"], loc="lower right")
        # plt.show()

    def plot_history(history, fig_name=None):
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

    def plot_curves(fig_name, kfold, trained_model, X, y, cv=3, return_times=True):
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
