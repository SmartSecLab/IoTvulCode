"""
Copyright (C) 2023 SmartSecLab, Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT
@Programmer: Guru Bhandari
"""
# Analysis of IoTvulCode tool generated output for feeding non-vul statements:

import warnings
from configparser import ConfigParser
from pathlib import Path

import yaml
import tensorflow as tf

warnings.filterwarnings("ignore")


class Utility():
    """
    This class does several supporting utility functions
    """

    def load_config(self, yaml_file):
        """Load the yaml file and return a dictionary
        """
        assert Path(yaml_file).is_file(), \
            f'The configuration file does not exist: {yaml_file}!'
        with open(yaml_file, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as err:
                return err

    # function to keep logging of the verbose and save log line to a file
    def log(self, log_file, log_line, verbose):
        """Log the verbose to a file
        """
        if verbose:
            print(log_line)
        with open(log_file, 'a') as f:
            f.write(log_line + '\n')

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

    def init_neptune(self, model_name, data_file, epochs):
        """Return neptune init object if it is enabled"""
        import neptune

        nt_config = ConfigParser()
        neptune_file = ".neptune.ini"
        nt_config.read(neptune_file)
        project = nt_config["neptune_access"]["project"]
        api_token = nt_config["neptune_access"]["api_token"]
        # epochs = str(self.config['dnn']['epochs'])
        exprem_tags = [model_name, f"epochs:{epochs}", data_file]

        print("\n" + "-" * 30 + "Neptune" + "-" * 30 + "\n")
        print("Reading neptune config file: ", neptune_file)

        # put your neptune credentials here
        nt_run = neptune.init_run(
            project=project,
            api_token=api_token,
            name="Tiny-Vul",
            tags=exprem_tags
        )
        # save the configuration and module file to Neptune.
        nt_run["configurations"].upload("config/classifier.yaml")
        nt_run["model_archs"].upload("classifier/models.py")
        nt_run["code"].upload("classifier/classifier.py")
        return nt_run
