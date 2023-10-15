# Analysis of IoTvulCode tool generated output for feeding non-vul statements:

import os
import random
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
import yaml
from humanfriendly import format_timespan

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
