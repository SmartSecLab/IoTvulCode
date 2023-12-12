"""
Copyright (C) 2023 SmartSecLab, Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT
@Programmer: Guru Bhandari
"""

import random
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
import yaml
from humanfriendly import format_timespan
from zipfile import ZipFile

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

    def check_internet(self, url: str) -> bool:
        """check if the internet is working or not."""
        try:
            response = requests.get(url)
            return True if response.status_code < 400 else False
        except Exception as exc:
            print(f"Invalid URL! {exc}")
            return False

    def retrieve_zip(self, url: str):
        """Fetching list of C/C++ files from zip file of the project url."""
        if self.check_internet(url):
            r = requests.get(url)
            # BytesIO keeps the file in memory
            return ZipFile(BytesIO(r.content))
        else:
            print("Internet is not working or the URL is Invalid!")
            return None

    def show_time_elapsed(self, start_time):
        """verbose time elapsed so far"""
        time_elapsed = time.time() - start_time
        print(f"Time elapsed: {format_timespan(time_elapsed)}")
        print("Continue gathering....\n")

    def get_benign_context(self, config, row) -> pd.DataFrame():
        """
        filter all lines if it is less than min threshold
        randomly shuffled lines
        """
        threshold = int(config["save"]["threshold_lines"])
        benign_ratio = config["save"]["benign_ratio"]
        seed = config["save"]["seed"]
        df = pd.DataFrame()

        # threshold [7] to filter short functions
        lines = [x for x in enumerate(
            row["code"].splitlines()) if len(x[1]) > threshold]

        # randomly shuffle lines and takes 1/4 of the total number of lines.
        random.seed(seed)
        lines = random.sample(
            population=lines,
            k=int(len(lines) * benign_ratio))

        # convert it to dataframe and add additional columns
        df = pd.DataFrame(data=lines, columns=["line", "context"])

        # remove leading and trailing whitespace
        df["context"] = df["context"].apply(
            lambda x: re.sub(r"\s+", " ", x).strip())
        df["cwe"] = "Benign"
        df["tool"] = "sampling"
        df["file"] = row["file"]
        df['line'] = 'unknown'
        return df

    def gen_benign(self, config, dfm) -> pd.DataFrame():
        """create benign samples to the dataframe"""
        df_fun = pd.DataFrame()

        for i in range(len(dfm)):
            df_get = self.get_benign_context(config, dict(dfm.iloc[i]))
            df_fun = pd.concat(
                [df_fun, df_get],
                ignore_index=True).reset_index(drop=True)
        return df_fun

    def filter_benign(self, code_dup, cwe):
        """ masking to filter duplicates and benign 
        keeping vulnerable samples as it is.
        """
        if code_dup is True and cwe == 'Benign':
            return True
        else:
            return False

    def filter_results(self, df: pd.DataFrame()) -> pd.DataFrame():
        """ apply filtering to remove duplicates and ambiquious samples
        """
        print('='*40)
        print('Removing duplicates and ambiquious samples...')
        print('-'*40)

        # table-wise columns to check duplicates
        code_col = 'code' if 'code' in df.columns else 'context'

        df = df.sort_values(by=[code_col, 'cwe'], ascending=False)
        original_size = len(df)
        print(f'Original data size before filtering: {len(df)}')

        # step1: remove duplicates checking [code_col, 'cwe']
        df = df.drop_duplicates(
            subset=[code_col, 'cwe']).reset_index(drop=True)
        step1_size = len(df)
        print(f"Step1-filtering on [code and cwe]: {len(df)}" +
              f" [filtered {original_size-step1_size}]")

        # step2: remove duplicates checking [code_col]
        df['code_dup'] = df[code_col].duplicated()
        # df = df[~df['code_dup']] # this removes vul samples as well
        df['filter_mask'] = df.apply(
            lambda row: self.filter_benign(row.code_dup, row.cwe), axis=1)
        df = df[~df['filter_mask']].reset_index(drop=True)
        df = df.drop(labels=['code_dup', 'filter_mask'], axis=1)
        step2_size = len(df)
        print(f"Step2-filtering on [code] with vuls: {len(df)}" +
              f" [filtered {step1_size-step2_size}]")
        print('='*40)
        return df

    def show_info_pd(self, df: pd.DataFrame, name: str):
        print(f"\nShape of [{name}] data of all the projects: {df.shape}")
        print(f" #vulnerable: {len(df[df.cwe!='Benign'])}")
        print(f" #benign: {len(df[df.cwe=='Benign'])}\n")

    def save_binary(self, filename: str, df: pd.DataFrame()) -> pd.DataFrame():
        """save a dataframe to a binary file"""
        df["isVul"] = df["cwe"].apply(
            lambda x: 1 if x != "Benign" else 0)
        if 'context' in df.columns:
            df = df.rename(columns={"context": "code"})
        df = df[["code", "isVul"]]
        df.to_csv(filename, index=False)
        return df
