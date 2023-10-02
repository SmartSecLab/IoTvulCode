# Statement-level:
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

    # def __init__(self):
    #     # self.config = {}
    #     pass

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

        # TODO: remove the ambiguous vul line from the 'benign' lines if present
        # vul_line = df.line[i]
        # lines = [x for x in lines if x[0]!=vul_line]

        # convert it to dataframe and add additional columns
        df = pd.DataFrame(data=lines, columns=["line", "context"])

        # remove leading and trailing whitespace
        df["context"] = df["context"].apply(
            lambda x: re.sub(r"\s+", " ", x).strip())
        df["cwe"] = "Benign"
        df["tool"] = "sampling"
        df["file"] = row["file"]

        # # TODO: add line number to the dataframe
        # line_col = df["line"].astype(int) + int(row["start_line"])
        # max_line = max(list(line_col)) if list(line_col) else 0
        # end_line = int(row["end_line"])

        # # print(f"max of lines: {max_line} and end_line: {end_line}")
        # assert max_line <= end_line, "Line number shouldn't exceed function length!"
        # df["line"] = line_col
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

    # def filter_results(self, df):
    #     """apply several filters to the dataframe"""
    #     print("-" * 50)
    #     code_col = "code"
    #     print(f'\n\ndataframe columns: {df.columns}\n')
    #     # statement preprocessing
    #     if 'context' in list(df.columns):
    #         code_col = 'context'
    #         df[code_col] = df[code_col].apply(
    #             lambda x: re.sub(r"\s+", " ", str(x)).strip())

    #     # Step 1: drop duplicates from all rows
    #     len_s0 = len(df)
    #     # df = df.drop_duplicates(
    #     #     subset=["cwe", code_col],
    #     #     keep='first',
    #     #     ignore_index=True)
    #     df = df.drop_duplicates().reset_index(drop=True)

    #     len_s1 = len(df)
    #     print(f"Filter-1: #[{len_s0-len_s1} out of {len_s0}]"
    #           + "duplicates were dropped!")

    #     # Step 2: drop duplicates from ambiguous rows on the context column
    #     # (keeping only a first occurrence, i.e, vul/cwe sample)
    #     df = (
    #         df.sort_values(by="cwe", ascending=True)
    #         .drop_duplicates(subset=code_col,
    #                          keep="first",
    #                          ignore_index=True)
    #         # .reset_index(drop=True)
    #     )
    #     print(f"Filter-2: #[{len_s1-len(df)} out of {len_s1}]"
    #           + "ambiquous were dropped!")
    #     print("-" * 50 + "\n")
    #     return df

    def filter_results(self, df: pd.DataFrame()) -> pd.DataFrame():
        '''filter duplicates based on given columns'''
        print("-" * 50)
        check_cols = ['code', 'context', 'cwe']

        # table-wise columns to check duplicates
        check_cols = [x for x in df.columns if x in check_cols]
        # Step 1: drop duplicates from all rows
        len_s0 = len(df)
        df = df.drop_duplicates(
            subset=check_cols, keep='first').reset_index(drop=True)
        print(f"#[{len_s0-len(df)} out of {len_s0}] "
              + "duplicates were dropped!")
        print(f'cwe_values: {df.cwe.value_counts()}')
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


if __name__ == "__main__":
    dataset = 'raspberryZip'
    config = yaml.safe_load(open("ext_projects.yaml"))

    stat = config["save"]["statement"]
    fun = config["save"]["function"]
    binary_file = config["save"]["statement"].rsplit('.')[0] + '_Binary.csv'
    override = config["save"]["override"]

    if override is False:
        if os.path.exists(stat) or os.path.exists(fun):
            print(f"The statement/function dataset"
                  + "you want to create already exist: "
                  + "{stat}/{fun}\n provide another filename")
            exit(0)

    print('-'*30)
    print(f'Reading files: {stat} and {fun}...')
    df = pd.read_csv(stat, engine="c")
    dfm = pd.read_csv(fun, engine="c")
    print(f'Shape of the files: {df.shape} and {dfm.shape}')

    if not os.path.exists("data"):
        os.mkdir("data")

    # generate benign samples
    df_fun = gen_benign(dfm)
    df = df.append(df_fun).reset_index(drop=True)

    # remove duplicates
    df = filter_results(df)  # mutates df
    dfs = save_binary(binary_file, df)
