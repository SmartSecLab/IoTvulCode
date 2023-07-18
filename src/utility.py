# Statement-level:
# Analysis of IoTvulCode tool generated output for feeding non-vul statements:

import os
import random
import re
import subprocess as sub
import warnings
import xml.etree.ElementTree as et

import lizard
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import yaml
from pylibsrcml import srcml

warnings.filterwarnings("ignore")


class Utility():
    def __init__(self):
        self.config = {}
        # with open("ext_projects.yaml", "r") as stream:
        #     try:
        #         self.config = yaml.safe_load(stream)
        #     except Exception as e:
        #         print("Error loading configuration file: " + str(e))

    def load_config(yaml_file):
        """
        load the yaml file and return a dictionary
        """
        with open(yaml_file, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                return exc

    def get_benign_context(self, row):
        """
        filter all lines if it is less than min threshold
        randomly shuffled lines
        """
        threshold = self.config["save"]["threshold_lines"]
        benign_ratio = self.config["save"]["benign_ratio"]
        seed = self.config["save"]["seed"]

        df = pd.DataFrame()

        # threshold 7 to filter short functions
        lines = [x for x in enumerate(
            row["code"].splitlines()) if len(x[1]) > threshold]

        # randomly shuffle lines and takes 1/4 of the total number of lines.
        random.seed(seed)
        lines = random.sample(
            population=lines, k=int(len(lines) * benign_ratio))

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

        line_col = df["line"].astype(int) + int(row["start_line"])
        max_line = max(list(line_col)) if list(line_col) else 0
        end_line = int(row["end_line"])

        # print(f"max of lines: {max_line} and end_line: {end_line}")
        assert max_line <= end_line, "Line number shouldn't exceed function length!"
        df["line"] = line_col
        return df

    def gen_benign(self, dfm):
        """create benign samples to the dataframe"""
        # print("-" * 50)
        # print("Generating benign samples (wait)...")
        df_fun = pd.DataFrame()

        for i in range(len(dfm)):
            df_get = self.get_benign_context(dict(dfm.iloc[i]))
            df_fun = df_fun.append(df_get).reset_index(drop=True)

        # print("#Benign samples generated: ", len(df_fun))
        # print("-" * 50)
        return df_fun

    def filter_results(self, df):
        """apply several filters to the dataframe"""
        print("\n" + "-" * 50)
        code_col = "code"

        if 'context' in list(df.columns):
            code_col = 'context'
            df[code_col] = df[code_col].apply(
                lambda x: re.sub(r"\s+", " ", str(x)).strip())

        # Step 1: drop duplicates from all rows
        len_s0 = len(df)
        df = df.drop_duplicates(
            subset=["cwe", code_col]).reset_index(drop=True)
        len_s1 = len(df)
        print(
            f"Total {len_s0-len_s1} duplicate samples were dropped from {len_s0} samples.")

        # Step 2: drop duplicates from ambiguous rows on the context column
        # (keeping only a first occurrence, i.e, vul/cwe sample)
        df = (
            df.sort_values(by="cwe", ascending=True)
            .drop_duplicates(subset=code_col, keep="first")
            .reset_index(drop=True)
        )
        print(
            f"Total {len_s1-len(df)} ambiquous samples were dropped from {len_s1} samples.")
        print("-" * 50 + "\n")
        return df

    def show_info_pd(self, df, name):
        print(
            f"\nShape of the {name}-level metrics of all the projects: {df.shape}")
        print(
            f"#vulnerable: {len(df[df.cwe!='Benign'])}")
        print(
            f"#benign: {len(df[df.cwe=='Benign'])}\n")

    def save_binary(filename, dfs):
        """save a dataframe to a binary file"""
        dfs["isMalicious"] = dfs["cwe"].apply(
            lambda x: 1 if x != "benign" else 0)
        dfs = dfs.rename(columns={"context": "code"})
        dfs[["code", "isMalicious"]].to_csv(filename, index=False)
        return dfs[["code", "isMalicious"]]


if __name__ == "__main__":

    dataset = 'raspberryZip'

    # stat = f"data/{dataset}_statement.csv"
    # fun = f"data/{dataset}_function.csv"
    # binary_file = f"data/{dataset}_binary.csv"

    config = yaml.safe_load(open("ext_projects.yaml"))

    stat = config["save"]["statement"]
    fun = config["save"]["function"]
    binary_file = config["save"]["statement"].rsplit('.')[0] + '_Binary.csv'

    override = config["save"]["override"]

    if override == False:
        if os.path.exists(stat) or os.path.exists(fun):
            print(
                f"The statement/function dataset you want to create already \
                    exist: {stat}/{fun}\n provide another filename"
            )
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
