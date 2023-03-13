# Analysis of IoTvulCode tool generated output for feeding non-vul statements:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import lizard
import subprocess as sub
from pylibsrcml import srcml
import os
import re
import xml.etree.ElementTree as et
import warnings
import random

warnings.filterwarnings("ignore")


def get_benign_context(row):
    """
    filter all lines if it is less than min threshold
    randomly suffled lines
    """
    df = pd.DataFrame()
    lines = [x for x in enumerate(row["code"].splitlines()) if len(x[1]) > 7]
    # lines = [(x[0], str(x).strip()) for x in lines]

    random.seed(0)
    lines = random.sample(population=lines, k=int(len(lines) / 2))

    # TODO: remove the ambiguous vul line from the 'benign' lines if present
    # vul_line = df.line[i]
    # lines = [x for x in lines if x[0]!=vul_line]

    ## convert it to dataframe and add additional columns
    df = pd.DataFrame(lines, columns=["line", "context"])
    # remove leading and trailing whitespace
    df["context"] = df["context"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    df["cwe"] = "benign"
    df["tool"] = "sampling"
    df["file"] = row["filename"]
    line_col = df["line"].astype(int) + int(row["start_line"])

    max_line = max(list(line_col)) if list(line_col) else 0
    end_line = int(row["end_line"])

    # print(f"max of lines: {max_line} and end_line: {end_line}")
    assert max_line <= end_line, "Line number shouldn't exceed function length!"
    df["line"] = line_col
    return df


def drop_rows(df):
    """applied several filters to the dataframe"""
    df["context"] = df["context"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    len_s0 = len(df)
    # Step 1: drop duplicates from all rows
    df = df.drop_duplicates(subset=["cwe", "context"]).reset_index(drop=True)
    len_s1 = len(df)
    print(f"{len_s0-len_s1} duplicate samples were dropped from {len_s0} samples.")

    # Step 2: drop duplicates from ambiguous rows on context column
    ## (keeping only a first occurrence, i.e, vul/cwe sample)
    df = (
        df.sort_values(by="cwe", ascending=True)
        .drop_duplicates(subset="context", keep="first")
        .reset_index(drop=True)
    )
    len_s2 = len(df)
    print(f"{len_s1-len_s2} ambiquous samples were dropped from {len_s1} samples.")
    print("-" * 50)
    return df


def gen_benign(dfm):
    """create benign samples to the dataframe"""
    print("-" * 50)
    print("#samples: ", len(dfm))
    print("Generating benign samples...")
    df_fun = pd.DataFrame()
    for i in range(len(dfm)):
        df_get = get_benign_context(dict(dfm.iloc[i]))
        df_fun = df_fun.append(df_get).reset_index(drop=True)
    print("#benign samples generated: ", len(df_fun))
    print("-" * 50)
    return df_fun


def save_binary(filename, dfs):
    """save a dataframe to a binary file"""
    dfs["isMalicious"] = dfs["cwe"].apply(lambda x: 1 if x != "benign" else 0)
    dfs = dfs.rename(columns={"context": "code"})
    print('columns: ', dfs.columns)
    dfs[["code", "isMalicious"]].to_csv(filename, index=False)
    return dfs[["code", "isMalicious"]]


if __name__ == "__main__":
    df = pd.read_csv("data/contiki-master_flaw.csv")
    dfm = pd.read_csv("data/contiki-master_metrics.csv")

    if not os.path.exists("data"):
        os.mkdir("data")

    # df.context.str.len().sort_values(ascending=False).reset_index(drop=True).plot(kind='box')
    df_fun = gen_benign(dfm)
    df = df.append(df_fun).reset_index(drop=True)

    df = drop_rows(df)  # mutates df
    dfs = save_binary("data/contiki-master_Binary1.csv", df)
