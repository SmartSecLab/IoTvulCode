## Grepping functions from the vulnerability context of the file.
# file, function and statement-level information

import collections
import pandas as pd
from matplotlib import pyplot as plt
import json
import ast
import re
import os
import csv
import subprocess
import requests

# import requests
import tempfile
from io import BytesIO, StringIO
from zipfile import ZipFile
from guesslang import Guess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import lizard
import subprocess as sub
from pylibsrcml import srcml
import itertools


pl_list = ["C", "C++"]


def check_internet(url):
    response = requests.get(url)
    return True if response.status_code < 400 else False


def retrieve_zip(url):
    """Fetching list of C/C++ files from zip file of the project url."""
    if check_internet(url):
        r = requests.get(url)
        # BytesIO keeps the file in memory
        return ZipFile(BytesIO(r.content))
    else:
        print("Internet is not working!")
        return None


# # TBD: Function Under Construction:
# def srcML_funs(file):
#     """find function blocks of the given file using srcML"""
#     fun_ptn = "string(//src:function)"
#     funblk_ptn = "string((//src:function/src:name))"
#     # file_ptn = "string(//src:unit/@filename)"

#     # cmd = sub.Popen(["srcml", "--xpath", fun_ptn, file], stderr=sub.STDOUT)
#     # out, err = cmd.communicate()
#     cmd = ["srcml", "--xpath", funblk_ptn, xml_file]
#     process = sub.Popen(cmd, stderr=sub.STDOUT)
#     return process


def guess_pl(file, zip_obj=None):
    """guess programming language of the input file."""
    guess = Guess()
    if zip_obj is not None:
        # extract a specific file from the zip container
        with zip_obj.open(file, "r") as f:
            lang = guess.language_name(f.read())
    else:
        with open(file, "r", encoding="unicode_escape") as f:
            lang = guess.language_name(f.read())
    return lang


def apply_flawfinder(file_or_dir):
    """find flaws in the file using CppCheck tool"""
    if os.path.isfile(file_or_dir):
        cmd = "flawfinder --csv " + file_or_dir
    elif os.path.isdir(file_or_dir):
        cmd = "flawfinder --csv --inputs " + file_or_dir
    else:
        print("Please provide a valid project dir/file/link!")

    process = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
    output = process.stdout.read()
    df = pd.read_csv(StringIO(str(output, "utf-8")))
    return df.reset_index(drop=True)


def file2df(file, zip_obj=None):
    """convert zipped file stream - tempfile to pandas dataframe."""
    file_content = ""
    df_flaw = pd.DataFrame()
    df_metrics = pd.DataFrame()

    if zip_obj:
        # io.StringIO(sf.read().decode("utf-8")).read()
        with zip_obj.open(file) as fc:
            # file_content = fc.read().encode('UTF-8')
            file_content = fc.read()
    else:
        with open(file) as fc:
            file_content = fc.read().encode("utf-8")

    fp = tempfile.NamedTemporaryFile(suffix="_Flawfinder", prefix="Filename_")

    # deal with the temp file of extracted zipped file
    try:
        fp.write(file_content)
        fp.seek(0)  # move reader's head to the initial point of the file.
        file_name = fp.name
        df_flaw = apply_flawfinder(file_or_dir=file_name)

        if len(df_flaw) > 0:
            df_metrics = file2metrics(source_file=file_name, df_flaw=df_flaw)
            print(f"Shape of the found flaws data of the file: {df_flaw.shape}")
            print(f"Shape of the file flaws metrics: {df_metrics.shape}")

    except OSError:
        print("Could not open/read file:", fp)
        sys.exit(1)
    finally:
        fp.close()
    return df_flaw, df_metrics


def urlzip2df(url):
    """concatenate all the output dataframes of all the files"""
    print("=" * 35)
    print(
        "Generating composite dataframe from the given project URL of zip file...\n",
        url,
    )
    print("=" * 35)

    zipobj = retrieve_zip(url)
    files = zipobj.namelist()
    selected_files = [x for x in files if guess_pl(x, zipobj) in pl_list]

    if selected_files:
        df_flaw_prj = pd.DataFrame()
        df_metrics_prj = pd.DataFrame()

        for i in range(len(selected_files)):
            df_flaw_file, df_metrics_file = file2df(selected_files[i], zipobj)
            df_flaw_prj = pd.concat([df_flaw_prj, df_flaw_file])
            df_metrics_prj = pd.concat([df_metrics_prj, df_metrics_file])

        print("Shape of the FlawFinder data of the project:", df_flaw_prj.shape)
        print("Shape of the function level metrics of prj:", df_metrics_prj.shape)
        return df_flaw_prj.reset_index(drop=True), df_metrics_prj.reset_index(drop=True)
    else:
        print(f"No file in the specified project  by the given PL: {pl_list}")
        return None, None


if __name__ == "__main__":
    # Example prj: The list of the URL links of the project zip files.
    prj_dir_urls = [
        # "https://sourceforge.net/projects/contiki/files/Contiki/Contiki%202.4/contiki-sky-2.4.zip/download",
        "data/projects/contiki-2.4/apps/"
    ]
    df_flaw = pd.DataFrame()
    df_metrics = pd.DataFrame()

    # iterate on every projects:
    for prj in prj_dir_urls:
        df_flaw_prj, df_met_prj = urlzip2df(prj)
        df_flaw = pd.concat([df_flaw_prj, df_metrices])
        df_metrices = pd.concat([df_metrices, df_met_prj])

    if len(df_flaw) > 0 and len(df_metrics) > 0:
        df_flaw.to_csv("data/contiki24_flaw.csv")
        df_metrics.to_csv("data/contiki24_metrics.csv")
    else:
        print("The given project URL does not have any specified files to analyze!")
