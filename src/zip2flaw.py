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


# TBD: Function Under Construction:
def srcML_funs(file):
    """finds function blocks of the given file using srcML"""
    fun_ptn = "string(//src:function)"
    funblk_ptn = "string((//src:function/src:name))"
    # file_ptn = "string(//src:unit/@filename)"

    # cmd = sub.Popen(["srcml", "--xpath", fun_ptn, file], stderr=sub.STDOUT)
    # out, err = cmd.communicate()
    cmd = ["srcml", "--xpath", funblk_ptn, xml_file]
    process = sub.Popen(cmd, stderr=sub.STDOUT)
    return process


# # Fetching the functions which have given line context/statement.
# def file2metrics(source_file, df_flaw):
#     """ split the given file into a list of function blocks and return their metrics into a dataframe
#     """
#     lines = list(set(list(df_flaw.Line)))
#     df = pd.DataFrame()

#     with open(source_file, "r") as fp:
#         liz_file = lizard.analyze_file.analyze_source_code(source_file,  fp.read())

#         for x in range(len(liz_file.function_list)):

#             fun_metrics = liz_file.function_list[x].__dict__
#             df_fun = pd.DataFrame()
#             df_fun = pd.DataFrame.from_dict(fun_metrics)

#             start = int(fun_metrics['start_line'])
#             end = int(fun_metrics['end_line'])
#             fp.seek(0) # move header to the initial point of the file

#             vul_statement, cwe, vul_bool = '', [], False

#             for l in lines:
#                 code_lines = [line for line in itertools.islice(fp, start, end)]
#                 df_fun['code'] =  fun_metrics['long_name'] + ''.join(code_lines)

#                 # check if the vulnerability content/statement appear in the function block or not.
#                 if start <= l <= end:
#                     vul_bool =  True
#                     # vul_statement = vul_statement + ' \n ' +  df_flaw[df_flaw.Line==l]['Context'].values[0]
#                     vul_statement = df_flaw[df_flaw.Line==l]['Context'].values[0]
#                     vul_type = df_flaw[df_flaw.Line==l]['CWEs'].values[0]
#                     cwe.append((vul_type, vul_statement))

#                 df_fun['CWEs'] = str(cwe)
#                 # df_fun['vul_statements'] = vul_statement

#             df_fun['is_vul'] = vul_bool
#             df = pd.concat([df, df_fun])

#     print(f'Dataframe of a file: {source_file}, \n {df}')

#     # <guru> I think there is a problem in lizard detecting the correct full_parameters
#     # either we have to concatenate two lines of full_parameters or ignore it and take it from long_name if needed.
#     # drop['full_parameters', 'fan_in', 'fan_out', 'general_fan_out'] because lizard has not properly
#     # implemented these parameters yet.

#     cols_filter = ['full_parameters', 'fan_in', 'fan_out', 'general_fan_out']
#     df = df.drop(cols_filter, axis=1).drop_duplicates().reset_index(drop=True)
#     print('Shape of the dataframe: ', df.shape)
#     return df


def file2metrics(source_file, lines, cwes, tool="cppcheck"):
    """split the given file into a list of function blocks and return their metrics into a dataframe"""
    lines = [eval(l) for l in lines]  # python default casting to integer
    df = pd.DataFrame()

    with open(source_file, "r") as fp:
        liz_file = lizard.analyze_file.analyze_source_code(source_file, fp.read())

        for ifun in range(len(liz_file.function_list)):
            fun_metrics = liz_file.function_list[ifun].__dict__
            df_fun = pd.DataFrame.from_dict(fun_metrics)

            start = int(fun_metrics["start_line"])
            end = int(fun_metrics["end_line"])
            fp.seek(0)  # move header to the initial point of the file

            vul_content = ""
            vul_statements = []
            cwe = []
            vul_bool = False

            # check if any of the lines of the file belong to any functions
            for index, (l, c) in enumerate(zip(lines, cwes)):
                fun_block = [line for line in itertools.islice(fp, start, end)]
                fp.seek(0)
                df_fun["code"] = fun_metrics["long_name"] + "".join(fun_block)

                # check if the vulnerability content/statement appear in the function block or not.
                # For 'is_vul' equals  True
                if start <= l <= end:
                    if tool == "cppcheck":
                        # option 1
                        vul_content = fun_block[l - start]
                        vul_statements.append(vul_content)
                        fp.seek(0)

                        # option 2 - can be removed one if there is no error at the end
                        vul_stat2 = fp.readlines()[l]

                        assert (
                            vul_stat2 == vul_content
                        ), "Cross-check why two vul statements are different!"
                        fp.seek(0)
                    else:
                        # ToDo: take actual Context from FlawFinder's result
                        vul_content = df_flaw[df_flaw.line == l]["Context"].values[0]

                    cwe.append(c)

                df_fun["fun_name"] = fun_metrics["name"]
                df_fun["content"] = (
                    str(vul_statements) if len(vul_statements) > 0 else ""
                )
                df_fun["is_vul"] = True if cwe else False

                if len(cwe) > 0:
                    # df_fun['cwe'] = 'unknown'  if np.isnan(cwe).all() else str(cwe)
                    df_fun["cwe"] = "unknown" if all(i != i for i in cwe) else str(cwe)
                else:
                    df_fun["cwe"] = "benign"

            df = pd.concat([df, df_fun])

    # <guru> I think there is a problem in lizard detecting the correct full_parameters
    # either we have to concatenate two lines of full_parameters or ignore it and take it from long_name if needed.
    # drop['full_parameters', 'fan_in', 'fan_out', 'general_fan_out'] because lizard has not properly
    # implemented these parameters yet.

    cols_filter = [
        "full_parameters",
        "fan_in",
        "fan_out",
        "general_fan_out",
        "top_nesting_level",
    ]
    df = df.drop(cols_filter, axis=1).drop_duplicates().reset_index(drop=True)
    return df


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


def find_flaw(file_or_dir):
    """find flaws ini the file using flawfinder tool
    return : flawfinder output as a CSV file.
    Usage: cmd = 'flawfinder --csv --inputs ' + path + ' >> output.csv'
    """
    if os.path.isfile(file_or_dir):
        cmd = "flawfinder --csv " + file_or_dir
    elif ps.path.isdir(file_or_dir):
        cmd = "flawfinder --csv --inputs " + file_or_dir

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = process.stdout.read()
    return pd.read_csv(StringIO(str(output, "utf-8")))


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
            file_content = fc.read().encode("UTF-8")

    fp = tempfile.NamedTemporaryFile(suffix="_Flawfinder", prefix="Filename_")

    # deal with the temp file of extracted zipped file
    try:
        fp.write(file_content)
        fp.seek(0)  # move reader's head to the initial point of the file.
        file_name = fp.name
        df_flaw = find_flaw(file_or_dir=file_name).reset_index(drop=True)

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
        print(
            "Shape of the function level metrics data of the project:",
            df_metrics_prj.shape,
        )
        return df_flaw_prj.reset_index(drop=True), df_metrics_prj.reset_index(drop=True)
    else:
        print(
            "There is not any files in the project specified by the given programming language: {pl_list}"
        )
        return None, None


if __name__ == "__main__":
    # Example prj: The list of the URL links of the project zip files.
    url = [
        "https://sourceforge.net/projects/contiki/files/Contiki/Contiki%202.4/contiki-sky-2.4.zip/download"
    ]

    df_flaw, df_metrics = urlzip2df(url)

    if len(df_flaw) > 0 and len(df_metrics) > 0:
        df_flaw.to_csv("data/contiki24_flaw.csv")
        df_metrics.to_csv("data/contiki24_metrics.csv")
    else:
        print("The given project URL does not have any specified files to analyze!")
