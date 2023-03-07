## Grepping flaws from the given project list.
# Merging output generated by FlawFinder and CppCheck
## Grepping functions from the vulnerability context of the file.
# file, function and statement-level information

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import lizard
import subprocess as sub
from pylibsrcml import srcml
import os
import xml.etree.ElementTree as et
import sys
import csv
import requests
from io import BytesIO, StringIO
import numpy as np
import itertools
import tempfile
import yaml

from zipfile import ZipFile
from guesslang import Guess
from pathlib import Path


# user defined
from src.tools import apply_flawfinder, apply_cppcheck, apply_rats, merge_tools_result

pl_list = ["C", "C++", "CPP"]


def function_metrics(source_file, lines, cwes, context, tool=["cppcheck"]):
    """split the given file into a list of function blocks and return their metrics into a dataframe
    <guru> I think there is a problem in lizard detecting the correct full_parameters
    either we have to concatenate two lines of full_parameters or ignore it and take it from long_name if needed.
    drop['full_parameters', 'fan_in', 'fan_out', 'general_fan_out'] because lizard has not properly
    implemented these parameters yet."""

    # lines = [eval(l) for l in lines]  # python default casting to integer
    df_file = pd.DataFrame()

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
            for index, (l, c, t, cnt) in enumerate(zip(lines, cwes, context, tool)):
                fun_block = [line for line in itertools.islice(fp, start, end)]
                fp.seek(0)
                df_fun["code"] = fun_metrics["long_name"] + "".join(fun_block)

                # check if the vulnerability content/statement appear in the function block or not.
                # For 'is_vul' equals  True
                if start <= l <= end:
                    if t == "cppcheck":
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
                        vul_content = cnt
                    vul_content = ""
                    cwe.append(c)

                df_fun["fun_name"] = fun_metrics["name"]
                df_fun["content"] = (
                    str(vul_statements) if len(vul_statements) > 0 else ""
                )
                df_fun["is_vul"] = True if cwe else False

                # In case of Rats tool's 'unknown-vul' is list, make it just a single item.
                cwe = list(set(cwe))

                if len(cwe) > 0:
                    # df_fun['cwe'] = 'unknown'  if np.isnan(cwe).all() else str(cwe)
                    df_fun["cwe"] = (
                        "unknown_vul" if all(i != i for i in cwe) else str(cwe)
                    )
                else:
                    df_fun["cwe"] = "benign"

            df_file = pd.concat([df_file, df_fun])

    cols_filter = [
        "full_parameters",
        "fan_in",
        "fan_out",
        "general_fan_out",
        "top_nesting_level",
    ]
    df_file = df_file.drop_duplicates().reset_index(drop=True)

    if set(cols_filter).issubset(set(list(df_file.columns))):
        df_file = df_file.drop(cols_filter, axis=1)
    return df_file

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


def project_flaws(df):
    """find flaw entries of all the complete project scanning each unique file."""
    df_prj = pd.DataFrame()

    # iterate on every unique file
    for f in list(set(df.file)):
        lines = list(df[df.file == f]["line"])
        cwes = list(df[df.file == f]["cwe"])
        # vul_statements = list(df_flaw[df_flaw.file==x]['cwe'])
        # lines = [x[0] if len(x) == 1 else [x[0], x[1]] for x in lines]

        # TODO: check if any of the entries has multiple locations or lines
        # lines = [x[0] if len(x) == 1 else [x[0], x[1]] for x in lines]
        df_file = function_metrics(f, lines, cwes, tool="cppcheck")
        df_prj = pd.concat([df_prj, df_file])

    return df_prj.reset_index(drop=True).drop_duplicates()


def compose_file_flaws(file, zip_obj=None):
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
            # use encoding otherwise, flawfinder shows encoding error for some files.
            file_content = fc.read().encode("utf-8")

    fp = tempfile.NamedTemporaryFile(suffix="_Flawfinder", prefix="Filename_")

    # deal with the temp file of extracted zipped file
    try:
        fp.write(file_content)
        fp.seek(0)  # move reader's head to the initial point of the file.
        fname = fp.name

        # apply tools:
        df_ff = apply_flawfinder(file_or_dir=fname)
        df_cc = apply_cppcheck(file_or_dir=fname)
        df_rat = apply_rats(file_or_dir=fname)

        # merge the results of all tools
        df_flaw = merge_tools_result(df_ff, df_cc, df_rat)

        if len(df_flaw):
            df_metrics = function_metrics(
                source_file=fname,
                lines=list(df_flaw.line),
                cwes=list(df_flaw.cwe),
                context=list(df_flaw.context),
                tool=list(df_flaw.tool),
            )
    except OSError:
        print("Could not open/read file:", fp)
        sys.exit(1)
    finally:
        fp.close()
    return df_flaw, df_metrics


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


def guess_pl(file, zip_obj=None):
    """guess programming language of the input file.
    Recursively Remove .DS_Store which was introducing encoding error,
    https://jonbellah.com/articles/recursively-remove-ds-store
    ignore all files with . start and compiled sources
    TODO extract .zip file for further flaw finding
    """
    guess = Guess()
    try:
        if zip_obj is not None:
            # extract a specific file from the zip container
            with zip_obj.open(file, "r") as f:
                lang = guess.language_name(f.read())
        else:
            with open(file, "r") as f:
                lang = guess.language_name(f.read())
        return lang
    except Exception as e:
        print(f"Guesslang error: {e}")
        return "unknown"


def urlzip2df(url):
    """concatenate all the output dataframes of all the files"""
    print("\n" + "-" * 50)
    zipobj = None

    if os.path.isdir(url):
        print(f"Current project directory: {url}\nScanning for flaws....")
        files = [str(f) for f in Path(url).rglob("*.*")]
        selected_files = [x for x in files if guess_pl(x) in pl_list]
    else:
        print(f"Current project URL: {url}\nScanning for flaws....")
        zipobj = retrieve_zip(url)
        files = zipobj.namelist()
        selected_files = [x for x in files if guess_pl(x, zipobj) in pl_list]

    if selected_files:
        df_flaw_prj = pd.DataFrame()
        df_metrics_prj = pd.DataFrame()

        prj_count = 0
        # iterate on every unique file
        for file in list(set(selected_files)):
            df_flaw_file, df_metrics_file = compose_file_flaws(file, zipobj)
            df_flaw_prj = pd.concat([df_flaw_prj, df_flaw_file])
            df_metrics_prj = pd.concat([df_metrics_prj, df_metrics_file])

            prj_count = prj_count + 1
            if prj_count % 10 == 0:
                print(f"\n#Files: {prj_count}\nGathering file metrics....")

        print("\n" + "-" * 10 + " Project Report " + "-" * 10)
        print("Shape of the flaws data:", df_flaw_prj.shape)
        print("Shape of the function level metrics:", df_metrics_prj.shape)
        return df_flaw_prj.reset_index(drop=True), df_metrics_prj.reset_index(drop=True)
    else:
        print(f"No file in the specified project of the given PL list types: {pl_list}")
        return None, None


def iterate_projects(prj_dir_urls):
    """iterate on every project"""
    df_flaw = pd.DataFrame()
    df_metrics = pd.DataFrame()

    for prj in prj_dir_urls:
        if os.path.isfile(prj) != True or os.path.isdir(prj) != True:
            df_prj, df_prj_met = urlzip2df(prj)
            df_flaw = pd.concat([df_flaw, df_prj])
            df_metrics = pd.concat([df_metrics, df_prj_met])
        else:
            print("non-zipped prj")
    print("-" * 40)

    print("\n\n" + "=" * 20 + " Final Composite Report " + "=" * 20)
    if len(df_flaw) > 0 and len(df_metrics) > 0:
        print("Shape of the flaw data of all the projects:", df_flaw.shape)
        print(
            f"Shape of the function level metrics of all the \
            projects: {df_metrics.shape}"
        )
    else:
        print(
            "The given list of projects  doesn't have any specified files to analyze!"
        )
    return df_flaw, df_metrics


if __name__ == "__main__":
    # The list of the URL links of the project zip files.
    config = yaml.safe_load(open("ext_projects.yaml"))

    flaw_file = config["files"]["save_flaw"]
    metric_file = config["files"]["save_metrics"]
    override = config["files"]["override"]

    if not override:
        if os.path.exists(flaw_file) or os.path.exists(metric_file):
            print(
                f"The flaw/metric data file you want to create already \
                    exists: {flaw_file}/{metric_file}\n provide another filename"
            )
            exit(0)

    df_flaw, df_metrics = iterate_projects(config["projects"])

    print("=" * 50)
    if len(df_flaw):
        df_flaw.to_csv(flaw_file)
        print(f"The flaw data output is saved at {flaw_file}")
    if len(df_metrics):
        df_metrics.to_csv(metric_file)
        print(f"The fun metric data is saved at {metric_file}")
    print("=" * 50)
