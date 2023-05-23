# Grepping functions from the vulnerability context of the file.
# Fetching the functions which have given line context/statement.
# Parsing CppCheck output:


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
from io import BytesIO, StringIO

# from lxml import etree
import pandas as pd

# from pandas import DataFrame
import numpy as np
import subprocess as sub
from pathlib import Path
import sys
import shutil
import time
import yaml


class SecTools:
    def __init__(self):
        self.common_cols = ["file", "line", "column", "cwe", "note"]
        self.unique_cols = ["context", "defaultlevel", "level", "helpuri"]
        self.filter_cols = [
            "toolversion",
            "fingerprint",
            "ruleid",
            "suggestion",
            # "defaultlevel",
            # "level",
        ]
        self.pl_list = ["c", "c++", "cpp", "cxx", "cp", "h"]

    def guess_pl(self, file, zip_obj=None):
        """guess the programming language of the input file.
        Recursively Remove .DS_Store which was introducing encoding error,
        https://jonbellah.com/articles/recursively-remove-ds-store
        ignore all files with . start and compiled sources
        TODO extract .zip file for further flaw finding
        TODO fix: Empty source code provided
        """
        self.config = {}
        with open("ext_projects.yaml", "r") as stream:
            try:
                self.config = yaml.safe_load(stream)["save"]
            except Exception as e:
                print("Error loading configuration file: " + str(e))

        if self.config["apply_guesslang"]:
            guess = Guess()
            try:
                if zip_obj != None:
                    # extract a specific file from the zip container
                    with zip_obj.open(file, "r") as f:
                        lang = guess.language_name(f.read())
                else:
                    with open(file, "r") as f:
                        lang = guess.language_name(f.read())
                return lang.lower()
            except Exception as e:
                print(f"Guesslang error: {e}")
                return "unknown"
        else:
            pl = Path(file).suffix.replace(".", "").lower()
            return pl if pl in self.pl_list else "unknown"


############################## Applying CppCheck tool ##############################


    @staticmethod
    def fetch_location(flaw) -> dict:
        """get locations of all the error list generated by CppCheck"""
        dt_loc = {"file": [], "line": [], "column": [], "info": []}
        for loc in flaw.findall("location"):
            for key, val in (loc.attrib).items():
                dt_loc[key].append(val)

        # avoid list for single valued items;
        # TODO check if it applies for all the projects
        for key in dt_loc.keys():
            if len(dt_loc[key]) == 1:
                dt_loc[key] = dt_loc[key][0]
        # if len(dt_loc) > 1:
        # print(flaw)
        return dt_loc

    def xml2df_cppcheck(self, file) -> pd.DataFrame:
        """convert xml file of cppcheck to dataframe"""
        df = pd.DataFrame()
        if os.path.isfile(file):
            try:
                xtree = et.fromstring(open(file).read())

                for errors in xtree.findall(".//errors"):
                    for err in errors.findall("error"):
                        dt_err = err.attrib

                        # get the location of the vulnerable line content
                        dt_err.update(self.fetch_location(err))
                        df = pd.concat(
                            [df, pd.DataFrame([dt_err])], ignore_index=True
                        ).drop(columns=["file"], axis=1)

                df = df.rename(columns={"file0": "file"}
                               ).reset_index(drop=True)
            except Exception as e:
                print(f"Parsing error with CppCheck:  {e}")
        return df

    def apply_cppcheck(self, fname, xmlfile="data/output.xml") -> pd.DataFrame:
        """find flaws in the file using CppCheck tool
        example commands:
        !cppcheck --template=gcc ../data/projects/contiki-2.4/apps/ 2> err.txt
        !cppcheck --template="{file}; {line}; {severity}; {message}; {code}"
        --template-location=" {file};{line}; {info};{code}\n" <path> 2> err.txt
        """
        # cmd = ["cppcheck -a -f " + fname + " --xml 2> + " + xmlfile] # error on '-a'
        cmd = ["cppcheck -f " + fname + " --xml --xml-version=2 2> " + xmlfile]
        process = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
        # process.wait()  # let it complete the process, but it is very slow
        output = process.stdout.read()

        # TODO: try not to create output.xml file instead use BytesIO.
        df = self.xml2df_cppcheck(xmlfile)

        if len(df):
            # explode line row on each list items
            df = df.explode("line")
            df["line"] = df.line.astype(dtype=int, errors="ignore")
            df["tool"] = "CppCheck"

            # To make CWE column values uniform to FlawFinder output
            df["cwe"] = (
                "CWE-" + df["cwe"]
                if set(["cwe"]).issubset(df.columns)
                else "CWE-unknown"
            )
            df = df.reset_index(drop=True)
        return df

############################## Applying FlawFinder tool ##############################

    def apply_flawfinder(self, fname) -> pd.DataFrame:
        """find flaws in the file using CppCheck tool"""
        if os.path.isfile(fname):
            cmd = "flawfinder --csv " + fname
        elif os.path.isdir(fname):
            cmd = "flawfinder --csv --inputs " + fname
        else:
            print("Please provide a valid project dir/file/link!")

        process = sub.Popen(
            cmd,
            shell=True,
            stdout=sub.PIPE,
        )
        output = process.stdout.read().decode("utf-8")
        df = pd.read_csv(StringIO(output))

        if len(df) > 0:
            df["tool"] = "FlawFinder"
        return df.reset_index(drop=True)

############################## Applying Rats tool ##############################

    @staticmethod
    def xml2df_rats(xml) -> pd.DataFrame:
        """convert xml file of rats tool to dataframe"""
        df = pd.DataFrame()

        if isinstance(xml, str):
            xtree = et.fromstring(xml)

            for err in xtree.findall("vulnerability"):
                dt = {
                    "severity": err.find("severity").text,
                    "type": err.find("type").text if err.find("type") != None else None,
                    "message": err.find("message").text,
                }
                for loc in err.findall("file"):
                    dt["file"] = loc.find("name").text

                    for line in loc.findall("line"):
                        dt["line"] = line.text
                        df = pd.concat([df, pd.DataFrame([dt])],
                                       ignore_index=True)
        return df.reset_index(drop=True)

    def apply_rats(self, fname, xmlfile="output.xml") -> pd.DataFrame:
        """ The Rough Auditing Tool for Security is an open-source tool 
        developed by Secure Software Engineers
        https://security.web.cern.ch/recommendations/en/codetools/rats.shtml \
        For example: 
        `rats --quiet --xml -w 3 data/projects/contiki-2.4/apps/` 
        """
        # rats --quiet --xml -w 3 <dir_or_file>
        cmd = ["rats --quiet --xml -w 3 " + fname]
        process = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
        try:
            output = process.stdout.read().decode("utf-8")
        except Exception as e:
            print(f"Rats: {e}")

        df = self.xml2df_rats(output)

        if len(df):
            # RATS tool does not produce results with CWE type.
            df["cwe"] = "CWE-unknown"
            df["line"] = df.line.astype(int)
            df["tool"] = "Rats"
        return df.reset_index(drop=True)

############################## Merge the output of all tools ##############################

    @staticmethod
    def concat(*args):
        """merge two columns of the dataframw with numpy vectorize method"""
        concat_str = ""
        try:
            strs = [str(arg) for arg in args if not pd.isnull(arg)]
            concat_str = ",".join(strs) if strs else np.nan
        except Exception as e:
            print("Value Error: ", e)
            print(f"Args: {args}")
            print(concat_str)
        return concat_str

    def adjust_cols(self, df_ff, df_cc, df_rat):
        # Adjusting columns generated by FlawFinder, CppCheck and Rats tool
        df_ff = df_ff.rename(columns=str.lower, errors="ignore")
        df_ff = df_ff.rename(
            columns={"cwes": "cwe", "warning": "msg"}, errors="ignore")
        df_cc = df_cc.rename(
            columns={"info": "note", "id": "name"}, errors="ignore")
        df_rat = df_rat.rename(
            columns={"message": "msg", "type": "category"}, errors="ignore")

        # CppCheck: As we checked, 'msg' and 'verbose' columns have the same entries,
        # so let's keep only 'msg'.
        df_cc = (
            df_cc.drop(columns=["verbose"], axis=1, errors="ignore")
            # if "verbose" in list(df_cc.columns)
            # else df_cc
        )
        # do this after merging 'suggestion to 'note' column
        if len(df_ff) > 0:
            # np_concat = np.vectorize(concat)
            # df_ff["note"] = np_concat(df_ff["suggestion"], df_ff["note"])
            df_ff["note"] = (
                df_ff["suggestion"].astype(
                    str) + "  " + df_ff["note"].astype(str)
            )
            df_ff = df_ff.drop(
                columns=["suggestion", "note"], axis=1, errors="ignore")
        return df_ff, df_cc, df_rat

    def merge_tools_result(self, fname) -> pd.DataFrame:
        """merge dataframe generated by FlawFinder and CppCheck tools"""
        df_merged = pd.DataFrame()

        # apply tools:
        df_ff = self.apply_flawfinder(fname=fname)
        df_cc = self.apply_cppcheck(fname=fname)
        df_rat = self.apply_rats(fname=fname)

        if len(df_ff) > 0 or len(df_cc) > 0 or len(df_rat) > 0:
            df_ff, df_cc, df_rat = self.adjust_cols(df_ff, df_cc, df_rat)
            df_merged = pd.concat([df_ff, df_cc, df_rat]
                                  ).reset_index(drop=True)
            df_merged = df_merged.drop(columns=self.filter_cols,
                                       axis=1, errors="ignore")
            # print(f"columns of merged dataframe: \n{df_merged.columns}")

            # add severity column if not present in the merged dataframe.
            if 'severity' not in list(df_merged.columns):
                df_merged['severity'] = '-'
            if 'category' not in list(df_merged.columns):
                df_merged['category'] = '-'
            if 'msg' not in list(df_merged.columns):
                df_merged['msg'] = '-'
            if 'column' not in list(df_merged.columns):
                df_merged['column'] = '-'
            if 'context' not in list(df_merged.columns):
                df_merged['context'] = '-'
            if 'helpuri' not in list(df_merged.columns):
                df_merged['helpuri'] = '-'
            if 'defaultlevel' not in list(df_merged.columns):
                df_merged['defaultlevel'] = '-'
            if 'level' not in list(df_merged.columns):
                df_merged['level'] = '-'
            if 'note' not in list(df_merged.columns):
                df_merged['note'] = '-'
            if 'name' not in list(df_merged.columns):
                df_merged['name'] = '-'
            if 'type' not in list(df_merged.columns):
                df_merged['type'] = '-'

            # if 'defaultlevel' is in list(df_merged.columns):
            #     df_merged = df_merged.drop(columns=['defaultlevel'], axis=1, errors="ignore")
            # print(
            #     f"Columns after adding dummies: \n{df_merged.columns}")

            # Necessary columns
            df_merged = df_merged[df_merged["line"].notna()]
            df_merged = df_merged[df_merged["cwe"].notna()]

            # '-' for empty cells
            df_merged = df_merged.fillna("-")

            # print(
            #     f"Merged dataframe: \n{df_merged[['line', 'defaultlevel', 'level', 'severity', 'cwe']]}")
            # print("==" * 30)

        return df_merged


if __name__ == "__main__":
    test_file = "data/projects/contiki-2.4/apps/webbrowser/www.c"
    test_dir = "data/projects/contiki-2.4/"
    df_flaw = pd.DataFrame()
    st = SecTools()

    df_flaw = st.apply_flawfinder(test_file)
    # df_flaw = st.merge_tools_result(test_dir)
    print(df_flaw)
