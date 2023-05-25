# Grepping flaws from the given project list.
# Merging output generated by FlawFinder and CppCheck
# Grepping functions from the vulnerability context of the file.
# file, function and statement-level information

import ast
import csv
import itertools
import os
import subprocess as sub
import sys
import tempfile
import time
import xml.etree.ElementTree as et
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import lizard
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
import yaml
from guesslang import Guess
from pylibsrcml import srcml

# User defined modules
from src.utility import Utility
# from src.gen_benign import drop_rows, gen_benign
# import (self.conn, change_status, get_status, query_project_table, show_shape, table_exists)
from src.sqlite import Database
from src.tools import SecTools


class Scanner:
    def __init__(self):
        self.cols_filter = [
            "full_parameters",
            "fan_in",
            "fan_out",
            "general_fan_out",
            "top_nesting_level",
            "defaultlevel",

        ]
        self.sect = SecTools()
        self.db = Database()
        self.conn = self.db.conn
        self.util = Utility()

# ================= Process CWE ==================================
    def extract_cwe(self, cwe) -> str:
        """ Extract CWE type information,
        In case of Rats tool's 'CWE-unknown' list, make it just a single item."""
        cwe = list(set(cwe)) if isinstance(cwe, list) else cwe

        if len(cwe) > 0 and isinstance(cwe, list):
            if len(cwe) > 1 and 'CWE-unknown' in cwe:
                # remove 'CWE-unknown' if the sample is already labeled as a known vulnerability.
                cwe.remove('CWE-unknown')

            if len(cwe) == 1:
                cwe = cwe[0]
        else:
            cwe = 'CWE-unknown'
        return str(cwe)

# ================= Extract Functions ==================================
    def extract_functions(self, source_file, lines, cwes, context, tool=['cppcheck']):
        """split the given file into a list of function blocks and return their metrics into a dataframe.
        <guru> I think there is a problem in lizard detecting the correct full_parameters
        either we have to concatenate two lines of full_parameters or ignore it and take it from long_name if needed.
        drop['full_parameters', 'fan_in', 'fan_out', 'general_fan_out'] because lizard has not properly
        implemented these parameters yet.
        check if the vulnerability content/statement appears in the function block or not.
        type of the vul line should be int and then lies in the function block.
        # """
        df_file = pd.DataFrame()

        # TODO: review this code now
        with open(source_file, "r", errors="surrogateescape") as fp:

            source_code = fp.read()
            liz_file = lizard.analyze_file.analyze_source_code(
                source_file, source_code)

            for ifun in range(len(liz_file.function_list)):
                vul_statements = []
                cwe = []

                fun = liz_file.function_list[ifun].__dict__
                df_file_fun = pd.DataFrame.from_dict(fun)

                start = int(fun["start_line"])
                end = int(fun["end_line"])
                fp.seek(0)  # moves the header to the initial point of the file

                fun_block = [line for line in itertools.islice(fp, start, end)]
                fp.seek(0)

                # check if any of the lines of the file belong to any functions
                for index, (l, c, cnt, t) in enumerate(zip(lines, cwes, context, tool)):

                    # checks vulnerability condition
                    if (isinstance(l, int)) and (start <= l < end):
                        vline = ''
                        if t.lower() == "cppcheck" or t.lower() == "rats":
                            vline = fp.readlines()[l]
                            fp.seek(0)

                        if vline != '' and cnt != cnt:
                            cnt = vline

                        vul_statements.append((cnt, c))
                        cwe.append(c)

                if len(cwe) == 0:
                    cwe.append('Benign')

                # rename filename to file to make it consistent with the statement table
                df_file_fun = df_file_fun.rename(columns={"filename": "file"})

                df_file_fun['code'] = fun["long_name"] + "".join(fun_block)
                df_file_fun["fun_name"] = fun["name"]
                df_file_fun["content"] = (
                    str(vul_statements) if len(vul_statements) > 0 else ""
                )
                df_file_fun["isVul"] = 1 if cwe else 0
                df_file_fun["cwe"] = self.extract_cwe(cwe)
                df_file_fun["project"] = self.url

                df_file = pd.concat([df_file, df_file_fun])

        # drop duplicates and keep a single row
        df_file = df_file.drop_duplicates(
            subset=['file', 'long_name', 'start_line', 'end_line', 'cwe'], keep='last').reset_index(drop=True)

        if set(self.cols_filter).issubset(set(list(df_file.columns))):
            df_file = df_file.drop(self.cols_filter, axis=1)
        return df_file

# ================= Compose Flaws ==================================
    def compose_file_flaws(self, file, zip_obj=None):
        """convert zipped file stream - tempfile to pandas dataframe."""
        file_content = "".encode("utf-8")
        df_flaw = pd.DataFrame()
        df_fun = pd.DataFrame()

        if zip_obj:
            with zip_obj.open(file) as fc:
                file_content = fc.read()
        else:
            with open(file) as fc:
                try:
                    # use encoding otherwise, flawfinder shows encoding error for some files.
                    file_content = fc.read().encode("utf-8")
                except UnicodeDecodeError as e:
                    file_content = fc.read().encode("latin-1")
                    print(f"UnicodeDecodeError: {e} for file: {file}")

        fp = tempfile.NamedTemporaryFile(suffix="_Flawfinder", prefix="File_")

        # deal with the temp file of the extracted zipped file
        try:
            fp.write(file_content)
            # merge the results generated by all the tools
            df_flaw = self.sect.merge_tools_result(fname=file)

            if len(df_flaw):
                df_fun = self.extract_functions(
                    source_file=file,
                    lines=list(df_flaw.line),
                    cwes=list(df_flaw.cwe),
                    context=list(df_flaw.context),
                    tool=list(df_flaw.tool),
                )
        except OSError:
            print("Could not open/read file:", file)
            sys.exit(1)
        finally:
            fp.close()
        return df_flaw, df_fun

    def check_internet(self, url):
        response = requests.get(self.url)
        return True if response.status_code < 400 else False

    def retrieve_zip(self, url):
        """Fetching list of C/C++ files from zip file of the project url."""
        if check_internet(self.url):
            r = requests.get(self.url)
            # BytesIO keeps the file in memory
            return ZipFile(BytesIO(r.content))
        else:
            print("Internet is not working!")
            return None

# ================= Convert Project to DB ==================================
    def project2db(self, url, status):
        """concatenate all the output dataframes of all the files"""
        print("\n" + "-" * 50)

        # initialization of empty dataframes
        df_flaw_prj = pd.DataFrame()
        df_fun_prj = pd.DataFrame()
        selected_files = []
        zipobj = None
        fc = 0
        self.url = url
        print("Scanning for flaws (takes a while)....")

        if os.path.isdir(self.url):
            files = [str(f) for f in Path(self.url).rglob("*.*")]
            selected_files = [
                x for x in files if self.sect.guess_pl(x) in self.sect.pl_list]
        else:
            zipobj = self.retrieve_zip(self.url)
            files = zipobj.namelist()
            selected_files = [
                x for x in files if self.sect.guess_pl(x, zipobj) in self.sect.pl_list]

        print(f'#files in the project: {len(selected_files)}')

        # TODO: incremental fetching of the information.
        if self.db.table_exists('statement'):
            scanned_files = set(
                list(pd.read_sql('SELECT file from statement', self.conn)['file']))
            selected_files = list(set(selected_files) - scanned_files)

            print(f'#files already scanned: {len(scanned_files)}')
            print(f'#files to scan: {len(selected_files)}\n')

        if len(selected_files) > 0:
            change_stat = True if status == 'Not Started' else False

            # iterate on every unique file
            for file in list(set(selected_files)):
                df_benign = pd.DataFrame()

                df_flaw_file, df_fun_file = self.compose_file_flaws(
                    file, zipobj)

                if len(df_flaw_file) > 0 and df_flaw_file.file.isna().any() == False:

                    df_flaw_file = df_flaw_file.astype(str)

                    # Add benign samples to the statement-level data
                    if len(df_fun_file) > 0:
                        df_benign = self.util.gen_benign(df_fun_file)
                        df_flaw_file = pd.concat([df_flaw_file, df_benign])

                    df_flaw_file['project'] = self.url

                    df_flaw_file.to_sql('statement', con=self.conn,
                                        if_exists='append', index=False)

                    # Change status to 'In Progress' once if it is 'Not Started'
                    if change_stat:
                        self.db.change_status(self.url, "In Progress")
                        change_stat = False

                if len(df_fun_file) > 0:
                    df_fun_file = df_fun_file.astype(str)
                    df_fun_file['project'] = self.url
                    df_fun_file.to_sql('function', con=self.conn,
                                       if_exists='append', index=False)

                # verbose on every 100 files
                if fc % 100 == 0:
                    print(f"\n#Files: {fc + 1} file(s) completed!")
                    print("Continue gathering function metrics....\n")

                fc = fc + 1

            print("\n" + "-" * 10 + f" Project Report: {self.url} " + "-" * 10)
            self.db.show_shape('statement', project=self.url)
            self.db.show_cwe_benign('statement')

            self.db.show_shape('function', project=self.url)
            self.db.show_cwe_benign('function')
            print("-" * 50 + "\n")

        else:
            print(f"No file in the specified project \
                    of the given PL list types: {pl_list}")

        # Change the project status to complete
        self.db.change_status(self.url, 'Complete')

        # self.db.get_status(self.url)
        # return df_flaw_prj, df_fun_prj


# ================= Scan Every Project ==================================


    def iterate_projects(self, prj_dir_urls):
        """iterate on every project"""
        df_flaw = pd.DataFrame()
        df_fun = pd.DataFrame()

        df_prj = self.db.query_project_table()
        prj_dir_urls = list(df_prj['project'])

        for prj in prj_dir_urls:
            stat = self.db.get_status(prj)

            if stat == 'Complete':
                print(f"The project had been already scanned!")

            elif stat == 'Not Started' or stat == 'In Progress':
                if os.path.isfile(prj) == False or os.path.isdir(prj) == False:
                    self.project2db(prj, stat)
                else:
                    print("Non-zipped project!")
            else:
                print(f"Project: {prj} is not in the list!")
            print('\n\n')

        print("=" * 50)
        print("\n\n" + "=" * 15 + " Final Composite Report " + "=" * 15)

        # show the final report of the project
        self.db.show_shape(table_name='statement', project='all')
        self.db.show_cwe_benign('statement')

        self.db.show_shape(table_name='function', project='all')
        self.db.show_cwe_benign('function')
        print("=" * 50 + "\n")

        self.db.query_project_table()
        self.conn.commit()
        self.db.cursor.close()

        return df_flaw, df_fun


# ================= Refine Data ==================================

    def refine_data(self, table_name):
        """refine the data, and filter out duplicates"""
        df = pd.read_sql(f"SELECT * FROM {table_name}", con=self.db.conn)
        print("\n\n" + "="*15 + f" Refining Data: {table_name} " + "="*20)
        print('Before filtering:')
        self.util.show_info_pd(df, table_name)

        # filter the results
        df = self.util.filter_results(df)

        print('After filtering:')
        self.util.show_info_pd(df, table_name)
        print("="*50)
        print("\n" + "-"*50)
        df.to_sql(table_name,
                  con=self.conn,
                  if_exists='replace',
                  index=False)
        print(f"Table: [{table_name}] is updated!")
        print("-"*50 + "\n")
        return df


if __name__ == "__main__":
    # The list of the URL links of the project zip files.
    config = yaml.safe_load(open("ext_projects.yaml"))

    flaw_file = config["save"]["statement"]
    metric_file = config["save"]["function"]
    override = config["save"]["override"]

    if override == False and (os.path.exists(flaw_file) or os.path.exists(metric_file)):
        print(
            f"The flaw/metric data file you want to create already \
                    exists: {flaw_file}/{metric_file}\n provide another \
                        filename or set override=True in the config file."
        )
        exit(0)

    scan = Scanner()
    scan.iterate_projects(config["projects"])
    scan.refine_data('statement')
    scan.refine_data('function')
