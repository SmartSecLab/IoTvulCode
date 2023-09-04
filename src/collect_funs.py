# -*- coding: utf-8 -*-
"""
grepping function from the given file of the code. 
"""

import itertools
import os
import random
import subprocess as sub
import sys
import tempfile
import time
import xml.etree.ElementTree as et
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import lizard
import pandas as pd
import requests
import tqdm
import yaml
from humanfriendly import format_timespan
from pylibsrcml import srcml

from src.analyzers import Analyzers
# User defined modules
from src.src2funs import Src2Funs


class FunsCollector:
    """
    this class greps functions from the given file of the code. 
    """

    def __init__(self):
        self.srcML = Src2Funs()
        pass

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

    def is_fun_vul(self, fun, vul_statement):
        """check if the vulnerability content/statement appears 
        in the function block or not."""
        vul_bool = False
        if len(vul_statement) > 1:
            if ''.join(vul_statement.split()) in ''.join(fun.split()):
                vul_bool = True
        return vul_bool

    # def extract_functions_expensive(self, source_file, lines, cwes, context, tool=['cppcheck']):
    #     """split the given file into a list of function blocks
    #     and return their metrics into a dataframe."""
    #     df = pd.DataFrame()
    #     all_funs = self.srcML.src2src_functions(src=source_file)

    #     if len(all_funs) > 0:
    #         # most expensive steps below
    #         for fun in all_funs:
    #             for index, (line, cwe, vul_statement, _) in enumerate(zip(lines, cwes, context, tool)):
    #                 if isinstance(line, int):
    #                     row = {
    #                         'file': source_file,
    #                         'code': fun
    #                     }
    #                     if self.is_fun_vul(fun, vul_statement):
    #                         row['context'] = vul_statement
    #                         row['cwe'] = cwe
    #                         row['isVul'] = 1
    #                     else:
    #                         row['context'] = ''
    #                         row['cwe'] = 'Benign'
    #                         row['isVul'] = 0

    #                     df = pd.concat(
    #                         [df, pd.DataFrame([row])], ignore_index=True)
    #                 else:
    #                     print(f'Invalid line: {line} at code {vul_statement}')
    #     return df

    def extract_functions(self, source_file, lines, cwes, context, tool=['cppcheck']):
        """split the given file into a list of function blocks 
        and return their metrics into a dataframe."""
        df = pd.DataFrame()
        all_funs = self.srcML.src2src_functions(src=source_file)
        nall_funs = len(all_funs)

        if nall_funs > 0:
            # print(f'Len of funs: {len(all_funs)}')
            # most expensive steps below

            # for big file, we take only 500 functions,
            # otherwise it may introduce quadradic complexity

            flaw_records = list(
                dict(enumerate(zip(lines, cwes, context, tool))).values())
            # print(f'Flaw_records_size: {len(flaw_records)}')
            count = 0

            if nall_funs > 200 and len(flaw_records) > 1000:
                random.seed(20)
                all_funs = random.sample(all_funs, 10)
                print(f'Len of funs: {len(all_funs)}')

            for fun, record in itertools.product(all_funs, flaw_records):
                count = count+1
                # print(f'Scanning [{count} of {len(all_funs)}]...')

                line, cwe, vul_statement, _ = record

                row = {
                    'file': source_file,
                    'code': fun
                }
                if self.is_fun_vul(fun, vul_statement):
                    row['context'] = vul_statement
                    row['cwe'] = cwe
                    row['isVul'] = 1
                else:
                    row['context'] = ''
                    row['cwe'] = 'Benign'
                    row['isVul'] = 0

                df = pd.concat(
                    [df, pd.DataFrame([row])], ignore_index=True)
            # print('Done!')
        return df

    def polulate_function_table(self, file, df_flaw):
        """populate the function table"""
        df_fun = pd.DataFrame()
        try:
            if len(df_flaw) > 0:
                df_fun = self.extract_functions(
                    source_file=file,
                    lines=list(df_flaw.line),
                    cwes=list(df_flaw.cwe),
                    context=list(df_flaw.context),
                    tool=list(df_flaw.tool),
                )
        except Exception as err:
            print(f"Error while populating function table: {err} at {file}")
        return df_fun

    # def extract_each_function(self, liz_file, fp, fun_name, tuples):
    #     """get the context of the function
    #     """
    #     vul_statements = []
    #     cwe = []

    #     # get the function block
    #     fun = liz_file.fun_name.__dict__
    #     df_fun = pd.DataFrame.from_dict(fun)

    #     start = int(fun["start_line"])
    #     end = int(fun["end_line"])
    #     # moves the header to the initial point of the file
    #     fp.seek(0)

    #     fun_block = [
    #         line for line in itertools.islice(fp, start, end)]
    #     fp.seek(0)

    #     # check if any of the lines of the file belong to any functions
    #     for index, (l, c, cnt, t) in enumerate(tuples):

    #         # checks vulnerability condition
    #         if (isinstance(l, int)) and (start <= l < end):
    #             vline = ''
    #             if t.lower() == "cppcheck" or t.lower() == "rats":
    #                 vline = fp.readlines()[l]
    #                 fp.seek(0)
    #             if vline != '' and cnt != cnt:
    #                 cnt = vline

    #             vul_statements.append((cnt, c))
    #             cwe.append(c)

    #     if len(cwe) == 0:
    #         cwe.append('Benign')

    #     # rename filename to file to make it consistent with statement
    #     df_fun = df_fun.rename(
    #         columns={"filename": "file"})

    #     df_fun['code'] = fun["long_name"] + "".join(fun_block)
    #     df_fun["fun_name"] = fun["name"]
    #     df_fun["content"] = (
    #         str(vul_statements) if len(
    #             vul_statements) > 0 else "")
    #     df_fun["isVul"] = 1 if cwe else 0
    #     df_fun["cwe"] = self.extract_cwe(cwe)
    #     df_fun["project"] = self.project
    #     return df_fun

    # def extract_functions_using_lizard(self, source_file, lines, cwes, context, tool=['cppcheck']):
    #     """split the given file into a list of function blocks and return their metrics into a dataframe.
    #     <guru> I think there is a problem in lizard detecting the correct full_parameters
    #     either we have to concatenate two lines of full_parameters or ignore it
    #     and take it from long_name if needed.
    #     drop['full_parameters', 'fan_in', 'fan_out', 'general_fan_out'] because
    #     the lizard has not properly implemented these parameters yet.
    #     check if the vulnerability content/statement appears in the function block or not.
    #     type of the vul line should be int and then lies in the function block.
    #     # """
    #     df_file = pd.DataFrame()
    #     tuples = zip(lines, cwes, context, tool)

    #     # TODO: review this code now
    #     with open(source_file, "r", encoding='utf-8', errors="surrogateescape") as fp:

    #         source_code = fp.read()
    #         liz_file = lizard.analyze_file.analyze_source_code(
    #             source_file, source_code)

    #         # check liz_file does have fun_name attribute
    #         if hasattr(liz_file, 'fun_name'):

    #             for i, fun_name in enumerate(liz_file.function_list):
    #                 df_fun = self.extract_each_function(
    #                     liz_file, fp, fun_name, tuples)
    #                 df_file = pd.concat([df_file, df_fun])

    #     if len(df_file) > 0:
    #         # drop duplicates and keep a single row
    #         df_file = df_file.drop_duplicates(
    #             subset=['file', 'long_name', 'start_line', 'end_line', 'cwe'],
    #             keep='last'
    #         ).reset_index(drop=True)

    #         if set(self.cols_filter).issubset(set(list(df_file.columns))):
    #             df_file = df_file.drop(self.cols_filter, axis=1)
    #     else:
    #         print(f'No function found in the file: {source_file}')
    #     return df_file


if __name__ == "__main__":
    # funcol = FunsCollector()
    # funcol.polulate_function_table()
    pass
