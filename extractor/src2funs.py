# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 SmartSecLab, Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT
@Programmer: Guru Bhandari

File description:
grepping function from the given file of the code.
"""
import itertools
import random
import subprocess

import pandas as pd
from lxml import etree


class FunsCollector:
    """Extract functions from source code"""

    def src2xml(self, src):
        """generate srcML tree from the given source file or directory"""
        # srcml --xpath="//src:function" '../data/projects/contiki-2.4/apps/ftp/ftpc.c'
        # | srcml --xpath="string(//src:function)"
        src2xml_cmd = ["srcml", "--xpath=//src:function", src]
        # xml2code_cmd = ['srcml', '--xpath=string(//src:function)']
        xml = None
        try:
            ps = subprocess.Popen(src2xml_cmd,
                                  stdout=subprocess.PIPE,
                                  text=True,
                                  )
            xml = ps.stdout.read()
        except subprocess.TimeoutExpired as err:
            print(err)
        # print('xml generated!')
        return xml

    def xpath_on_tree(self, the_tree, xpath_query):
        """Run an xpath query on a srcML parsetree"""
        try:
            return the_tree.xpath(xpath_query, namespaces={
                'src': 'http://www.srcML.org/srcML/src'
            })
        except etree.XPathEvalError as err:
            print(err)
            return None

    def function_tree2source(self, fun_tree):
        """convert a function tree to source code"""
        head = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" url="prj">
        <unit revision="1.0.0" filename="prj" item="1">
        """
        # <unit revision="1.0.0" language="C" filename="prj" item="1">
        tail = """</unit>
        </unit>"""

        body = etree.tostring(fun_tree, pretty_print=True, encoding='unicode')
        # the extracted tree is bare function excluding head and tail
        tree = head + body + tail
        tree = etree.fromstring(tree.encode('utf-8'))
        return self.xpath_on_tree(tree, 'string(//src:function)')

    def extract_functions_from_srcML(self, tree):
        """get all function bodies"""
        fun_trees = self.xpath_on_tree(tree, '//src:function')
        functions = [self.function_tree2source(
            fun_tree) for fun_tree in fun_trees]
        return functions

    def write_functions_file(self, file, functions):
        """ write all functions to a file"""
        with open(file, 'w') as fp:
            fp.write('\n\n'.join(functions))

    def src2src_functions(self, src):
        """retrieve functions from the src:file/dir of source code"""
        tree = self.src2xml(src)
        tree = etree.fromstring(tree.encode('utf-8'))
        return self.extract_functions_from_srcML(tree)

    def fix_cwe_labeling(self, cwe) -> str:
        """ Extract CWE type information,
        In case of Rats tool's 'CWE-unknown' list, 
        make it just a single item.
        """
        cwe = list(set(cwe)) if isinstance(cwe, list) else [cwe]

        if len(cwe) > 1:
            if 'CWE-unknown' in cwe:
                # remove 'CWE-unknown' if already labeled as a known vul.
                cwe.remove('CWE-unknown')
        return str(cwe)

    def label_function(self, fun, vul_statement):
        """check if the vulnerability content/statement appears 
        in the function block or not."""
        vul_bool = False
        if len(vul_statement) > 1:
            if ''.join(vul_statement.split()) in ''.join(fun.split()):
                vul_bool = True
        return vul_bool

    def extract_functions(self, source_file, lines, cwes, context, tool=['cppcheck']):
        """split the given file into a list of function blocks 
        and return their metrics into a dataframe."""
        df = pd.DataFrame()
        all_funs = self.src2src_functions(src=source_file)
        nall_funs = len(all_funs)

        if nall_funs > 0:
            # most expensive steps below
            # for big file, we take only 500 functions,
            # otherwise it may introduce quadradic complexity

            flaw_records = list(
                dict(enumerate(zip(lines, cwes, context, tool))).values())

            if nall_funs > 200 and len(flaw_records) > 1000:
                random.seed(20)
                all_funs = random.sample(all_funs, 10)

            for fun, record in itertools.product(all_funs, flaw_records):

                line, cwe, vul_statement, _ = record
                row = {
                    'file': source_file,
                    'code': fun
                }
                if self.label_function(fun, vul_statement):
                    row['context'] = vul_statement
                    row['cwe'] = self.fix_cwe_labeling(cwe)
                else:
                    row['context'] = ''
                    row['cwe'] = 'Benign'

                df = pd.concat(
                    [df, pd.DataFrame([row])], ignore_index=True)

            if len(df) > 0:
                df = df.drop_duplicates().reset_index(drop=True)
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


if __name__ == "__main__":
    src = 'data/projects/contiki-2.4/apps/ftp/ftpc.c'
    funcollector = FunsCollector()
    all_functions = funcollector.src2src_functions(src)

    context = "c->dataconn.conntype = CONNTYPE_FILE;"
    vul_functions = [line for line in all_functions if context in line]

    df_file = pd.DataFrame()
    for x in vul_functions:
        print(x)
