import pandas as pd
import subprocess
from lxml import etree
from io import StringIO
import sys
import difflib
import os
import ctypes
import stat
from pylibsrcml import srcml

# source= '../data/projects/contiki-2.4/apps/ftp'
# xml = "myprj.xml"

# # Translate from a source-code file to a srcML file
# srcml.srcml(source, xml)


class Src2Funs:
    """Extract functions from source code"""

    # def __init__(self, src):
    #     self.src = src

    def src2xml(self, src):
        """generate srcML tree from the given source file or directory"""
        # srcml --xpath="//src:function" '../data/projects/contiki-2.4/apps/ftp/ftpc.c' | srcml --xpath="string(//src:function)"
        src2xml_cmd = ["srcml", "--xpath=//src:function", src]
        xml2code_cmd = ['srcml', '--xpath=string(//src:function)']

        # ps = subprocess.Popen(src2xml_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # output = subprocess.Popen(xml2code_cmd, stdin=ps.stdout, stdout=subprocess.PIPE, text=True)
        # output, errors = output.communicate()
        # print(output)
        # print(errors)
        ps = subprocess.Popen(src2xml_cmd, stdout=subprocess.PIPE, text=True)
        return ps.stdout.read()

    def xpath_on_tree(self, the_tree, xpath_query):
        """Run an xpath query on a srcML parsetree"""
        try:
            return the_tree.xpath(xpath_query, namespaces={'src': 'http://www.srcML.org/srcML/src'})
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
        functions = []

        for fun_tree in fun_trees:
            functions.append(self.function_tree2source(fun_tree))

        if len(functions) > 0:
            return functions
        else:
            return [head + tail]

    def write_functions_file(self, file, functions):
        """ write all functions to a file"""
        with open(file, 'w') as f:
            for item in functions:
                f.write("%s\n\n" % item)

    def src2src_functions(self, src):
        """retrieve source functions from the given src:file/dir of source code"""
        try:
            tree = self.src2xml(src)
            tree = etree.fromstring(tree.encode('utf-8'))
            return self.extract_functions_from_srcML(tree)
        except Exception as err:
            print(err)


if __name__ == "__main__":
    src = 'data/projects/contiki-2.4/apps/ftp/ftpc.c'
    src2funs = Src2Funs()
    all_functions = src2funs.src2src_functions(src)
    # print(functions)
    # src2funs.write_functions_file(
    #     'functions.txt', functions)

    context = "c->dataconn.conntype = CONNTYPE_FILE;"
    vul_functions = [line for line in all_functions if context in line]

    df_file = pd.DataFrame()
    for x in vul_functions:
        print(x)
