# -*- coding: utf-8 -*-
"""
Grepping flaws from the given project list.
Merging output generated by FlawFinder and CppCheck
Grepping functions from the vulnerability context of
file, function and statement-level information
"""

import itertools
import os
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
from tabulate import tabulate

from src.analyzers import Analyzers
from src.collect_funs import FunsCollector
# User defined modules
from src.sqlite import Database
from src.utility import Utility


class Extractor:
    def __init__(self):
        self.util = Utility()
        self.funcol = FunsCollector()
        self.config_file = "ext_projects.yaml"
        self.config = self.util.load_config(self.config_file)

        self.sect = Analyzers(self.config)
        self.pl_list = self.sect.pl_list

        self.cols_filter = [
            "full_parameters",
            "fan_in",
            "fan_out",
            "general_fan_out",
            "top_nesting_level",
            "defaultlevel",
        ]
        self.start_time = time.time()
        db_file = self.config['save']['database']
        self.db = Database()
        self.db.db_exists(db_file, self.config['save']['override'])

    def save_file_data(self, df_flaw, df_fun):
        """
        Generate flaw file and add it to the database
        """
        if len(df_flaw) > 0:
            df_flaw = df_flaw.astype(str)
            # Add benign samples to the statement-level data
            if len(df_fun) > 0:
                df_flaw_benign = self.util.gen_benign(self.config, df_fun)
                df_flaw = pd.concat([df_flaw, df_flaw_benign])

            df_flaw['project'] = self.project
            df_flaw.to_sql('statement', con=self.db.conn,
                           if_exists='append', index=False)

        if len(df_fun) > 0:
            df_fun = df_fun.astype(str)
            df_fun['project'] = self.project
            df_fun.to_sql('function', con=self.db.conn,
                          if_exists='append', index=False)

    def read_file_content(self, file, zip_obj):
        """read file content from zip_obj or file"""
        if zip_obj:
            with zip_obj.open(file) as fp:
                file_content = fp.read()
        else:
            with open(file) as fp:
                try:
                    # use encoding arg otherwise, FlawFinder shows
                    # an encoding error
                    file_content = fp.read().encode("utf-8")
                except UnicodeDecodeError as err:
                    file_content = fp.read().encode("latin-1")
                    print(f"UnicodeDecodeError: {err} for file: {file}")
        return file_content

    def compose_file_flaws(self, file, zip_obj=None):
        """convert zipped file stream - tempfile to pandas dataframe."""
        file_content = "".encode("utf-8")
        df_flaw = pd.DataFrame()

        # read file content from zip_obj or file
        file_content = self.read_file_content(file, zip_obj)

        fp = tempfile.NamedTemporaryFile(suffix="_Flawfinder", prefix="File_")
        try:
            fp.write(file_content)
            # merge the results generated by all the tools
            df_flaw = self.sect.merge_tools_result(fname=file)
        except OSError as err:
            print(f"Could not open/read file: {err} at", file)
        finally:
            fp.close()
        return df_flaw

    def run_fetching_files(self, files, status, zipobj):
        """run fetching files """
        if len(files) <= 0:
            print(f"None of the file in the specified project \
                    \nin given PL list types to extract: {self.sect.pl_list}")
        else:
            change_stat = True if status == 'Not Started' else False

            for index, file in enumerate(files):
                # print(f"\n\n" + "="*50)
                print(f"Scanning: {file}")
                # print("="*50)
                df_flaw = self.compose_file_flaws(file, zipobj)
                df_fun = self.funcol.polulate_function_table(file, df_flaw)

                self.save_file_data(df_flaw, df_fun)

                # verbose on every 100 files
                if index % 100 == 0:  # [0 % 100 = True] for first index
                    print(f"\n#Files: {index + 1} file(s) completed!")
                    self.util.show_time_elapsed(start_time=self.start_time)
                    self.db.check_progress(project=self.project)

                    if change_stat:
                        self.db.change_status(self.project, "In Progress")
                        change_stat = False

    def find_remaining_files(self, project_files):
        """incremental fetching of the remaining files."""
        project_files = set(project_files)

        if self.db.table_exists('statement'):
            extracted_files = set(
                list(pd.read_sql('SELECT file from statement', self.db.conn)['file']))
            remaining_files = list(project_files - extracted_files)

            print(f'#files already extracted: {len(extracted_files)}')
            print(f'#files to extract: {len(remaining_files)}\n')
        else:
            remaining_files = list(project_files)
        return remaining_files

    def project2db(self, project):
        """concatenate all the output dataframes of all files"""
        zipobj = None
        if os.path.isdir(project):
            # if the url is a directory, get all the files in it
            all_files = [str(f) for f in Path(project).rglob("*.*")]
            # get the files that match the file extensions
            program_files = [
                x for x in all_files if self.sect.guess_pl(x)
                in self.sect.pl_list]
        else:
            # if the url is a zip file, get all the files in it
            zipobj = self.util.retrieve_zip(project)
            if zipobj != None:
                files = zipobj.namelist()
                # get the files that match the file extensions
                program_files = [
                    x for x in all_files
                    if self.sect.guess_pl(x, zipobj) in self.sect.pl_list]
            else:
                print("Invalid URL!")
                return False

        if len(program_files) <= 0:
            print("No program files present! Please check the url!")
            return False
        else:
            print(f'#files in the project: {len(program_files)}')
            remaining_files = self.find_remaining_files(
                project_files=program_files)

            self.run_fetching_files(files=remaining_files,
                                    status=self.db.get_status(project),
                                    zipobj=zipobj
                                    )
            return True

    def refine_data(self, table):
        """refine the data, and filter out duplicates 
        after the extraction of all the raw data"""
        df = pd.DataFrame()
        if self.db.table_exists(table):
            df = pd.read_sql(f"SELECT * FROM {table}", con=self.db.conn)
            print("\n\n" + "="*15 +
                  f" Refining Data: [{table}] " + "="*15)
            print('=' * 55)
            print('Before filtering:')
            print("-"*20)
            self.util.show_info_pd(df, table)

            # filter the results
            df = self.util.filter_results(df)

            print('\nAfter filtering:')
            print("-"*20)
            self.util.show_info_pd(df, table)
            df.to_sql(name=table,
                      con=self.db.conn,
                      if_exists='replace',
                      index=False)
            print(f"[{table}] data got updated!")
            print("="*55 + "\n")
        else:
            print(f"Extraction is not possible on empty data: [{table}]!")
        return df

    def iterate_projects(self, prj_dir_urls):
        """iterate on every project"""
        df_prj = self.db.save_project_meta()
        prj_dir_urls = list(df_prj['project'])
        status_all_complete = False

        for prj in prj_dir_urls:
            self.project = prj
            stat = self.db.get_status(prj)

            if stat == 'Complete':
                # print(f"The project had been already extracted!")
                status_all_complete = True

            elif stat == 'Not Started' or stat == 'In Progress':
                # extract the project directory/URL
                success = self.project2db(prj)

                if success:
                    # Change the project status to complete
                    self.db.change_status(prj, 'Complete')
                else:
                    print("Non-zipped project!")
            else:
                print("Project is not in the list!")
                print('\n\n')

        print("=" * 50)
        print("All the given projects were extracted!")
        print("=" * 50)
        self.db.check_progress(project='all')
        self.db.display_table(table='project')
        # self.db.conn.commit()
        # self.db.cursor.close()
        return status_all_complete

    def run_extractor(self):
        """Add new projects to the database"""
        status_of_all = ext.iterate_projects(ext.config["projects"])

        # if status_of_all:
        #     print("Extraction were already complete for all projects!")
        # else:

        # Refine the data
        start_time = time.time()
        ext.refine_data('statement')
        ext.refine_data('function')
        time_elapsed = time.time() - start_time
        print("Time elapsed for filtering: " +
              f"{format_timespan(time_elapsed)}")

        # total time elapsed
        time_elapsed = time.time() - ext.start_time
        print("\n" + "="*50)
        print(f"Total time elapsed: {format_timespan(time_elapsed)}")
        print('The database is saved at:' +
              f'{self.config["save"]["database"]}')
        print("="*50)

        # final operations to the database
        self.db.conn.commit()
        self.db.cursor.close()


if __name__ == "__main__":
    ext = Extractor()
    ext.run_extractor()
