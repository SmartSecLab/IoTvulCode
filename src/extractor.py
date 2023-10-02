# -*- coding: utf-8 -*-
"""
Grepping flaws from the given project list.
Merging output generated by FlawFinder and CppCheck
Grepping functions from the vulnerability context of
file, function and statement-level information
"""
import os
import signal
import tempfile
import time
from pathlib import Path

import pandas as pd
# import time
from humanfriendly import format_timespan
# from pylibsrcml import srcml
from tabulate import tabulate

# User defined modules
from src.analyzers import Analyzers
from src.src2funs import FunsCollector
from src.sqlite import Database
from src.utility import Utility


def handle_timeout(signum, frame):
    raise TimeoutError


def set_alarm():
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(2)  # 5 seconds


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
        self.refine_on_every = self.config['save']['refine_on_every']

        self.db = Database()
        self.db.db_exists(db_file, self.config['save']['override'])

    def save_file_data(self, df_flaw, df_fun):
        """
        Generate a flaw file and add it to the database
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
                except TimeoutError as err:
                    print(f"TimeoutError: {err} for file: {file}")
                    # skip the file
                    file_content = "".encode("utf-8")
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

    def refine_data(self, table: str) -> pd.DataFrame():
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
            print('Updating...')
            df.to_sql(name=table,
                      con=self.db.conn,
                      if_exists='replace',
                      index=False)
            print(f"[{table}] data got updated!")
            print("="*55 + "\n")
        else:
            print(f"Extraction is not possible on empty data: [{table}]!")
        return df

    def run_fetching_files(self, files: list, status: bool, zipobj):
        """run fetching files """
        files_count = len(files)
        if files_count <= 0:
            print("None of the file in the specified project" +
                  f"\nin given PL list types to extract: {self.sect.pl_list}")
        else:
            # change_stat = True if status == 'Not Started' else False

            for index, file in enumerate(files):
                print(f"Scanning [{index+1} of {files_count}]: {file}...")
                df_flaw = self.compose_file_flaws(file, zipobj)
                df_fun = self.funcol.polulate_function_table(file, df_flaw)

                if len(df_fun) > 0:
                    # drop_duplicates to df with list does not work here, so
                    df_fun.loc[df_fun.astype(
                        str).drop_duplicates().index].reset_index(drop=True)

                if len(df_flaw) > 0:
                    df_flaw.loc[df_flaw.astype(
                        str).drop_duplicates().index].reset_index(drop=True)

                self.save_file_data(df_flaw, df_fun)

                if index % self.refine_on_every == 0:
                    # if change_stat:
                    self.db.change_status(self.project, "In Progress")
                    #     change_stat = False

                if index+1 % self.refine_on_every == 0:
                    print(f"\n#Files: {index + 1} file(s) completed!")
                    self.util.show_time_elapsed(start_time=self.start_time)

                    # apply refining on every 'refine_on_every' files
                    self.refine_data('statement')
                    self.refine_data('function')
                    self.db.conn.execute("VACUUM")  # optimize the storage
                    print('Scanning the remaining files...\n')

                    # total time elapsed
                    time_elapsed = format_timespan(
                        time.time() - ext.start_time)
                    print("\n" + "="*50)
                    print(f"Total time elapsed: {time_elapsed}")
                    print("\n" + "="*50)

    def find_remaining_files(self, files: list) -> list:
        """incremental fetching of the remaining files."""
        remaining_files = set(files)

        if self.db.table_exists('statement'):
            extracted_files = set(
                list(pd.read_sql('SELECT file from statement', self.db.conn)['file']))
            remaining_files = list(remaining_files - extracted_files)

            print(f'#files already extracted: {len(extracted_files)}')
            print(f'#files to extract: {len(remaining_files)}\n')

        return remaining_files

    def project2db(self, project: str) -> bool:
        """concatenate all the output dataframes of all files"""
        print(f'Extracting data from: [{project}]...')
        print('-'*30)

        zipobj = None
        complete_stat = False
        program_files = []

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
            if zipobj is not None:
                all_files = zipobj.namelist()
                # get the files that match the file extensions
                program_files = [
                    x for x in all_files
                    if self.sect.guess_pl(x, zipobj) in self.sect.pl_list]
            else:
                print("Invalid URL!")

        if len(program_files) == 0:
            print("No program files present! Please check the url!")
        else:
            print(f'#files in the project: {len(program_files)}')
            remaining_files = self.find_remaining_files(
                files=program_files)

            self.run_fetching_files(files=remaining_files,
                                    status=self.db.get_status(project),
                                    zipobj=zipobj
                                    )
            complete_stat = True
        # print(f'Status of prj: {complete_stat}')
        return complete_stat

    def iterate_projects(self):
        """iterate on every project"""
        df_prj = self.db.save_project_meta()

        df_prj_todo = df_prj[df_prj.status != 'Complete']
        project_links = df_prj_todo['project'].tolist()
        ext_stat = False

        if len(project_links) > 0:
            print(f'Projects to be extracted: \n{tabulate(df_prj_todo)}')
            # try:
            for project in project_links:
                self.project = project
                # extract the project directory/URL
                success_stat = self.project2db(project)

                if success_stat:
                    # Change the project status to complete
                    self.db.change_status(project, 'Complete')
                    self.db.display_table(table='project')
                else:
                    print("Non-zipped project!")
            ext_stat = True
            # except Exception as ex:
            #     print(f'[iterate_projects] Exception: {ex}')

            if self.db.table_exists('function'):
                self.refine_data('function')

            if self.db.table_exists('statement'):
                # apply refining on every 'refine_on_every' files
                self.refine_data('statement')
                self.db.conn.execute("VACUUM")  # optimize the storage
            print('Scanning the remaining files...\n')
        else:
            ext_stat = True
        return ext_stat

    def run_extractor(self):
        """Add new projects to the database"""
        status_of_all = ext.iterate_projects()

        if status_of_all:
            print("=" * 50)
            self.db.check_progress(project='all')
            print('All projects were extracted!')
        else:
            print('Unable to extract the projects!')

        print("="*50)
        # final operations to the database
        self.db.conn.commit()
        # print('Executing VACUUM...')
        self.db.conn.execute("VACUUM")  # optimize the storage
        self.db.cursor.close()

        # total time elapsed
        time_elapsed = time.time() - ext.start_time
        print("\n" + "="*50)
        print(f"Total time elapsed: {format_timespan(time_elapsed)}")
        print("="*50)

        print('The database is saved at: ' +
              f'{self.config["save"]["database"]}')
        print("="*50)


if __name__ == "__main__":
    ext = Extractor()
    ext.run_extractor()
