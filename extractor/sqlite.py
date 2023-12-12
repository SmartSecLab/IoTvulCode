"""
Copyright (C) 2023 SmartSecLab, Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT
@Programmer: Guru Bhandari
"""
import random
import sqlite3
from sqlite3 import connect
from pathlib import Path

import pandas as pd
import yaml
from tabulate import tabulate
from extractor.utility import Utility


class Database:
    def __init__(self):
        self.util = Utility()

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.conn.commit()
        self.cursor.close()

    def db_exists(self, db_file, override):
        """ check for existing file and prompt user to overwrite or not..."""
        print('='*40)
        if Path(db_file).exists():
            print(f"The database already exists: [{db_file}]")
            if override:
                print("Overridden of the existing database!")
            else:
                print("Provide another database name!")
                print("Or set override option to True in the config file!")
                exit(1)
        print('='*40)

        self.conn = connect(db_file)
        self.cursor = self.conn.cursor()

    def table_exists(self, table):
        """ checks if table exists in database """
        df = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' \
                AND name='" + table + "'", con=self.conn)
        if len(df) <= 0:
            # print(f"Table [{table}] not found in the record!\n")
            print('Scanning...')
            return False
        else:
            # print(f"Table [{table}] found in the record!\n")
            return True

    def insert_project(self, project):
        """ inserts a new project into the project table """
        try:
            pid = random.getrandbits(16)
            query = "INSERT INTO project (project_id, project, status) \
                VALUES ('" + str(pid) + "', '" + project + "', 'Not Started')"
            self.cursor.execute(query)
            print(f"Project {project} entry to the database!")

        except sqlite3.Error as error:
            print("Failed to update sqlite table, ", error)

    def display_table(self, table='project'):
        """ display the status of the projects """
        df = pd.read_sql("SELECT * from " + table, self.conn)
        print("-"*50)
        print("\n\n" + "="*20 + "Projects Status:" + "="*20)
        print(tabulate(df, headers='keys', tablefmt='fancy_outline'))
        print('\n')

    def add_todo_projects_meta(self, projects):
        """check whether there's any new project added and add them into DB"""
        db_projects = list(pd.read_sql(
            'SELECT project from project', self.conn)['project'])

        todo_projects = list(set(projects) - set(db_projects))

        if len(todo_projects) > 0:
            for prj in todo_projects:
                self.insert_project(prj)
        else:
            print("All projects are already in the database!")

    def save_project_meta(self):
        """ creates a table for the projects to track their status """
        df = pd.DataFrame()
        try:
            config = yaml.safe_load(open("config/extractor.yaml"))
            projects = config["projects"]

            if self.table_exists('project') is False:
                df = pd.DataFrame(data={
                    'project': projects,
                    'status': 'Not Started'
                })
                pid = [random.getrandbits(16) for pid in df['project']]
                df.insert(0, 'project_id', pid)
                df.to_sql('project', con=self.conn,
                          if_exists='replace', index=False)
                print("Project metadata is saved!")
            else:
                print("Project meta table already exists!")
                self.add_todo_projects_meta(projects)

                # get projects from the project table
                df = pd.read_sql('SELECT * from project', self.conn)

            self.display_table(table='project')

        except sqlite3.Error as error:
            print("Failed to create sqlite table, ", error)
        return df

    def show_shape(self, table, project):
        """ display the table shape """
        if self.table_exists(table):
            if project == 'all':
                nrows = pd.read_sql(
                    "SELECT COUNT(*) as len from '" + table + "'",
                    con=self.conn)
            else:
                nrows = pd.read_sql("SELECT COUNT(*) as len from '" +
                                    table + "' WHERE project='"
                                    + project + "'",
                                    con=self.conn
                                    )

            ncols = pd.read_sql("SELECT COUNT(*) as len from pragma_table_info('" + table + "')",
                                con=self.conn)
            print(
                f"Shape of the table: {table}(r,c) -> ({nrows['len'][0]}, {ncols['len'][0]})")
        else:
            # print(f"Table [{table}] not found in the database!")
            pass

    def change_status(self, project, status):
        """ changes the status of the project """
        try:
            query = "UPDATE project SET status ='" + \
                status + "' WHERE project='" + project + "'"
            self.cursor.execute(query)
            print('-'*50)
            print(f"Project status: {project} [{status}]\n")
            print('-'*50)

        except sqlite3.Error as error:
            print("Failed to update sqlite table, ", error)

    def get_status(self, project):
        """ returns the status of the project """
        status = 'Unknown'
        if self.table_exists('project'):
            try:
                query = "SELECT status FROM project where project='" + project + "'"
                self.cursor.execute(query)
                # self.conn.commit()
                result = self.cursor.fetchone()

                if result is None:
                    print(f"Project [{project}] not found in the database!\n")
                else:
                    status = result[0]
            except sqlite3.Error as error:
                print(f"Failed to update SQLite table, {error}")
        return status

    def show_table_info(self, table):
        """ display the table info """
        if self.table_exists(table):
            try:
                query = "SELECT * FROM " + table
                self.cursor.execute(query)
                result = self.cursor.fetchall()
                print(f"Table: [{table}] \n{result}")
            except sqlite3.Error as error:
                print(f"Failed to update SQLite table, {error}")

    def show_cwe_benign(self, table):
        """ display the count of benign and vulnerable samples 
        in the table """
        if self.table_exists(table):
            try:
                benigns = pd.read_sql("SELECT COUNT(*) as len from '" +
                                      table + "' WHERE cwe='Benign'",
                                      con=self.conn)['len'][0]
                non_benigns = pd.read_sql("SELECT COUNT(*) as len from '" +
                                          table + "' WHERE cwe!='Benign'",
                                          con=self.conn)['len'][0]
                print(f"[{table}] #Benign: {benigns}, #Vulnerable: {non_benigns}\n")
                return benigns, non_benigns

            except sqlite3.Error as error:
                print(f"Failed to update SQLite table, {error}")

    def check_progress(self, project):
        """ check the progress of the project """
        print("\n" + "-" * 10 + f" Summary of: [{project}] " + "-" * 10)
        if self.table_exists('statement'):
            self.show_shape(table='statement', project=project)
            self.show_cwe_benign(table='statement')
            print("-" * 50 + "\n")
        else:
            print("Scanning...\n")

        if self.table_exists('function'):
            self.show_shape(table='function', project=project)
            self.show_cwe_benign(table='function')
            print("-" * 50 + "\n")


if __name__ == "__main__":
    db = Database('data/IoT.db')
    db.create_project_table()
    db.show_shape('project')
    db.get_status('contiki-master')
    db.table_exists('function')
