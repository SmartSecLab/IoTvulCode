import random
import sqlite3
from sqlite3 import connect

import pandas as pd
import yaml


class Database:
    def __init__(self, db_file):
        self.conn = connect(db_file)
        self.cursor = self.conn.cursor()

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.conn.commit()
        self.cursor.close()

    def table_exists(self, table_name):
        """ checks if table exists in database """
        x = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' \
                AND name='" + table_name + "'", con=self.conn)
        if len(x) <= 0:
            print("function table not found in already scanned record!\
                     \nScanning from the beginning...")
            return False
        else:
            print(f'\nContinue ongoing extraction process...')
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

    def query_project_table(self):
        """ creates the project table """
        df = pd.DataFrame()
        try:
            config = yaml.safe_load(open("ext_projects.yaml"))
            projects = config["projects"]

            if self.table_exists('project') == False:
                df = pd.DataFrame(data={
                    'project': projects,
                    'status': 'Not Started'
                })
                pid = [random.getrandbits(16) for pid in df['project']]
                df.insert(0, 'project_id', pid)
                df.to_sql('project', con=self.conn,
                          if_exists='replace', index=False)
                print("Successfully created the project table!")
            else:
                db_projects = list(pd.read_sql(
                    'SELECT project from project', self.conn)['project'])
                print("Project table already exists!")

                for prj in projects:
                    if prj not in db_projects:
                        self.insert_project(prj)

            df = pd.read_sql('SELECT * from project', self.conn)
            print("-"*30)
            print(f"\nProjects Status: \n{df}")
            print(f"-"*30)
        except sqlite3.Error as error:
            print("Failed to create sqlite table, ", error)
        return df

    def show_shape(self, table_name, project):
        """ display the table shape """
        if project == 'all':
            nrows = pd.read_sql(
                "SELECT COUNT(*) as len from '" + table_name + "'",
                con=self.conn)
        else:
            nrows = pd.read_sql("SELECT COUNT(*) as len from '" +
                                table_name + "' WHERE project='"
                                + project + "'",
                                con=self.conn)

        ncols = pd.read_sql("SELECT COUNT(*) as len from pragma_table_info('" + table_name + "')",
                            con=self.conn)
        print(
            f"Shape of the table: {table_name}(r,c) -> ({nrows['len'][0]}, {ncols['len'][0]})")

    def change_status(self, project, status):
        """ changes the status of the project """
        try:
            query = "UPDATE project SET status ='" + \
                status + "' WHERE project='" + project + "'"
            self.cursor.execute(query)
            print(f"\nProject: {project} status changed to [{status}]!")
            print('-'*50)
            print(f"Project: {project} [{status}]\n")
            print('-'*50)

        except sqlite3.Error as error:
            print("Failed to update sqlite table, ", error)

    def get_status(self, project):
        """ returns the status of the project """
        status = 'Unknown'
        try:
            query = "SELECT status FROM project where project='" + project + "'"
            # print(f"Query: {query}")
            self.cursor.execute(query)
            result = self.cursor.fetchone()

            if result is None:
                print(f"Project {project} not found in the database!\n")
            else:
                status = result[0]
                print('-'*50)
                print(f"Project: {project} [{status}]")
                print('-'*50)
        except sqlite3.Error as error:
            print(f"Failed to update SQLite table, {error}")
        return status

    def show_table_info(self, table):
        """ display the table info """
        try:
            query = "SELECT * FROM " + table
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            print(f"Table: {table} \n{result}")
        except sqlite3.Error as error:
            print(f"Failed to update SQLite table, {error}")

    def show_cwe_benign(self, table):
        """ display the count of benign and vulnerable samples 
        in the table """
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


if __name__ == "__main__":
    create_project_table()
    show_shape('project')
    get_status('contiki-master')
    table_exists('function')
