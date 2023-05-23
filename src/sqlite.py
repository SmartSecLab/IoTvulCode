import sqlite3
import random
import yaml
from sqlite3 import connect
import pandas as pd


class Database:
    def __init__(self):
        self.conn = connect('IoTcode.db')
        self.cursor = self.conn.cursor()

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.conn.commit()
        self.cursor.close()

    def table_exists(self, table_name):
        """ checks if table exists in database """
        x = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='" + table_name + "'", con=self.conn)
        if len(x) <= 0:
            print("function table not found in already scanned record!\
                     \nScanning from the beginning...")
            return False
        else:
            print(f'\nContinue ongoing extraction process...')
            return True

    def create_project_table(self):
        """ creates the project table """
        df = pd.DataFrame()
        try:
            if self.table_exists('project') == False:
                config = yaml.safe_load(open("ext_projects.yaml"))
                projects = config["projects"]
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
                df = pd.read_sql('SELECT * from project', self.conn)
                print("Project table already exists!")

            print("-"*30)
            print(f"\nProjects: \n{df}")
            print(f"-"*30)
        except sqlite3.Error as error:
            print("Failed to create sqlite table, ", error)
        return df

    def show_shape(self, table_name, project):
        """ display the table shape """
        nrows = pd.read_sql("SELECT COUNT(*) as len from '" +
                            table_name + "' WHERE project='" + project + "'", con=self.conn)
        ncols = pd.read_sql(
            "SELECT COUNT(*) as len from pragma_table_info('" + table_name + "')", con=self.conn)
        print(
            f"Shape of the {table_name} table: ({nrows['len'][0]}, {ncols['len'][0]})")

    def change_status(self, project, status):
        """ changes the status of the project """
        try:
            query = "UPDATE project SET status ='" + \
                status + "' where project='" + project + "'"
            self.cursor.execute(query)

        except sqlite3.Error as error:
            print("Failed to update sqlite table, ", error)

    def get_status(self, project):
        """ returns the status of the project """
        try:
            query = "SELECT status FROM project where project='" + project + "'"
            # print(f"Query: {query}")
            self.cursor.execute(query)
            result = self.cursor.fetchone()

            if result is None:
                print(f"Project {project} not found in the database!")
                return None
            else:
                status = result[0]
                print(f"Project {project} [{status}]")
                return status

        except sqlite3.Error as error:
            print("Failed to update SQLite table, ", error)


if __name__ == "__main__":
    create_project_table()
    show_shape('project')
    get_status('contiki-master')
    table_exists('function')
