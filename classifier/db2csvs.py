import pandas as pd
from sqlite3 import connect
from pathlib import Path
import ruamel
import sys


def load_db2dataframes(database):
    """ Load the database into Pandas DataFrames """
    conn = connect(database)
    df_fun = pd.read_sql(
        sql="SELECT code, cwe as label FROM function", con=conn)
    df_stat = pd.read_sql(
        sql="SELECT context as code, cwe as label FROM statement", con=conn)
    df_prj = pd.read_sql(sql="SELECT * FROM project", con=conn)
    return df_fun, df_stat, df_prj


def correct_multi_label(df):
    """ retrieve binary data from multiclass df """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    df = df.drop_duplicates()
    return df


def convert_multi2binary(df):
    """ retrieve binary data from multiclass df """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    df['label'] = df.apply(lambda row: 0 if row['label']
                           == 'Benign' else 1, axis=1)
    df['label'] = df['label'].astype(int)
    df = df.drop_duplicates()
    return df


def save_csvs_from_TinyVul(df_fun, df_stat, db_name):
    """ save the data to csv files """
    # Save multiclass CSVs
    print('Saving datasets...')
    df_fun = correct_multi_label(df_fun)
    df_stat = correct_multi_label(df_stat)
    df_fun.to_csv(f"data/{db_name}-function-multiclass.csv", index=False)
    df_stat.to_csv(f"data/{db_name}-statement-multiclass.csv", index=False)

    # Save binary CSVs
    df_fun = convert_multi2binary(df_fun)
    df_stat = convert_multi2binary(df_stat)
    df_fun.to_csv(f"data/{db_name}-function-binary.csv", index=False)
    df_stat.to_csv(f"data/{db_name}-statement-binary.csv", index=False)
    print(f'Saved at: data/{db_name}-<name>.csv')


database = '/Users/guru/research/TinyVul-v2.db'
df_fun, df_stat, df_prj = load_db2dataframes(database)

db_name = Path(database).name.replace('.db', '')
save_csvs_from_TinyVul(df_fun, df_stat, db_name=db_name)

# Save project details into a CSV file (optional)
df_prj.to_csv(f"data/{db_name}-project.csv", index=False)


# # Change drop_dup variable in configuration file:
# def change_config_drop_dup():
#         """ change the config file to drop duplicates """
#         with open("../ext_projects_ruamel.yaml") as f:
#                 # config = yaml.load(f, Loader=yaml.RoundTripLoader)
#                 doc = ruamel.yaml.load(f.read(), Loader=ruamel.yaml.RoundTripLoader)
#                 doc['save']['drop_dup'] = '10'
#                 print(ruamel.yaml.dump(doc, Dumper=ruamel.yaml.RoundTripDumper))
#                 ruamel.yaml.preserve_quotes = True
#                 config['save']['drop_dup'] = '10'
#                 print(config)
#         ## Or
#         # # yaml = yaml.YAML()
#         # # config = yaml.safe_load(open("../ext_projects.yaml"))
#         # with open("../ext_projects.yaml") as f:
#         #     yaml = ruamel.yaml.YAML()
#         #     yaml.indent(mapping=4, sequence=4, offset=1)
#         #     yaml.preserve_quotes = True
#         #     params = yaml.load(f.read())
#         #     params['projects'].yaml_add_eol_comment('some comment', key='new_key', column=40)
#         #     config['save']['drop_dup'] = '10'
#         #     # params['ParentTest']['test'].yaml_add_eol_comment('some comment', key='new_key', column=40) # column is optional
#         #     yaml.dump(params, sys.stdout)
