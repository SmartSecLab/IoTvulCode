"""
Copyright (C) 2023 SmartSecLab, Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT
@Programmer: Guru Bhandari
"""
import pandas as pd
from sqlite3 import connect
from pathlib import Path
from argparse import ArgumentParser


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


def save_csv_from_TinyVul(df_fun, df_stat, csv_path):
    """ save the data to csv files """
    # Save multiclass CSVs
    print('Saving datasets...')
    df_fun = correct_multi_label(df_fun)
    df_stat = correct_multi_label(df_stat)
    df_fun.to_csv(f"{csv_path}-function.csv", index=False)
    df_stat.to_csv(f"{csv_path}-statement.csv", index=False)
    print(f'Saved at: data/{csv_path}-<name>.csv')


def parse_args():
    """ parse command line arguments 
    Usage: python db2csv.py -d <path/to/database.db> -o <path/to/output/dir>
    """
    parser = ArgumentParser(
        description='Convert IoTvulCode DB to CSVs')
    # give database file as an argument
    parser.add_argument('--database', '-d', type=str,
                        default='data/IoTvulCode.db',
                        help='IoTvulCode database file')
    # give output file as another argument
    parser.add_argument('--output', '-o', type=str, default="data/IoTvulCode/",
                        help="Output directory to save the CSVs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    db_name = Path(args.database).name.replace('.db', '')
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"Converting : {args.database}")
    csv_path = args.output + db_name

    df_fun, df_stat, df_prj = load_db2dataframes(args.database)
    save_csv_from_TinyVul(df_fun, df_stat, csv_path=csv_path)

    # Save project details into a CSV file (optional)
    df_prj.to_csv(f"{csv_path}-project.csv", index=False)
    print(f'Project data saved at: data/{csv_path}-project.csv')
