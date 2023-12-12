# Dataset Description:

To obtain the necessary .csv files for analysis, run the provided script on the original IoTvulCode.db file, accessible on [zenodo](). Utilize the script db2csv.py for the conversion of the SQLite3 database into CSV files. The script can be executed with the following command:

Syntax:

```
python -m classifier.db2csv -d <path/to/database.db> -o <path/to/output/dir>
```

-d or --database: Specifies the path to the SQLite database file. \
-o or --output: Sets the output directory for saving CSV files.

For example:

```
python -m classifier.db2csv -d data/IoTvulCode.db data/IoTvulCode
```

Once you execute the script, it will bring CSV files including 'data/IoTvulCode/statement.csv' which is used for the experimentation purposes in this study.

To comparison dataset and ML models, we have used anothor publicly available dataset which can be retrieved at [GitHub](https://github.com/idetect2022/iDetect).
Use `notebooks/preprocess_iDetect.ipynb` to perform further preprocessing or refining the `iDetect` dataset.
