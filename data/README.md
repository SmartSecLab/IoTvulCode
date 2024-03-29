# Dataset Description

## IoTvulCode dataset

The `IoTvulCode` dataset is a collection of vulnerable codes from various Internet of Things (IoT) platforms. In the current version of the extracted dataset, there are 1,014,548 statements (948,996 benign and 65,052 vulnerable samples) and 548,089 functions (481,390 benign and 66,699 vulnerable samples). We have collected the vulnerable data from the following IoT platforms.

### IoT Specific Projects

| Project    | version   | URL                                   |
| ---------- | --------- | ------------------------------------- |
| linux-rpi  | 6.1.y     | www.raspberrypi.com/software/         |
| ARMmbed    | 6.17.0    | https://os.mbed.com/mbed-os/          |
| FreeRTOS   | 202212.01 | www.freertos.org/a00104.html          |
| RIOT       | 2023.07   | https://github.com/RIOT-OS/RIOT       |
| contiki    | 2.4       | https://github.com/contiki-os/contiki |
| gnucobol   | 3.2       | https://gnucobol.sourceforge.io/      |
| mbed-os    | 6.17.0    | https://github.com/ARMmbed/mbed-os    |
| miropython | 1.12.0    | https://micropython.org/              |
| mosquito   | 2.0.18    | https://github.com/eclipse/mosquitto  |
| openwrt    | 23.05.2   | https://github.com/openwrt/openwrt    |

Among all extracted projects, `linux-rpi` has the most recorded entries with 816,672 total statements and 456,380 functions, which is followed by `ARMmbed` with 43,782 statements and 26,095 functions. Of course, the severity of the project can be seen in the size of the vulnerability and weakness samples present in the project. However, `linux-rpi` being the biggest project in size in the list can tend to hold a higher number of vulnerable samples. The SQLite database file has three tables, namely `project` for project-level information, `statement` for statement-level information, and
`function` for function-level information.

## Convert SQLite to CSV files

To obtain the necessary .csv files for analysis, run the provided script on the original `IoTvulCode.db` file, accessible on [zenodo](https://zenodo.org/uploads/10203899) (with DOI:10.5281/zenodo.10203899).
Utilize the script `db2csv.py` for the conversion of the SQLite3 database into CSV files. The script can be executed with the following command:

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

For comparison of dataset and ML models, we have used anothor publicly available dataset which can be retrieved at [GitHub](https://github.com/idetect2022/iDetect).
Use `notebooks/preprocess_iDetect.ipynb` to perform further preprocessing or refining the `iDetect` dataset.
