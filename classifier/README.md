# ML for IoT vulnerability detection program

This project is a Machine Learning (ML) based solution to detect potential security risks in Internet of Things (IoT) software.

## Run the machine learning pipeline:

Once required packages were installed, run the command to extract the vulnerability database from the given input projects as listed in `config/classifier.yaml`:

```
python3 -m classifier.classifier
```

Command to run the classifier with overriding model, multiclass and granularity-level parameters.

```
python3 -m classifier.classifier --model LSTM --type multiclass --granular function
```

## Additional script:

Script can be executed before `classifier.py`, to make `<data>.csv` file available for the training, testing and the evaluation of the machine learning and deep learning models. The script get statement and function tables of the `IoTvulCode` database and save as `data/IoTvulCode/IoTvulCode-statement.csv` and `data/IoTvulCode/IoTvulCode-function.csv`.

```
python3 -m classifier.db2csvs -d data/IoTvulCode.db -o data/IoTvulCode/
```

## Configurations:

The `config/classifier.yaml` lists the configurations for the extraction the vulnerability data from the given `projects:` list, for example:

#### Note: Run all the commands from the parent directory of the repository `IoTvulCode`.
