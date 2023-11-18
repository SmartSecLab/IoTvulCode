[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Vulnerability detection in IoT software analyzing open-source code:

In this study, we have presented an IoT vulnerability data extraction tool and machine learning methods to detect vulnerabilities in the C\C++ source code of IoT operating systems(OS) and applications.
The source code of various IoT OSs and applications was used to create a binary and multi-class labeled dataset including both vulnerable and benign samples.
To make the dataset more generic to the common standard of the C\C++ source code, we have selected the IoT OS, and application if there is at least an entry of the project in the Common Vulnerability and Enumerations (CVE).
The types of vulnerabilities in the presented dataset are linked to the Common Weakness Enumeration (CWE) records.

## Dataset extraction approach for IoT vulnerability analysis:

The proposed method for vulnerability data collection is as follows:

![framework](figure/framework.jpg?raw=true "The proposed framework for vulnerability data collection")

Follow the IoT vulnerability dataset extraction instruction [here](extractor/README.md)

## ML method for IoT vulnerability detection:

The novel vulnerability detection approach in IoT OSs and applications:

![framework](figure/MLframework.jpg?raw=true "The proposed method for vulnerability detection in IoT OSs and applications")

Follow the vulnerability classification instruction [here](classifier/README.md)

# Software Dependencies:

- Python (3.7)
- pip 23.3.1
- FlawFinder 2.0.19
- Cppcheck 2.10.3
- Clang Static Analyzer 15.0.0

## Python Dependencies:

The code is written in python 3.7. The program requires the following python packages:

Follow `requirements.txt` to see the python APIs used in the repository to reproduce the result. Run the following command to create a virtual environment, activate it and install all thre required python dependencies.

```
conda create -n iotvul python==3.8
conda activate iotvul
pip install pip==23.3.1
pip install -r requirements.txt
```

## Acknowledgements:

This work is a part of the [ENViSEC](https://smartseclab.com/envisec/) project which has received funding from the‌ European Union’s Horizon 2020 research and innovation program within the framework of the NGI POINTER Project funded under grant agreement# 871528.# IoTvulCode
