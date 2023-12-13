[![source under MIT licence](https://img.shields.io/badge/source%20license-MIT-green)](LICENSE.txt)
[![data under CC BY 4.0 license](https://img.shields.io/badge/data%20license-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)

# Vulnerability detection in IoT software analyzing open-source code:

In this study, we have presented an IoT vulnerability data extraction tool and machine learning methods to detect vulnerabilities in the C\C++ source code of IoT operating systems(OS) and applications.
The source code of various IoT OSs and applications was used to create a binary and multi-class labeled dataset including both vulnerable and benign samples. The types of vulnerabilities in the presented dataset are linked to the Common Weakness Enumeration (CWE) records.

## Dataset extraction approach for IoT vulnerability analysis:

The proposed method for vulnerability data collection is as follows:

![framework](figure/framework.png?raw=true "The proposed framework for vulnerability data collection")

Follow the IoT vulnerability dataset extraction instruction [here](extractor/README.md)

## ML method for IoT vulnerability detection:

The novel vulnerability detection approach in IoT OSs and applications:

![framework](figure/MLframework.png?raw=true "The proposed method for vulnerability detection in IoT OSs and applications")

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

The research presented in this paper has benefited from the Experimental Infrastructure for Exploration of Exascale Computing (eX3), which is financially supported by the Research Council of Norway under contract 270053.
