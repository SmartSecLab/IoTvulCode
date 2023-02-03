import collections
import pandas as pd
from matplotlib import pyplot as plt
import json 
import ast
import re
import os
import csv
import subprocess
import requests
import tempfile
from io import BytesIO, StringIO
from zipfile import ZipFile
from guesslang import Guess


def check_internet(url):
    response = requests.get(url)
    return True if response.status_code < 400 else False
    
    
def retrieve_zip(url):
    """ Fetching list of C/C++ files from zip file of the project url. 
    """
    if check_internet(url):
        r = requests.get(url)
        # BytesIO keeps the file in memory
        return ZipFile(BytesIO(r.content))  
    else:
        print('Internet is not working!')
        return None


def guess_pl(file, zip_obj=None):
    """ guess programming language of the input file. 
    """ 
    guess = Guess()
    if zip_obj is not None:
        # extract a specific file from the zip container
        with zip_obj.open(file, 'r') as f:
            lang = guess.language_name(f.read())
    else:
        with open(file, 'r', encoding= 'unicode_escape') as f:
            lang = guess.language_name(f.read())
    return lang


def find_flaw(file_or_dir):
    """ find flaws ini the file using flawfinder tool
    return : flawfinder output as a CSV file.
    Usage: cmd = 'flawfinder --csv --inputs ' + path + ' >> output.csv'
    """
    if os.path.isfile(file_or_dir):
        cmd = 'flawfinder --csv ' + file_or_dir
    elif ps.path.isdir(file_or_dir):
        cmd = 'flawfinder --csv --inputs ' + file_or_dir
        
    process = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE)
    output = process.stdout.read()
    return pd.read_csv(StringIO(str(output,'utf-8')))
    

def file2df(file, zip_obj=None):
    """ convert zipped file stream - tempfile to pandas dataframe. 
    """
    file_content = ''
    
    if zip_obj:
        # io.StringIO(sf.read().decode("utf-8")).read()
        with zip_obj.open(file) as fc:
            # file_content = fc.read().encode('UTF-8')
            file_content = fc.read()
    else:
        with open(file) as fc:
            file_content = fc.read().encode('UTF-8')

    fp = tempfile.NamedTemporaryFile(suffix='_Flawfinder',
                                    prefix='Filename_')
    # deal with the temp file of extracted zipped file
    try:
        fp.write(file_content)
        fp.seek(0)  # move reader's head to the initial point of the file. 
        file_name = fp.name
        df = find_flaw(file_name)
    except OSError:
        print("Could not open/read file:", fp)
        sys.exit(1)
    finally:
        fp.close()
    return df.reset_index(drop=True)


def url_zip2df(url):
    """ concatenate all the output dataframes of all the files
    """
    print('='*35)
    print('Generating composite dataframe from the given project URL of zip file...\n', url)
    print('='*35)
    zipobj = retrieve_zip(url)
    files = zipobj.namelist() 
    selected_files = [x for x in files if guess_pl(x, zipobj) in ['C', 'C++']]
    df = pd.concat([file2df(selected_files[i], zipobj) for i in range(len(selected_files))])
    print('Shape of the data:', df.shape)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    url  = 'https://sourceforge.net/projects/contiki/files/Contiki/Contiki%202.4/contiki-sky-2.4.zip/download'
    url_zip2df(url).to_csv('../data/')