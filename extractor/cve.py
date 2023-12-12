"""
Copyright (C) 2023 SmartSecLab, Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT
@Programmer: Guru Bhandari

File Description:
Code to retrieve the references of IoT projects.
Obtaining and processing CVE json **files**
The code is to download nvdcve zip files from NIST since 2002 to the current year,
unzip and append all the JSON files together,
and extracts all the entries from JSON files of the projects.
"""

import datetime
import json
import os
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
import yaml
from pandas import json_normalize

# ---------------------------------------------------------------------------------------------------------------------

urlhead = "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-"
urltail = ".json.zip"
initYear = 2002
currentYear = datetime.datetime.now().year
DATA_PATH = "data"

SAMPLE_LIMIT = 0

# Consider only current year CVE records when sample_limit>0 for the simplified example.
if SAMPLE_LIMIT > 0:
    initYear = currentYear

df = pd.DataFrame()

ordered_cve_columns = [
    "cve_id",
    "published_date",
    "last_modified_date",
    "description",
    "nodes",
    "severity",
    "obtain_all_privilege",
    "obtain_user_privilege",
    "obtain_other_privilege",
    "user_interaction_required",
    "cvss2_vector_string",
    "cvss2_access_vector",
    "cvss2_access_complexity",
    "cvss2_authentication",
    "cvss2_confidentiality_impact",
    "cvss2_integrity_impact",
    "cvss2_availability_impact",
    "cvss2_base_score",
    "cvss3_vector_string",
    "cvss3_attack_vector",
    "cvss3_attack_complexity",
    "cvss3_privileges_required",
    "cvss3_user_interaction",
    "cvss3_scope",
    "cvss3_confidentiality_impact",
    "cvss3_integrity_impact",
    "cvss3_availability_impact",
    "cvss3_base_score",
    "cvss3_base_severity",
    "exploitability_score",
    "impact_score",
    "ac_insuf_info",
    "reference_json",
    "problemtype_json",
]

cwe_columns = [
    "cwe_id",
    "cwe_name",
    "description",
    "extended_description",
    "url",
    "is_category",
]

# ---------------------------------------------------------------------------------------------------------------------


def rename_columns(name):
    """
    converts the other cases of string to snake_case, and further processing of column names.
    """
    name = name.split(".", 2)[-1].replace(".", "_")
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    name = (
        name.replace("cvss_v", "cvss")
        .replace("_data", "_json")
        .replace("description_json", "description")
    )
    return name


def preprocess_jsons(df_in):
    """
    Flattening CVE_Items and removing the duplicates
    :param df_in: merged dataframe of all years json files
    """
    print("Flattening CVE items and removing the duplicates...")
    cve_items = json_normalize(df_in["CVE_Items"])
    df_cve = pd.concat([df_in.reset_index(), cve_items], axis=1)

    # Removing all CVE entries which have null values in reference-data at [cve.references.reference_data] column
    df_cve = df_cve[df_cve["cve.references.reference_data"].str.len() != 0]

    # Re-ordering and filtering some redundant and unnecessary columns
    df_cve = df_cve.rename(columns={"cve.CVE_data_meta.ID": "cve_id"})
    df_cve = df_cve.drop(
        labels=[
            "index",
            "CVE_Items",
            "cve.data_type",
            "cve.data_format",
            "cve.data_version",
            "CVE_data_type",
            "CVE_data_format",
            "CVE_data_version",
            "CVE_data_numberOfCVEs",
            "CVE_data_timestamp",
            "cve.CVE_data_meta.ASSIGNER",
            "configurations.CVE_data_version",
            "impact.baseMetricV2.cvssV2.version",
            "impact.baseMetricV2.exploitabilityScore",
            "impact.baseMetricV2.impactScore",
            "impact.baseMetricV3.cvssV3.version",
        ],
        axis=1,
        errors="ignore",
    )

    # renaming the column names
    df_cve.columns = [rename_columns(i) for i in df_cve.columns]

    # ordering the cve columns
    df_cve = df_cve[ordered_cve_columns]
    return df_cve


def import_cves():
    """
    gathering CVE records by processing JSON files.
    """
    print("-" * 70)
    # if db.table_exists('cve'):
    #     print('The cve table already exists, loading and continuing extraction...')
    #     # df_cve = pd.read_sql(sql="SELECT * FROM cve", con=db.conn)
    # else:
    for year in range(initYear, currentYear + 1):
        extract_target = "nvdcve-1.1-" + str(year) + ".json"
        zip_file_url = urlhead + str(year) + urltail

        # Check if the directory already has the json file or not ?
        if os.path.isfile(Path(DATA_PATH) / "json" / extract_target):
            print(
                f"Reusing the {year} CVE json file that was downloaded earlier...")
            json_file = Path(DATA_PATH) / "json" / extract_target
        else:
            # url_to_open = urlopen(zip_file_url, timeout=10)
            r = requests.get(zip_file_url)
            z = ZipFile(BytesIO(r.content))  # BytesIO keeps the file in memory
            json_file = z.extract(extract_target, Path(DATA_PATH) / "json")

        with open(json_file) as f:
            yearly_data = json.load(f)
            if year == initYear:  # initialize the df_methods by the first year data
                df_cve = pd.DataFrame(yearly_data)
            else:
                df_cve = df_cve.append(pd.DataFrame(yearly_data))
            print(f"The CVE json for {year} has been merged")

    df_cve = preprocess_jsons(df_cve)
    df_cve = df_cve.applymap(str)
    print(len(df_cve))
    assert df_cve.cve_id.is_unique, "\nPrimary keys are not unique in cve records!"
    # df_cve.to_sql(name="cve", con=db.conn, if_exists="replace", index=False)
    print("All CVEs have been merged into the cve table")
    print("\nExamples: \n", df_cve.head(5))
    df_cve.to_csv(Path(DATA_PATH) / "cve-records.csv")
    print("-" * 70)


def check_project_in_cve(df, prj):
    """Check if a project is in CVE."""
    prj = prj.split('/')[-1]
    if len(df[df.description.str.lower().str.contains(prj.lower())]) > 0:
        print(f'Project [{prj}] is in CVE list.')
    else:
        print(f'Project [{prj}] is not in CVE list.')


def run_checking():
    with open("ext_projects.yaml") as fp:
        config = yaml.safe_load(fp)
        print(f'List of projects: \n{config["projects"]}\n')

        df = pd.read_csv(cve_file)
        print('CVE records have been loaded.')

        for prj in config["projects"]:
            check_project_in_cve(df, prj)


if __name__ == "__main__":
    cve_file = Path(DATA_PATH) / "cve-records.csv"

    # crawl cve_records if not exists
    if cve_file.is_file() is False:
        import_cves()

    run_checking()
