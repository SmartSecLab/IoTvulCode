# Instruction on dataset extraction:

README.md instruction of execution of extract.py file to extract vulnerability dataset based on the `config/extractor.yaml` configuration file.

## Run the extration script:

Once required packages were installed, run the command to extract the vulnerability database from the given input projects as listed in `config/extractor.yaml`:

```
python3 -m extractor.extract
```

## Configurations:

The `config/extractor.yaml` lists the configurations for the extraction the vulnerability data from the given `projects:` list, for example:

```
projects: [
projects/gnucobol,
projects/contiki,
projects/linux-rpi,
projects/ARMmbed,
projects/FreeRTOS,
projects/micropython,
projects/openwrt,
projects/RIOT,
projects/tinyos,
]
save:
    database: data/IoT.db  # name of the extracted database
    override: True  # replace the existing dataset in the dir
    threshold_lines: 5  # to filter the short functions
    benign_ratio: 0.25
    seed: 41  # for reproducibility
    drop_dup: True
    refine_on_every: 3000 # filter after scanning #files
    apply_guesslang: False # use guesslang or extension based approach
```

Do you want to use guess lang to classify files or use extension?
