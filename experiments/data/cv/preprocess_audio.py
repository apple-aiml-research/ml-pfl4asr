#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import csv
import os
import sys

import pandas
import sox
from joblib import Parallel, delayed


def save(inputfile):
    outfile = inputfile.replace(".mp3", ".flac").replace("clips", "flac-16kHz")
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(
        file_type="flac", encoding="signed-integer", rate=16000, bits=16
    )
    sox_tfm.build(inputfile, outfile)


if __name__ == "__main__":
    all_files = []
    for name in ["train.tsv", "dev.tsv", "test.tsv"]:
        print(sys.argv[1] + "/" + name)
        data = pandas.read_csv(
            sys.argv[1] + "/" + name,
            sep="\t",
            engine=None,
            encoding="utf8",
            quoting=csv.QUOTE_NONE,
        )
        all_files += list(data["path"])

    os.mkdir(sys.argv[1] + "/flac-16kHz/")
    all_files = [sys.argv[1] + "/clips/" + i for i in all_files]
    with Parallel(n_jobs=30) as parallel:
        parallel(delayed(save)(inputfile) for inputfile in all_files)
