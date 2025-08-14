#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import argparse

import pandas as pd
from unidecode import unidecode

parser = argparse.ArgumentParser(description="Apply unidecode to transcriptions")

parser.add_argument("--input_csv", type=str)
parser.add_argument("--output_csv", type=str)

if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv, sep=" ")
    data = data.sample(frac=1, random_state=444).reset_index(drop=True)

    data["transcription"] = data["transcription"].apply(lambda x: unidecode(x))

    with open(args.output_csv, "w+") as f:
        f.write(data.to_csv(index=False, sep=" "))
