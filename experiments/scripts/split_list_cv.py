#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Create client percentage splits")

parser.add_argument("--input_csv", type=str)
parser.add_argument("--proportion1", type=float)
parser.add_argument("--output1", type=str)
parser.add_argument("--output2", type=str)
parser.add_argument("--random-state", type=int, default=123)

if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    if not (args.proportion1 > 0 and args.proportion1 < 1):
        raise RuntimeError("Wrong proportions")

    data = pd.read_csv(args.input_csv, sep=" ")

    client_ids = set(data["client_id"])
    client_ids1, client_ids2 = train_test_split(
        list(client_ids), train_size=args.proportion1, random_state=args.random_state
    )

    data1 = data[data["client_id"].isin(client_ids1)]
    data2 = data[~data["client_id"].isin(client_ids1)]

    print("data1.shape:", data1.shape)
    print("data2.shape:", data2.shape)

    with open(args.output1, "w+") as f1:
        f1.write(data1.to_csv(index=False, sep=" "))

    with open(args.output2, "w+") as f2:
        f2.write(data2.to_csv(index=False, sep=" "))
