#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import argparse
import tarfile
from pathlib import Path

import pandas

parser = argparse.ArgumentParser(description="Create federated data lists")
parser.add_argument("--input_csv", type=str, default=None)
parser.add_argument("--client_key", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--tar_name", type=str, default=None)
parser.add_argument("--tar_key", type=str, default="tar_file")
parser.add_argument("--input_key", type=str, default="filename")

if __name__ == "__main__":
    args = parser.parse_args()
    dst = Path(args.output_dir)
    dst.mkdir(exist_ok=True, parents=True)
    data = pandas.read_csv(args.input_csv, sep=" ")
    names = []
    csvs = []
    sizes = []
    original_tar = dict()
    members = dict()
    for t_name in args.tar_name.split(","):
        original_tar[t_name] = tarfile.open(t_name, "r")
        members[t_name] = {
            member.name: member for member in original_tar[t_name].getmembers()
        }
    for name, group in data.groupby(args.client_key, sort=False):
        client_path_csv = dst / Path(str(name)).with_suffix(".csv")
        group[args.tar_key] = client_path_csv.with_suffix(".tar")
        with tarfile.open(client_path_csv.with_suffix(".tar"), "w") as new_tar:
            for fname in group[args.input_key]:
                for t_name, member in members.items():
                    if fname in member:
                        extracted = original_tar[t_name].extractfile(member[fname])
                        info = member[fname]
                        new_tar.addfile(info, extracted)
        group.to_csv(client_path_csv, sep=" ", index=False)
        sizes.append(len(group))
        csvs.append(str(client_path_csv).replace("lists/", ""))
        names.append(name)
    pandas.DataFrame({"csv": csvs, "client_id": names, "client_size": sizes}).to_csv(
        Path(args.input_csv).parent.joinpath("ids-" + Path(args.input_csv).name),
        sep=" ",
        index=False,
    )
    for t_name in original_tar:
        original_tar[t_name].close()
