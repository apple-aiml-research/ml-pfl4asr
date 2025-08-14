#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import sys
import tarfile
from pathlib import Path

import pandas


def process_tar(tar_name):
    with tarfile.open(tar_name + ".tar", "r:*") as tar:
        file_names = tar.getnames()
        audio = [name for name in file_names if name.endswith(".flac")]
        text = [name for name in file_names if name.endswith(".txt")]
        text_data = dict()
        for text_file in text:
            f = tar.extractfile(text_file)
            if f is None:
                continue
            for line in f:
                tmp = line.decode("utf-8").strip().split(" ")
                text_data[tmp[0]] = " ".join(tmp[1:]).lower()
        transcription = [text_data[Path(name).name.split(".")[0]] for name in audio]
        client_id = [name.split("/")[-3] for name in audio]
        return pandas.DataFrame(
            {
                "id": audio,
                "filename": audio,
                "transcription": transcription,
                "tar_file": [tar_name + ".tar"] * len(audio),
                "client_id": client_id,
            }
        )


if __name__ == "__main__":
    dst = Path(sys.argv[1])
    dst.mkdir(exist_ok=True, parents=True)

    train_data = dict()
    for tar_name in ["train-clean-100", "train-clean-360", "train-other-500"]:
        train_data[tar_name] = process_tar(tar_name)
    train_all = pandas.concat(list(train_data.values()), axis=0)
    train_all = train_all.sample(frac=1, random_state=444).reset_index(drop=True)
    train_all.to_csv(dst.joinpath("train-all-960.csv"), sep=" ", index=False)
    train_860 = pandas.concat(
        [train_data["train-clean-360"], train_data["train-other-500"]], axis=0
    )
    train_860 = train_860.sample(frac=1, random_state=444).reset_index(drop=True)
    train_860.to_csv(dst.joinpath("train-860.csv"), sep=" ", index=False)
    train_100 = (
        train_data["train-clean-100"]
        .sample(frac=1, random_state=444)
        .reset_index(drop=True)
    )
    train_100.to_csv(dst.joinpath("train-clean-100.csv"), sep=" ", index=False)

    for tar_name in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        data = process_tar(tar_name)
        data.to_csv(dst.joinpath(tar_name).with_suffix(".csv"), sep=" ", index=False)
