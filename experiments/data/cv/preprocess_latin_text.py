#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas
from unidecode import unidecode

de_dict = set(
    (b" -'abcdefghijklmnopqrstuvwxyz\xc3\xb6\xc3\xa4\xc3\xbc\xc3\x9f").decode()
)
fr_dict = set(
    (
        b" -'abcdefghijklmnopqrstuvwxyz\xc3\xa0\xc3\xa2\xc3\xa6\xc3\xa7\xc3\xa9\xc3\xa8\xc3\xaa\xc3\xab\xc3\xae\xc3\xaf\xc3\xb4\xc5\x93\xc3\xb9\xc3\xbb\xc3\xbc\xc3\xbf"
    ).decode()
)
en_dict = set(" -'abcdefghijklmnopqrstuvwxyz")
all_dicts = {
    "de": de_dict,
    "fr": fr_dict,
    "en": en_dict,
    "all": set().union(de_dict, fr_dict, en_dict),
}

print(len(all_dicts["all"]), all_dicts["all"])
with open("latin_letters.txt", "w") as f:
    for el in all_dicts["all"]:
        f.write(el + "\n")

en_chars = set(" abcdefghijklmnopqrstuvwxyz")
punctuation = '!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~'
chars_all = defaultdict(int)

lang_dict = all_dicts["all"]


def process(sentence):
    # lower
    sentence = sentence.lower()
    out = ""
    for index, ch in enumerate(sentence):
        # remove punctuation except ' and -
        if ch in set(punctuation):
            continue
        elif ch in lang_dict:
            # keep in vocab
            out += ch
        elif len(set(unidecode(ch).lower()) - en_chars) == 0:
            # convert to eng letters and check if it is only them, then add
            out += unidecode(ch).lower()
        elif ch == b"\xe2\x80\x99".decode():
            if (index > 0 and unidecode(sentence[index - 1]) in punctuation + "-") or (
                index + 1 < len(sentence)
                and unidecode(sentence[index + 1]) in punctuation + "-"
            ):
                pass
            else:
                out += unidecode(ch).lower()
        else:
            # else ignore token
            continue
    for k in out:
        chars_all[k.encode("utf-8")] += 1
    out = out.replace("- ", " ")
    out = out.replace(" -", " ")
    out = re.sub(" +", " ", out)
    return out


dst = Path(sys.argv[2])
dst.mkdir(exist_ok=True, parents=True)

for name in ["train.tsv", "dev.tsv", "test.tsv"]:
    print(sys.argv[1] + "/" + name)
    data = pandas.read_csv(sys.argv[1] + "/" + name, sep="\t", quoting=csv.QUOTE_NONE)
    data_new = dict()
    data_new["filename"] = [
        sys.argv[1] + "/flac-16kHz/" + p.replace("mp3", "flac")
        for p in list(data["path"])
    ]
    data_new = pandas.DataFrame(data_new)
    data_new["id"] = data_new["filename"]
    data_new["transcription"] = [process(str(tr)) for tr in data["sentence"]]
    data_new["raw_transcription"] = data["sentence"]
    data_new["client_id"] = data["client_id"]
    data_new["tar_file"] = sys.argv[1].split("/")[-1] + ".tar"
    print("data len", len(data_new))
    data_new = data_new.dropna()
    print("data len after dropna", len(data_new))

    data_new.to_csv(
        dst.joinpath(sys.argv[1].split("/")[-1] + "-" + name.replace("tsv", "csv")),
        sep=" ",
        index=False,
    )

s = b""
for k, v in chars_all.items():
    if k.decode() in lang_dict:
        continue
    s += k
print(s)
