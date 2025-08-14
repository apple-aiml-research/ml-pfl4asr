#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
export lang="$1"
pushd experiments || exit
python3 ./scripts/create_unidecode_list.py --input_csv=lists/"${lang}"-train.csv --output_csv=lists/"${lang}"-train-transf.csv
python3 ./scripts/create_unidecode_list.py --input_csv=lists/"${lang}"-test.csv --output_csv=lists/"${lang}"-test-transf.csv
python3 ./scripts/create_unidecode_list.py --input_csv=lists/"${lang}"-dev.csv --output_csv=lists/"${lang}"-dev-transf.csv

python3 ./scripts/split_list_cv.py \
  --input_csv=lists/"${lang}"-train-transf.csv \
  --output1=lists/"${lang}"-train-10p-transf.csv \
  --output2=lists/"${lang}"-train-90p-transf.csv \
  --proportion1=0.1 \
  --random-state=123

# create fl data for cv clients
python3 ./scripts/create_fl_list.py \
  --input_csv=lists/"${lang}"-train-transf.csv \
  --client_key=client_id \
  --output_dir=lists/"${lang}"-cv-fl-data \
  --tar_name="${lang}".tar

# create fl data for cv 90 percentage of clients
python3 ./scripts/create_fl_list.py \
  --input_csv=lists/"${lang}"-train-90p-transf.csv \
  --client_key=client_id \
  --output_dir=lists/"${lang}"-cv90-fl-data \
  --tar_name="${lang}".tar

popd || exit
  