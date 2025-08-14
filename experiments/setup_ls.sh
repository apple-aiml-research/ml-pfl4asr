#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
pushd experiments || exit

# create fl data for ls clients
python3 ./scripts/create_fl_list.py \
  --input_csv=lists/train-all-960.csv \
  --client_key=client_id \
  --output_dir=lists/train-all-960-fl-data \
  --tar_name=train-clean-100.tar,train-clean-360.tar,train-other-500.tar

python3 ./scripts/create_fl_list.py \
  --input_csv=lists/train-860.csv \
  --client_key=client_id \
  --output_dir=lists/train-860-fl-data \
  --tar_name=train-clean-100.tar,train-clean-360.tar,train-other-500.tar

popd || exit
  