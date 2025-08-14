#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
pushd experiments || exit
for name in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other 
do
    wget https://www.openslr.org/resources/12/${name}.tar.gz 
    gunzip ${name}.tar.gz
done

python data/ls/generate_lists.py lists

popd || exit