#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

pushd experiments || exit
# 1. Download data with wget -O "name" "url"
# 2. Uncompress data
for k in *.tar.gz
do
  tar -xzf "$k"
done
# 3. Rename folder
mv cv-corpus-13.0-2023-03-09 cv-v13.0

# 4. Preprocess all audio from mp3 format to flac with downsampling to 16kHz from 48kHz
for val in fr de en
do
	python data/cv/preprocess_audio.py cv-v13.0/${val}/
done

# 5. Compress back to tar files for efficient reading from disk
for val in fr de en
do
  tar --exclude='*.mp3' -cf ${val}.tar  cv-v13.0/${val}/
done

# 6. Preprocess text (Latin, others) 
for val in fr de en
do
  python data/cv/preprocess_latin_text.py cv-v13.0/${val} lists
done

popd || exit