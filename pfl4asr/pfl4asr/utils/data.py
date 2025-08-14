#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import jax
import mlx.data as dx
import mlx.data.core as dxcore
import numpy as np
from einops import rearrange
from loguru import logger
from mlx.data.features import mfsc

from ..modules.functions import length_masked_normalize2d
from .speech import mfsc, sliding_window_output_length

BLANK_TOKEN = "@"


def construct_eng_char_trie_for_ctc(additional_chars: str):
    trie = dxcore.CharTrie()
    trie.insert(" ")
    trie.insert("'")
    for c in range(ord("a"), ord("z") + 1):
        trie.insert(chr(c))
    if additional_chars:
        for c in additional_chars:
            trie.insert(c)
    trie.insert(BLANK_TOKEN)  # blank
    return trie


def create_dataset(
    name,
    trie,
    batch_size_s,
    max_target_length=500,
    n_threads=8,
    world_size=1,
    rank=0,
    # either csv file column name containing tar file where audio is stored,
    # or the tar file itself if all data are stored in it
    tar="",
    target_pad=False,
    input_key="file",
    target_key="transcription",
    sample_rate_hz=16000,
    pad_to_seconds=5,
    pad_to_batch=8,
    batching_buffer_size=512,
    shuffle_buffer_size=10000,
    # used for central training
    loop=1,
    # used for pfl to repeat dataset in batches to be efficient as it may be only 1 batch on device
    loop_batches=0,
    max_audio_len_s=30,
):
    logger.info(f"Creating dataset from {name}")
    dataset = dx.stream_csv_reader(name, " ", quote='"')
    if world_size > 1:
        # partition across devices
        dataset = dataset.partition(world_size, rank)
    dataset = dataset.repeat(loop)
    dataset = dataset.shuffle(shuffle_buffer_size)
    if tar:
        dataset = dataset.read_from_tar(
            tar,
            input_key,
            "input",
            from_key=not tar.endswith(".tar"),
        ).load_audio("input", "input", from_memory=True, sample_rate=sample_rate_hz)
    else:
        dataset = dataset.load_audio(input_key, "input", sample_rate=sample_rate_hz)
    # 1 channel audio assumption!
    dataset = dataset.squeeze("input", -1)
    dataset = dataset.shape("input", "input_length", 0)
    dataset = dataset.filter_by_shape(
        "input", 0, low=1, high=max_audio_len_s * sample_rate_hz
    )
    dataset = dataset.tokenize(
        target_key,
        trie,
        output_key="target",
    )

    if target_pad:  # key, dim, left, right, value
        dataset = dataset.pad(
            "target", 0, 1, 1, trie.search(" ").id
        )  # pad target with silence
    dataset = dataset.shape("target", "target_length", 0)
    # batching
    dataset = dataset.dynamic_batch(
        batching_buffer_size,
        "input",
        max_data_size=int(batch_size_s * sample_rate_hz),
        shuffle=True,
        pad={"input": 0, "target": trie.search(BLANK_TOKEN).id},
        num_threads=n_threads,
    )
    # pad to every pad_to_seconds
    dataset = dataset.pad_to_multiple(
        "input", 1, int(pad_to_seconds * sample_rate_hz), trie.search(BLANK_TOKEN).id
    )
    dataset = dataset.pad_to_size("target", 1, max_target_length, trie.search(" ").id)
    # pad to batch
    dataset = dataset.pad_to_multiple(
        "input", 0, pad_to_batch, trie.search(BLANK_TOKEN).id
    )
    dataset = dataset.pad_to_multiple("target", 0, pad_to_batch, trie.search(" ").id)
    dataset = dataset.pad_to_multiple("input_length", 0, pad_to_batch, 0)
    dataset = dataset.pad_to_multiple("target_length", 0, pad_to_batch, 0)
    if loop_batches != 0:
        # will recreate the stream from csv, thus we shuffle again big buffer
        dataset = dataset.repeat(loop_batches)
    dataset = dataset.prefetch(4, 4)
    return dataset


def get_next_sample_on_device(dataset, trie, devices):
    sample, sample_device = None, None
    num_local_devices = len(devices)
    try:
        sample = dataset.__next__()
        x = sample["input"]
        y = sample["target"]
        x_length = sample["input_length"]
        y_length = sample["target_length"]
        # sanity checks
        if np.isnan(x.sum()):
            logger.info("BUG: input contains NaN")
        if np.isnan(y.sum()):
            logger.info("BUG: target contains NaN")
        if y.min() < 0 or y.max() > trie.num_keys():
            logger.info("BUG: target has invalid values")
        if x_length.min() < 0 or x_length.max() > x.shape[1]:
            logger.info("BUG: x_length has invalid values")
        if y_length.min() < 0 or y_length.max() > y.shape[1]:
            logger.info("BUG: x_length has invalid values")

        x = rearrange(x, "(devices b) ... -> devices b ...", devices=num_local_devices)
        y = rearrange(y, "(devices b) ... -> devices b ...", devices=num_local_devices)
        x_length = rearrange(
            x_length, "(devices b) ... -> devices b ...", devices=num_local_devices
        )
        y_length = rearrange(
            y_length, "(devices b) ... -> devices b ...", devices=num_local_devices
        )
        x = jax.device_put_sharded(list(x), devices)
        y = jax.device_put_sharded(list(y), devices)
        x_length = jax.device_put_sharded(list(x_length), devices)
        y_length = jax.device_put_sharded(list(y_length), devices)
        sample_device = (x, y, x_length, y_length)
    except StopIteration:
        logger.info("Dataset is exhausted")
    return sample, sample_device


def create_mfsc_func(
    num_mels: int = 80,
    sample_rate_hz: int = 16000,
    frame_size_ms: int = 25,
    frame_stride_ms: int = 10,
):
    mfsc_base = mfsc(
        n_filterbank=num_mels,
        sampling_freq=sample_rate_hz,
        frame_size_ms=frame_size_ms,
        frame_stride_ms=frame_stride_ms,
    )
    mfsc_batch = jax.vmap(mfsc_base)  # batch

    def mfsc_func(x, x_length):
        x = mfsc_batch(x)
        x_length = sliding_window_output_length(
            frame_size_ms * sample_rate_hz // 1000,
            frame_stride_ms * sample_rate_hz // 1000,
            x_length,
        )
        x = length_masked_normalize2d(x, x_length)
        return x, x_length

    return mfsc_func
