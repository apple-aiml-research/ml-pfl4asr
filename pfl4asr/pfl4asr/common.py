#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass

from jax import numpy as jnp


@dataclass
class ConfigTrainingShared:
    # Comma separated eval lists
    eval: str = "dev-clean.csv,dev-other.csv,test-clean.csv,test-other.csv"
    # common prefix to eval, train csv files
    lists_dir: str = "lists"
    # either csv file column name containing tar file where audio is stored,
    # or the tar file itself if all data are stored in it
    tar: str = ""
    input_key: str = "file"
    target_key: str = "transcription"
    target_nopad: bool = False
    max_target_len: int = 400
    batching_buffer_size: int = 10
    # validation batch size given in seconds per 1 device
    valid_batch_size: int = 300
    threads: int = 10
    use_cape: bool = True
    dtype: jnp.dtype = jnp.float32
    n_blocks: int = 36
    seed: int = 42
    start_saug: int = 5000
    # logging report
    report: int = 500
    runname: str = "artifacts/jax-model"
    # Path to the model, optimizer state, and state variables checkpoints
    restore_checkpoint: str | None = None
    is_restore_optim: bool = True
    # Use these additional characters to construct the trie used for tokenization. Models with different
    # characters are currently incompatible. E.g. '-'.
    additional_chars: str | None = None
    num_mels: int = 80
    sample_rate_hz: int = 16000
    frame_size_ms: int = 25
    frame_stride_ms: int = 10
    host_ip_address: str | None = None
    distributed_port: int | str | None = None
    rank: int = 0
    world_size: int = 1
