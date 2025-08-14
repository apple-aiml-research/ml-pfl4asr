#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import json
import pickle
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import mlx.data.core as dxcore
from flax import linen as nn
from loguru import logger

from ..modules.cape1d import ConfigCAPE1d
from ..modules.specaugment import ConfigSpecAugment, SpecAugment
from ..modules.transformer import ConfigConvBlock, TransformerEncoder
from ..modules.transformer_block import ConfigTransformerBlock


def create_model(
    use_cape: bool,
    block_config: ConfigTransformerBlock,
    trie: dxcore.CharTrie,
    input_mel: int = 80,
    n_blocks: int = 36,
):
    logger.info("Creating model...")
    saug_config = ConfigSpecAugment(
        num_freq_masks=2,
        freq_mask_max_width=30,
        num_time_masks=10,
        time_mask_max_width=50,
        time_mask_width_ratio=0.1,
        avg_mask_strategy=False,
    )
    saug = SpecAugment(saug_config)
    if use_cape:
        logger.info("CAPE embedding will be used...")
        cape_config = ConfigCAPE1d(
            emb_dim=block_config.emb_dim,
            # 30s left, 30s right
            max_global_shift=30,
            # frame duration is 30ms=0.03s
            normalize=True,
            freq_scale=30,
        )
        positions_delta = 0.03
    else:
        logger.info("Absolute sinpos embedding will be used...")
        cape_config = ConfigCAPE1d(
            emb_dim=block_config.emb_dim,
        )
        positions_delta = None
    model = TransformerEncoder(
        input_mel=input_mel,
        nlabel=trie.num_keys(),
        conv_config=ConfigConvBlock(),
        cape_config=cape_config,
        block_config=block_config,
        n_blocks=n_blocks,
        dtype=block_config.dtype,
        positions_delta=positions_delta,
    )
    return saug, model


def init_model(
    model: nn.Module, model_rng_keys: dict, batch_size: int, num_mel: int, seed: int
):
    rng = jax.random.PRNGKey(seed)
    rng_vals = jax.random.split(rng, len(model_rng_keys) + 1)
    x = jax.random.uniform(rng_vals[0], (batch_size, 1000, num_mel))  # 10s, 80 filters
    l = jnp.ones(batch_size) * 1000

    params = model.init(
        {key: rng_vals[index + 1] for index, key in enumerate(model_rng_keys)},
        x,
        l,
        deterministic=True,
    )
    return params


def save_model(params, filename: str):
    sparams = jax.tree_util.tree_map(lambda x: x[0], params)  # only on GPU 0
    with open(filename, "wb") as f:
        f.write(flax.serialization.to_bytes(sparams))


def load_model(params, filename: str):
    if not Path(filename).exists():
        logger.info(f"{filename} does not exist, don't reinitialize")
        return params
    bytes = Path(filename).read_bytes()
    params_new = flax.serialization.from_bytes(params, bytes)
    return params_new


def save_state(state, filename: str):
    with open(filename, "wb") as handle:
        pickle.dump(
            jax.tree_util.tree_map(lambda x: x[0], state),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_state(filename: str):
    with open(filename, "rb") as handle:
        opt_state = pickle.load(handle)
    return opt_state


def save_other_state(data: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_other_state(filename: str):
    if not Path(filename).exists():
        logger.info(f"{filename} does not exist, use empty state")
        return dict()
    with open(filename, "r") as f:
        data = json.load(f)
    return data
