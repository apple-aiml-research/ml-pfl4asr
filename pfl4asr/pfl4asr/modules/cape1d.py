#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import jax.numpy as jnp
from einops import rearrange, repeat
from flax import linen as nn
from jax import random

Array = Any


def _generate_symmetric_uniform(rng, shape, value):
    return random.uniform(rng, shape, dtype=jnp.float32) * 2 * value - value


@dataclass
class ConfigCAPE1d:
    # embedding dimension
    emb_dim: int
    # max global shift augmentation
    max_global_shift: float = 0.0
    # max local shift augmentation
    max_local_shift: float = 0.0
    # max global scaling augmentation
    max_global_scale: float = 1.0
    # bool, make positions to be centred around zero position
    normalize: bool = False
    # float, rescale the frequency (helpful in ASR)
    freq_scale: float = 1.0


class CAPE1d(nn.Module):
    """
    Construct sinusoidal positional embedding. Allows applying CAPE augmentation on positions.
    """

    config: ConfigCAPE1d

    def setup(self):
        if self.config.max_global_shift < 0:
            raise RuntimeError(
                f"Max global shift is {self.config.max_global_shift} but should be >= 0."
            )
        if self.config.max_local_shift < 0:
            raise RuntimeError(
                f"Max local shift is {self.config.max_local_shift}, but should be >= 0."
            )
        if self.config.max_global_scale < 1:
            raise RuntimeError(
                f"Global scaling is {self.config.max_global_scale}, but should be >= 1."
            )

        self.config.freq = self.config.freq_scale * jnp.exp(
            -2.0
            * jnp.floor(jnp.arange(self.config.emb_dim) / 2)
            * (math.log(1e4) / self.config.emb_dim)
        )

        self.config.cos_shifts = (jnp.pi / 2.0) * (jnp.arange(self.config.emb_dim) % 2)

    def __call__(
        self,
        x: Array,
        x_lengths: Optional[Array] = None,
        positions_delta: Optional[Union[int, Array]] = None,
        deterministic: Optional[bool] = None,
    ) -> Array:
        batch_size, n_tokens, _ = x.shape  # b, t, c
        positions = repeat(jnp.arange(n_tokens), "t -> b t", b=batch_size)
        # x_length vector of length in terms of number of t steps
        if x_lengths is not None:
            padding_mask = positions >= x_lengths[:, None]
            positions = jnp.where(
                padding_mask, jnp.zeros(positions.shape) + float("nan"), positions
            )

        if positions_delta is None:
            positions_delta = 1
        else:
            if (
                isinstance(positions_delta, jnp.ndarray)
                and len(positions_delta.shape) == 1
            ):
                positions_delta = rearrange(positions_delta, "b -> b 1")
            positions *= positions_delta

        if self.config.normalize:
            positions -= jnp.nanmean(positions, axis=1, keepdims=True)

        positions = self.augment_positions(positions, positions_delta, deterministic)

        positions = rearrange(positions, "b t -> b t 1")
        product = positions * self.config.freq

        pos_emb = jnp.sin(product + self.config.cos_shifts)
        pos_emb = jnp.nan_to_num(pos_emb, nan=0)

        return pos_emb

    def augment_positions(
        self,
        positions: Array,
        positions_delta: Optional[Union[int, Array]] = None,
        deterministic: Optional[bool] = None,
    ):
        if deterministic:
            return positions

        rng = self.make_rng("cape1d")
        rngs = random.split(rng, 3)

        batch_size, n_tokens = positions.shape

        if self.config.max_global_shift:
            delta = _generate_symmetric_uniform(
                rngs[0], (batch_size,), self.config.max_global_shift
            )
            delta = rearrange(delta, "b -> b 1")
        else:
            delta = 0.0

        if self.config.max_local_shift:
            delta_local = _generate_symmetric_uniform(
                rngs[2], (batch_size, n_tokens), self.config.max_local_shift
            )
            if positions_delta is not None:
                if (
                    isinstance(positions_delta, jnp.ndarray)
                    and len(positions_delta.shape) == 1
                ):
                    positions_delta = rearrange(positions_delta, "b -> b 1")
                delta_local *= positions_delta
        else:
            delta_local = 0.0

        if self.config.max_global_scale > 1.0:
            log_lambdas = _generate_symmetric_uniform(
                rngs[2], (batch_size,), math.log(self.config.max_global_scale)
            )
            log_lambdas = rearrange(log_lambdas, "b -> b 1")
        else:
            log_lambdas = jnp.zeros(1)

        return (positions + delta + delta_local) * jnp.exp(log_lambdas)
