#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Any, Optional, Union

import jax.nn.initializers as initjax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

from .cape1d import CAPE1d, ConfigCAPE1d
from .functions import length_masked_normalize2d
from .transformer_block import ConfigTransformerBlock, TransformerBlock

Array = Any


def _init_dense_bias_function(fan_in):
    def init(rng, shape, dtype):
        return random.uniform(rng, shape, dtype, -1) * jnp.sqrt(1 / fan_in)

    return init


class LengthMaskedNorm2d(nn.Module):
    epsilon: float = 1e-7
    param_dtype: Optional[Any] = None
    dtype: Optional[Any] = None

    @nn.compact
    def __call__(self, input, length):
        scale = self.param(
            "scale", lambda rng, shape: jnp.ones(shape, dtype=self.param_dtype), (1,)
        )
        bias = self.param(
            "bias", lambda rng, shape: jnp.zeros(shape, dtype=self.param_dtype), (1,)
        )
        res, mask = length_masked_normalize2d(input, length, self.epsilon, True)
        res = jnp.where(mask, res * scale + bias, 0)
        return res.astype(self.dtype)


@dataclass
class ConfigConvBlock:
    kernel: int = 7
    stride: int = 3
    dtype: Optional[Any] = None


class TransformerEncoder(nn.Module):
    input_mel: int
    nlabel: int
    conv_config: ConfigConvBlock
    cape_config: ConfigCAPE1d
    block_config: ConfigTransformerBlock
    n_blocks: int = 36
    positions_delta: Optional[Union[int, Array]] = (None,)
    dtype: Optional[Any] = None

    def setup(self):
        if self.block_config.ln_norm_type not in {"pre", "post"}:
            raise ValueError(
                "Speech transformer: ln_norm_type should be one of pre, post, spectral, spectral-ln"
            )
        self.conv = nn.Conv(
            features=self.block_config.emb_dim * 2,
            kernel_size=(self.conv_config.kernel,),
            strides=(self.conv_config.stride,),
            kernel_init=initjax.lecun_uniform(),
            bias_init=_init_dense_bias_function(
                self.input_mel * self.conv_config.kernel
            ),
            dtype=self.conv_config.dtype,
        )
        self.do = nn.Dropout(self.block_config.dropout)
        self.ln = LengthMaskedNorm2d(
            dtype=self.dtype,
        )
        # just create simple sinpos
        self.cape = CAPE1d(self.cape_config)

        self.tf_blocks = [
            TransformerBlock(self.block_config) for _ in range(self.n_blocks)
        ]
        if self.block_config.ln_norm_type == "pre":
            self.ln_final = nn.LayerNorm(
                epsilon=self.block_config.ln_epsilon,
                dtype=self.dtype,
            )
        self.linear = nn.Dense(
            features=self.nlabel,
            kernel_init=initjax.lecun_uniform(),
            bias_init=_init_dense_bias_function(self.block_config.emb_dim),
            dtype=self.dtype,
        )

    def __call__(self, x: Array, x_length: Array, deterministic: Optional[bool] = None):
        # expected input (b t c)
        x = self.ln(x, x_length)
        x = self.conv(x)
        x_length = x_length // self.conv_config.stride
        x = nn.glu(x, axis=-1)
        x = self.do(x, deterministic=deterministic)
        x = x + self.cape(
            x,
            x_lengths=x_length,
            deterministic=deterministic,
            positions_delta=self.positions_delta,
        ).astype(self.dtype)
        for block in self.tf_blocks:
            x = block(x, x_length, deterministic=deterministic)
        if self.block_config.ln_norm_type == "pre":
            x = self.ln_final(x)
        return self.linear(x), x_length
