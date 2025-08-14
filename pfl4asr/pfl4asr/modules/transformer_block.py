#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.nn.initializers as initjax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import linen as nn
from jax import random

Array = Any


_init_dense_function = initjax.variance_scaling(
    1.0 / 3.0,
    "fan_in",
    "uniform",
    in_axis=-2,
    out_axis=-1,
    batch_axis=(),
    dtype=jnp.float_,
)


@dataclass
class ConfigTransformerBlock:
    emb_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 4
    dropout: float = 0.3
    layer_dropout: float = 0.3
    ln_norm_type: str = "post"  # pre, post
    kernel_init: Callable = _init_dense_function
    bias: bool = False
    bias_init: Callable = initjax.zeros
    ln_epsilon: float = 1e-05
    dtype: Optional[Any] = None


class TransformerBlock(nn.Module):
    config: ConfigTransformerBlock

    def setup(self):
        # normalizations
        self.ln1 = nn.LayerNorm(
            epsilon=self.config.ln_epsilon,
            dtype=self.config.dtype,
        )
        self.ln2 = nn.LayerNorm(
            epsilon=self.config.ln_epsilon,
            dtype=self.config.dtype,
        )

        # self attention
        self.wqkv = nn.Dense(
            features=self.config.emb_dim * 3,
            use_bias=self.config.bias,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            dtype=self.config.dtype,
        )
        self.wf = nn.Dense(
            features=self.config.emb_dim,
            use_bias=self.config.bias,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            dtype=self.config.dtype,
        )
        # mlp block
        self.w1 = nn.Dense(
            features=self.config.mlp_dim,
            use_bias=self.config.bias,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            dtype=self.config.dtype,
        )
        self.w2 = nn.Dense(
            features=self.config.emb_dim,
            use_bias=self.config.bias,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            dtype=self.config.dtype,
        )
        self.do = nn.Dropout(self.config.dropout)

    def _selfAttention(
        self, x: Array, x_length: Array, deterministic: Optional[bool] = None, rng=None
    ):
        q, k, v = rearrange(
            self.wqkv(x),
            "b t (qkv h hc) -> qkv b t h hc",
            qkv=3,
            h=self.config.num_heads,
            hc=self.config.emb_dim // self.config.num_heads,
        )
        b, t, h, _hc = k.shape
        padding_mask = repeat(jnp.arange(t), "tk -> b h tq tk", b=b, h=h, tq=t)
        padding_mask = padding_mask < rearrange(x_length, "b -> b 1 1 1")

        result = nn.dot_product_attention(
            q,
            k,
            v,
            mask=padding_mask,
            dropout_rng=rng,
            broadcast_dropout=False,
            dropout_rate=self.config.dropout,
            deterministic=deterministic,
        )
        result = rearrange(result, "b t h hc -> b t (h hc)")
        return self.wf(result)

    def _mlp(self, x: Array, deterministic: Optional[bool] = None):
        x = self.w1(x)
        x = nn.relu(x)
        x = self.do(x, deterministic=deterministic)
        x = self.w2(x)

        return x

    def __call__(self, x: Array, x_length: Array, deterministic: Optional[bool] = None):
        rng = self.make_rng("speech_tr_block")
        rngs_key = random.split(rng, 2)
        do_layer_drop = jnp.zeros(shape=(1,))
        if not deterministic:
            do_layer_drop = (
                random.uniform(rngs_key[0], shape=(1,)) < self.config.layer_dropout
            )
        if self.config.ln_norm_type == "post":
            x = self.ln1(
                (1 - do_layer_drop)
                * self._selfAttention(
                    x, x_length, deterministic=deterministic, rng=rngs_key[1]
                )
                + x
            )
            return self.ln2(
                (1 - do_layer_drop) * self._mlp(x, deterministic=deterministic) + x
            )
        elif self.config.ln_norm_type == "pre":
            x = (1 - do_layer_drop) * self._selfAttention(
                self.ln1(x), x_length, deterministic=deterministic, rng=rngs_key[1]
            ) + x
            return (1 - do_layer_drop) * self._mlp(
                self.ln2(x), deterministic=deterministic
            ) + x
        else:
            raise RuntimeError(f"Unknown layer norm type {self.config.ln_norm_type}")
