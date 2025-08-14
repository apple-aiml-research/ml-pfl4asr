#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import jax
import jax.numpy as jnp
from einops import rearrange


def length_to_mask(lengths, max_length: int):
    indices = rearrange(jnp.arange(max_length), "t -> 1 t")
    return indices < rearrange(lengths, "b -> b 1")


# x: NxWxC length: N
def masked_mean2d(x, x_length, return_mask=False):
    x_mask = jnp.expand_dims(length_to_mask(x_length, x.shape[1]), axis=-1)
    C = x.shape[2]
    scale = 1 / (C * jnp.maximum(1, x_length))
    scale = scale.astype(x.dtype)
    mean = (x * x_mask).sum(axis=(1, 2)) * scale
    if return_mask:
        return mean, x_mask
    else:
        return mean


# x: NxWxC length: N
def length_masked_normalize2d(x, x_length, epsilon=1e-7, return_mask=False):
    x_mask = jnp.expand_dims(length_to_mask(x_length, x.shape[1]), axis=-1)
    C = x.shape[2]
    scale = 1 / (C * jnp.maximum(1, x_length))
    scale = scale.astype(x.dtype)
    mean = (x * x_mask).sum(axis=(1, 2)) * scale
    x = x - rearrange(mean, "b -> b 1 1")
    # beware: sqrt grad 0 is inf...
    norm = jax.lax.rsqrt((x * x * x_mask).sum(axis=(1, 2)) * scale + epsilon)
    x = jnp.where(x_mask, x * rearrange(norm, "b -> b 1 1"), 0)
    if return_mask:
        return x, x_mask
    else:
        return x
