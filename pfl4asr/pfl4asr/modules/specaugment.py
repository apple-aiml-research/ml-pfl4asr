#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import random

from .functions import length_to_mask, masked_mean2d


# offset and width should be num_masks x batch_size
def _band_mask(size, offset, width):
    num_masks = offset.shape[0]
    batch_size = offset.shape[1]
    offset = rearrange(offset, "n b -> n b 1")
    width = rearrange(width, "n b -> n b 1")
    # num_masks x batch_size x size
    mask = repeat(jnp.arange(size), "t -> n b t", b=batch_size, n=num_masks)
    mask = (mask >= offset) & (mask < (offset + width))
    mask = jnp.max(mask, 0)  # reduce over num_masks
    return mask


@dataclass
class ConfigSpecAugment:
    num_freq_masks: int
    freq_mask_max_width: int
    num_time_masks: int
    time_mask_max_width: int
    time_mask_width_ratio: float
    avg_mask_strategy: bool


class SpecAugment(nn.Module):
    config: ConfigSpecAugment

    # assumes input is NWC
    def __call__(self, input, input_length, deterministic=True):
        if deterministic:
            return input

        batch_size = input.shape[0]
        input_max_length = input.shape[1]
        num_channels = input.shape[2]

        res = input
        rng = self.make_rng("spec_augment")
        rngs = random.split(rng, 4)

        if self.config.avg_mask_strategy:
            fill_value, length_mask = masked_mean2d(
                input, input_length, return_mask=True
            )
            fill_value = fill_value.reshape(-1, 1, 1)
        else:
            fill_value = 0.0
            length_mask = length_to_mask(input_length, input_max_length)
            length_mask = jnp.expand_dims(length_mask, axis=-1)

        # freq mask
        if self.config.freq_mask_max_width >= num_channels:
            raise RuntimeError("Invalid freq_mask_max_width")
        f_widths = random.randint(
            rngs[0],
            (self.config.num_freq_masks, batch_size),
            0,
            self.config.freq_mask_max_width,
        )
        f_offsets = random.randint(
            rngs[1],
            (self.config.num_freq_masks, batch_size),
            0,
            num_channels - f_widths,
        )
        mask = _band_mask(num_channels, f_offsets, f_widths)
        mask = mask.reshape((batch_size, 1, num_channels))
        res = jnp.where(mask, fill_value, res)

        time_mask_width = jnp.minimum(
            self.config.time_mask_max_width,
            input_length * self.config.time_mask_width_ratio,
        ).astype(jnp.int32)
        t_widths = random.randint(
            rngs[2], (self.config.num_time_masks, batch_size), 0, time_mask_width
        )
        t_offsets = random.randint(
            rngs[3],
            (self.config.num_time_masks, batch_size),
            0,
            input_length - t_widths,
        )
        mask = _band_mask(input_max_length, t_offsets, t_widths)
        mask = mask.reshape((batch_size, input_max_length, 1))
        res = jnp.where(mask, fill_value, res)
        res = jnp.where(length_mask, res, 0)

        return res
