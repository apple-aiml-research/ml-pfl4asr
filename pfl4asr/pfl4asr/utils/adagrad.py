#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Union

import jax
import jax.numpy as jnp
from optax._src import alias, base, combine, transform

ScalarOrSchedule = Union[float, base.Schedule]


def scale_by_rss_my(
    initial_accumulator_value: float = 0.1, eps: float = 1e-8
) -> base.GradientTransformation:
    """Rescale updates by the root of the sum of all squared gradients to date.
    Difference with optax version in the epsilon usage
    """

    def init_fn(params):
        sum_of_squares = jax.tree_map(
            lambda t: jnp.full_like(t, initial_accumulator_value), params
        )
        return transform.ScaleByRssState(sum_of_squares=sum_of_squares)

    def update_fn(updates, state, params=None):
        del params
        sum_of_squares = jax.tree_map(
            lambda g, t: transform._abs_sq(g) + t, updates, state.sum_of_squares
        )
        inv_sqrt_g_square = jax.tree_map(
            lambda t: jnp.where(t > 0, jax.lax.rsqrt(t) + eps, 0.0), sum_of_squares
        )
        updates = jax.tree_map(lambda scale, g: scale * g, inv_sqrt_g_square, updates)
        return updates, transform.ScaleByRssState(sum_of_squares=sum_of_squares)

    return base.GradientTransformation(init_fn, update_fn)


def adagrad(
    learning_rate: ScalarOrSchedule,
    initial_accumulator_value: float = 0.0,
    eps: float = 1e-8,
) -> base.GradientTransformation:
    """The Adagrad optimizer. Modified version compared to optax.Adagrad - difference in eps placemenet"""
    return combine.chain(
        scale_by_rss_my(initial_accumulator_value=initial_accumulator_value, eps=eps),
        alias._scale_by_learning_rate(learning_rate),
    )
