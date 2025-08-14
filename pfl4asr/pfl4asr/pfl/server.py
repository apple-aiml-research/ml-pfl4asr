#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional

import flax
import jax
import jax.numpy as jnp
import optax
from jax._src.typing import Array
from loguru import logger

from ..utils.pfunctions import (
    pclip_grads,
    pcopy_tree,
    pgauss_noise,
    pl2_tree_func,
    psum_tree_func,
    pval_div_tree_func,
)


def generate_optim_step(optim):
    @jax.jit
    def update(deltas, optim_state, params):
        effective_deltas, optim_state = optim.update(deltas, optim_state, params)
        params = optax.apply_updates(params, effective_deltas)
        return params, optim_state

    pupdate = jax.pmap(update, "xl")
    return pupdate


@jax.jit
def all_reduce(data):
    return jax.lax.pmean(data, "xl")


pall_reduce = jax.pmap(all_reduce, "xl")


class Server:
    """Basic version of FL server that includes aggregation of deltas from client
    and server optimization step, also adding the differential private noise"""

    def __init__(
        self,
        params,
        model_state,  # non-learned parameters
        optim,
        clip_cohort_deltas: Optional[float] = None,
        normalize_cohort_deltas: bool = False,
        dp_sigma: Optional[float] = None,
        num_local_devices: int = 8,
    ):
        logger.info("Server PFL: init()")
        self.params = params
        self.optim = optim
        self.model_state = model_state
        self.optim_state = optim.init(params)
        self.params = flax.jax_utils.replicate(self.params)
        self.model_state = flax.jax_utils.replicate(self.model_state)
        self.optim_state = flax.jax_utils.replicate(self.optim_state)
        self.update_step_func = generate_optim_step(self.optim)
        self.clip_cohort_deltas = clip_cohort_deltas
        self.normalize_cohort_deltas = normalize_cohort_deltas
        self.dp_sigma = dp_sigma
        self.num_local_devices = num_local_devices

    def copy_current_params(self):
        logger.info("Server PFL: copy_current_params()")
        return pcopy_tree(self.params)

    def current_params(self):
        return self.params

    def current_model_state(self):
        return self.model_state

    def reset_deltas(self):
        logger.info("Server PFL: reset_deltas()")
        self.deltas = None
        self.num_deltas = 0

    def add_deltas(self, deltas):
        logger.info("Server PFL: add_deltas()")
        if self.deltas is None:
            self.deltas = deltas
            self.num_deltas = 1
        else:
            self.deltas = psum_tree_func(self.deltas, deltas)
            self.num_deltas += 1

    def step(self, rng_noise: Array) -> tuple[Array, Array, Array]:
        logger.info("Server PFL: step()")
        logger.info(f"learning_rate: {self.optim_state.hyperparams['learning_rate']}")
        logger.info(f"num_deltas: {self.num_deltas}")

        self.deltas = pval_div_tree_func(self.deltas, self.num_deltas)
        logger.info("Delta normalized per process")
        # sync across processes
        self.deltas = pall_reduce(self.deltas)
        logger.info("Deltas all reduce")
        l2_norm = pl2_tree_func(self.deltas)
        jax.debug.print(
            "Server PFL deltas norm: {l2_norm} {shape}",
            l2_norm=float(l2_norm[0]),
            shape=l2_norm.shape,
        )
        if self.dp_sigma is not None:
            rng_noise, prng_noise = jax.random.split(rng_noise, 2)
            self.deltas, l2_norm_noise = pgauss_noise(
                self.deltas,
                jnp.array([prng_noise] * self.num_local_devices),
                self.dp_sigma,
            )
            jax.debug.print(
                "server deltas norm after noise: {l2_norm}",
                l2_norm=float(l2_norm_noise[0]),
            )
        else:
            l2_norm_noise = l2_norm
        if self.clip_cohort_deltas is not None:
            self.deltas, l2_norm, _ = pclip_grads(
                self.deltas, flax.jax_utils.replicate(self.clip_cohort_deltas)
            )
            jax.debug.print(
                "server deltas norm before clipping: {l2_norm}",
                l2_norm=float(l2_norm[0]),
            )
        if self.normalize_cohort_deltas:
            logger.info("Normalizing deltas by norm")
            self.deltas = pval_div_tree_func(self.deltas, float(l2_norm[0]))
        self.params, self.optim_state = self.update_step_func(
            self.deltas, self.optim_state, self.params
        )
        return l2_norm, l2_norm_noise, rng_noise

    def get_learning_rate(self) -> float:
        return self.optim_state.hyperparams["learning_rate"][0]

    def restore_from_checkpoint(self, params=None, optim_state=None):
        if params is not None:
            self.params = params
            self.params = flax.jax_utils.replicate(self.params)
        if optim_state is not None:
            self.optim_state = optim_state
            self.optim_state = flax.jax_utils.replicate(self.optim_state)
