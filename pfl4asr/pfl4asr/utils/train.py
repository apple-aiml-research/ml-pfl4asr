#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from loguru import logger

from .pfunctions import clip_grads, minus_tree_func


def create_step_and_eval_funcs(
    model,
    specaug,
    optimizer,
    preprocess_function,
    ctc_loss_from_logits: Callable,
    model_rng_keys,
):
    @partial(jax.jit, static_argnums=[5, 6])
    def eval_step(params, model_state, x, x_length, rng, use_saug, use_dropout):
        rngs = jax.random.split(rng, len(model_rng_keys) + 1)
        x, x_length = preprocess_function(x, x_length)
        x = specaug.apply(
            {},
            x,
            x_length,
            deterministic=(not use_saug),
            rngs={"spec_augment": rngs[0]},
        )
        (logits, x_length), updated_model_state = model.apply(
            {"params": params, **model_state},
            x,
            x_length,
            deterministic=(not use_dropout),
            mutable=list(model_state.keys()),
            rngs={key: rngs[index + 1] for index, key in enumerate(model_rng_keys)},
        )
        return (logits, x_length), updated_model_state

    @partial(jax.jit, static_argnums=[8, 9, 10])
    def loss(
        params,
        model_state,
        x,
        x_length,
        y,
        y_length,
        rng,
        fedprox_params,
        fedprox_mu,
        use_saug,
        use_dropout,
    ):
        (logits, logits_length), updated_model_state = eval_step(
            params, model_state, x, x_length, rng, use_saug, use_dropout
        )
        logits = logits.astype(jnp.float32)
        ctc_loss = ctc_loss_from_logits(logits, logits_length, y, y_length)

        if fedprox_mu and fedprox_mu > 0:
            fedprox_param_diff = minus_tree_func(params, fedprox_params)
            fedprox_param_diff_leaves, _ = jax.tree_util.tree_flatten(
                fedprox_param_diff
            )
            fedprox_norm = jnp.sqrt(
                sum(jnp.vdot(x, x) for x in fedprox_param_diff_leaves)
            )
            jax.debug.print(
                "fedprox_norm: {fedprox_norm}\nfedprox_mu: {fedprox_mu}",
                fedprox_norm=(fedprox_norm.astype(float)),
                fedprox_mu=fedprox_mu,
            )
            total_loss = ctc_loss + fedprox_mu * fedprox_norm
        else:
            total_loss = ctc_loss

        return (
            total_loss,
            updated_model_state,
        )

    @partial(jax.jit, static_argnums=[9, 10, 11, 12, 13])
    def train_step(
        params,
        model_state,
        optim_state,
        x,
        x_length,
        y,
        y_mask,
        rng,
        fedprox_params,
        fedprox_mu,
        use_saug,
        use_dropout,
        grad_clip,
        all_reduce,
    ):
        (loss_value, updated_model_state), grads = jax.value_and_grad(
            loss, has_aux=True
        )(
            params,
            model_state,
            x,
            x_length,
            y,
            y_mask,
            rng,
            fedprox_params,
            fedprox_mu,
            use_saug,
            use_dropout,
        )
        if all_reduce:
            loss_value = jax.lax.pmean(loss_value, "xl")
            grads = jax.lax.pmean(grads, "xl")
        grads, grads_l2_norm, per_layer = clip_grads(
            grads, -1 if grad_clip is None else grad_clip
        )
        updates, optim_state = optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return (
            params,
            updated_model_state,
            optim_state,
            loss_value,
            grads_l2_norm,
            per_layer,
        )

    peval_step = jax.pmap(eval_step, "xl", static_broadcasted_argnums=[5, 6])
    ptrain_step = jax.pmap(
        train_step, "xl", static_broadcasted_argnums=[9, 10, 11, 12, 13]
    )
    return peval_step, ptrain_step


# Eval / Logging
# ------------


def print_log(log: dict):
    txt = []
    for key, value in log.items():
        txt.append(key + ":")
        txt.append(str(value))
    logger.info("\t".join(txt))
