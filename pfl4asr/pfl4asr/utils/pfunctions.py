#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from functools import partial

import flax
import jax
import optax
from jax import numpy as jnp

from ..modules.functions import length_to_mask


def create_ctc_loss_from_logits(blank_id: int):
    @jax.jit
    def ctc_loss_from_logits(logits, logits_length, y, y_length):
        logits_mask = length_to_mask(logits_length, logits.shape[1])
        logits_mask = jnp.logical_not(logits_mask).astype(jnp.float32)
        y_mask = length_to_mask(y_length, y.shape[1])
        y_mask = jnp.logical_not(y_mask).astype(jnp.float32)
        existing = y_length > 0
        return jnp.sum(
            existing * optax.ctc_loss(logits, logits_mask, y, y_mask, blank_id=blank_id)
        ) / (jnp.sum(existing) + 1e-10)

    pctc_loss_from_logits = jax.pmap(ctc_loss_from_logits)
    return ctc_loss_from_logits, pctc_loss_from_logits


@jax.jit
def clip_grads(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    leaves, _treedef = jax.tree_util.tree_flatten(grad_tree)
    per_layer = [jnp.vdot(x, x) for x in leaves]
    l2_norm = jnp.sqrt(sum(per_layer))

    def normalize(g):
        return jnp.where(l2_norm < max_norm, g, g * (max_norm / (l2_norm + 1e-6)))

    return jax.tree_util.tree_map(normalize, grad_tree), l2_norm, per_layer


@jax.jit
def clip_layer_grads(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`. Per layer clipping."""
    leaves, _treedef = jax.tree_util.tree_flatten(grad_tree)
    per_layer = [jnp.vdot(x, x) for x in leaves]
    l2_norm = jnp.sqrt(sum(per_layer))
    n_layers = len(per_layer)

    def normalize_equal(g):
        return jnp.where(
            l2_norm < max_norm,
            g,
            g / (jnp.sqrt(jnp.vdot(g, g)) + 1e-10) * (max_norm / (n_layers**0.5)),
        )

    return jax.tree_util.tree_map(normalize_equal, grad_tree), l2_norm, per_layer


@jax.jit
def clip_layer_grads_params(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`. Per layer clipping with dimension accounting."""
    leaves, _treedef = jax.tree_util.tree_flatten(grad_tree)
    per_layer = [jnp.vdot(x, x) for x in leaves]
    l2_norm = jnp.sqrt(sum(per_layer))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(grad_tree))

    def normalize_per_param(g):
        return jnp.where(
            l2_norm < max_norm,
            g,
            g
            / (jnp.sqrt(jnp.vdot(g, g)) + 1e-10)
            * (max_norm * (g.size**0.5) / (n_params**0.5)),
        )

    return jax.tree_util.tree_map(normalize_per_param, grad_tree), l2_norm, per_layer


@jax.jit
def clip_layer_grads_params_w12(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`. Special clipping for FC layer"""
    leaves, _treedef = jax.tree_util.tree_flatten(grad_tree)
    per_layer = [jnp.vdot(x, x) for x in leaves]
    l2_norm = jnp.sqrt(sum(per_layer))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(grad_tree))
    # hack version for specific arch
    n_params_w12 = sum(
        p.size * (len(p.shape) == 2 and (p.shape[1] == 3072 or p.shape[0] == 3072))
        for p in jax.tree_util.tree_leaves(grad_tree)
    )
    n_params = n_params + 9 * n_params_w12

    def normalize_per_param(g):
        mask = len(g.shape) == 2 and (g.shape[1] == 3072 or g.shape[0] == 3072)
        mult = 10 * mask + (1 - mask)
        return jnp.where(
            l2_norm < max_norm,
            g,
            g
            / (jnp.sqrt(jnp.vdot(g, g)) + 1e-10)
            * (max_norm * ((g.size * mult) ** 0.5) / (n_params**0.5)),
        )

    return jax.tree_util.tree_map(normalize_per_param, grad_tree), l2_norm, per_layer


@jax.jit
def clip_layer_grads_params_power(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    leaves, _treedef = jax.tree_util.tree_flatten(grad_tree)
    per_layer = [jnp.vdot(x, x) for x in leaves]
    l2_norm = jnp.sqrt(sum(per_layer))
    n_params = sum(p.size**1 / 3.0 for p in jax.tree_util.tree_leaves(grad_tree))

    def normalize_per_param(g):
        return jnp.where(
            l2_norm < max_norm,
            g,
            g
            / (jnp.sqrt(jnp.vdot(g, g)) + 1e-10)
            * (max_norm * (((g.size**1 / 3.0) / n_params) ** 0.5)),
        )

    return jax.tree_util.tree_map(normalize_per_param, grad_tree), l2_norm, per_layer


@jax.jit
def sum_tree_func(tree1, tree2):
    return jax.tree_util.tree_map(lambda p, q: (p + q), tree1, tree2)


@jax.jit
def minus_tree_func(tree1, tree2):
    return jax.tree_util.tree_map(lambda p, q: (p - q), tree1, tree2)


@partial(jax.jit, static_argnums=[1])
def val_div_tree_func(tree, val: float):
    return jax.tree_util.tree_map(lambda p: p / val, tree)


@jax.jit
def copy_tree(tree):
    return jax.tree_util.tree_map(lambda x: x, tree)


@jax.jit
def norm_tree_func(tree):
    def normalize(x):
        return jnp.sqrt(jnp.sum(x * x))

    return jax.tree_util.tree_map(normalize, tree)


@jax.jit
def l2_tree_func(tree):
    leaves, _ = jax.tree_util.tree_flatten(tree)
    l2_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
    return l2_norm


def flatten(p, label=None):
    if isinstance(p, flax.core.frozen_dict.FrozenDict):
        for k, v in p.items():
            yield from flatten(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, p)


@partial(jax.jit, static_argnums=[2])
def gauss_noise(tree, rng, sigma: float):
    """
    Add Gaussian noise to the tree - used for DP
    """
    tree_str = jax.tree_util.tree_structure(tree)
    rngs = jax.random.split(rng, tree_str.num_leaves)
    rngs_tree = jax.tree_util.tree_unflatten(tree_str, rngs)
    noise = jax.tree_util.tree_map(
        lambda p, rng: p + jax.random.normal(rng, p.shape) * sigma, tree, rngs_tree
    )
    leaves, _ = jax.tree_util.tree_flatten(noise)
    l2_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

    return noise, l2_norm


@jax.jit
def zero_params(tree):
    leaves, _ = jax.tree_util.tree_flatten(tree)
    zeros = sum((x == 0).sum() for x in leaves)
    num_params = sum(x.size for x in leaves)
    return zeros * 100.0 / num_params


@jax.jit
def reduce_clients(n_clients):
    return jax.lax.psum(n_clients, axis_name="xl")


# pmap all functions to create distributed counterparts
pclip_grads = jax.pmap(clip_grads, "xl")
pclip_layer_grads = jax.pmap(clip_layer_grads, "xl")
pclip_layer_grads_params = jax.pmap(clip_layer_grads_params, "xl")
pclip_layer_grads_params_w12 = jax.pmap(clip_layer_grads_params_w12, "xl")
pclip_layer_grads_params_power = jax.pmap(clip_layer_grads_params_power, "xl")
pl2_tree_func = jax.pmap(l2_tree_func, "xl")
pnorm_tree_func = jax.pmap(norm_tree_func, "xl")
psum_tree_func = jax.pmap(sum_tree_func, "xl")
pminus_tree_func = jax.pmap(minus_tree_func, "xl")
pval_div_tree_func = jax.pmap(val_div_tree_func, "xl", static_broadcasted_argnums=[1])
pcopy_tree = jax.pmap(copy_tree, "xl")
pgauss_noise = jax.pmap(gauss_noise, "xl", static_broadcasted_argnums=[2])
pzero_params = jax.pmap(zero_params, "xl")
preduce_clients = jax.pmap(reduce_clients, "xl")
