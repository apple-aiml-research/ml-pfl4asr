#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import optax

from ..utils import adagrad


@dataclass
class ConfigServerOptimization:
    optim: str = "adagrad"
    # Optimizer b1
    optim_b1: float = 0.9
    # Optimizer b2
    optim_b2: float = 0.999
    # Optimizer eps
    optim_eps: float = 1e-6  # should be 0 for lars
    # Optimizer weight decay
    weight_decay: float = 0.0
    # A multiplier for the trust ratio for lars optimizer. Only used if lars is used.
    trust_coefficient: float = 0.001
    # Decay rate for momentum for lars optimizer. Only used if lars is used.
    optim_momentum: float = 0.9
    lr: float = 0.03
    # server lr schedule decay: warmup_step, warmup_exp, step, exp, const
    lr_schedule: str = "warmup_step"
    lr_decay_start_update: int = 330000
    lr_decay_updates: int = 50000
    lr_step_decay_factor: float = 2.0
    lr_exponential_decay_transition_steps: int = 25000
    # Exponential decay rate for the learning rate (all but adagrad which uses legacy LR schedule).
    lr_exponential_decay_rate: float = 1.0
    # End value for the server decay schedule (lr won't go below this value).
    lr_end_value: Optional[float] = None
    warmup: int = 64000
    # Clipping of the aggregated deltas on server
    max_grad_norm: float = 1.0


@dataclass
class ConfigLocalOptimization:
    optim: str = "sgd"
    lr: float = 0.1
    # Clipping during local optimization for each client
    max_grad_norm: float = 1.0
    # number of epochs to train on a client
    num_epochs: int = 1
    # number of steps to train on a client: if -1 then use epochs
    num_steps: int = -1


@optax.inject_hyperparams
def my_adagrad(learning_rate, config: ConfigServerOptimization):
    return adagrad(learning_rate)


@optax.inject_hyperparams
def my_sgd(learning_rate, config: ConfigServerOptimization | ConfigLocalOptimization):
    return optax.sgd(learning_rate)


@optax.inject_hyperparams
def my_lamb(learning_rate, config: ConfigServerOptimization):
    return optax.lamb(
        learning_rate,
        b1=config.optim_b1,
        b2=config.optim_b2,
        weight_decay=config.weight_decay,
        eps=config.optim_eps,
    )


@optax.inject_hyperparams
def my_adam(learning_rate, config: ConfigServerOptimization):
    return optax.adam(learning_rate, b1=config.optim_b1, b2=config.optim_b2)


@optax.inject_hyperparams
def my_adamw(learning_rate, config: ConfigServerOptimization):
    return optax.adamw(learning_rate, b1=config.optim_b1, b2=config.optim_b2)


@optax.inject_hyperparams
def my_yogi(learning_rate, config: ConfigServerOptimization):
    return optax.yogi(learning_rate, b1=config.optim_b1, b2=config.optim_b2)


@optax.inject_hyperparams
def my_lars(learning_rate, config: ConfigServerOptimization):
    return optax.lars(
        learning_rate,
        weight_decay=config.weight_decay,
        trust_coefficient=config.trust_coefficient,
        eps=config.optim_eps,
        momentum=config.optim_momentum,
    )


@optax.inject_hyperparams
def my_shampoo(learning_rate, config: ConfigServerOptimization):
    from optax_shampoo import distributed_shampoo

    return distributed_shampoo.distributed_shampoo(
        learning_rate=learning_rate,
        block_size=32,
    )


def get_optimizer(
    config: ConfigServerOptimization | ConfigLocalOptimization, lr_schedule=None
):
    lr = config.lr if lr_schedule is None else lr_schedule
    if config.optim == "sgd":
        return my_sgd(learning_rate=lr, config=config)
    elif config.optim == "adagrad":
        return my_adagrad(learning_rate=lr, config=config)
    elif config.optim == "lamb":
        return my_lamb(learning_rate=lr, config=config)
    elif config.optim == "lars":
        return my_lars(learning_rate=lr, config=config)
    elif config.optim == "adam":
        return my_adam(learning_rate=lr, config=config)
    elif config.optim == "adamw":
        return my_adamw(learning_rate=lr, config=config)
    elif config.optim == "yogi":
        return my_yogi(learning_rate=lr, config=config)
    elif config.optim == "shampoo":
        return my_shampoo(learning_rate=lr, config=config)
    else:
        RuntimeError("Undefined local optimizer!")
        exit(-1)


def create_step_decay_func(
    start_lr, decay_updates, transition_begin=-1, decay_factor=2, end_value=None
):
    def schedule(updates):
        if end_value is not None:
            sched = jnp.where(
                updates <= transition_begin,
                start_lr,
                start_lr
                / jnp.power(
                    decay_factor, abs(updates - transition_begin) // decay_updates + 1
                ),
            )
            return jnp.where(sched < end_value, end_value, sched)
        else:
            return jnp.where(
                updates <= transition_begin,
                start_lr,
                start_lr
                / jnp.power(
                    decay_factor, abs(updates - transition_begin) // decay_updates + 1
                ),
            )

    return schedule


def get_lr_schedule(config: ConfigServerOptimization) -> optax.Schedule:
    warmup_constant_lr_schedule = optax.linear_schedule(
        init_value=0,
        end_value=config.lr,
        transition_steps=config.warmup,
        transition_begin=0,
    )
    decay_lr_schedule = create_step_decay_func(
        config.lr,
        decay_updates=config.lr_decay_updates,
    )

    lr_schedule_warmup_linear_decay = optax.join_schedules(
        [warmup_constant_lr_schedule, decay_lr_schedule],
        [config.lr_decay_start_update],
    )
    # ---------

    lr_schedule_exp_decay = optax.exponential_decay(
        config.lr,
        transition_steps=config.lr_exponential_decay_transition_steps,
        transition_begin=config.lr_decay_start_update,
        decay_rate=config.lr_exponential_decay_rate,
        end_value=config.lr_end_value,
    )

    # ---------

    decay_lr_schedule = create_step_decay_func(
        config.lr,
        decay_updates=config.lr_decay_updates,
        end_value=config.lr_end_value,
    )

    lr_schedule_warmup_exp_decay = optax.warmup_exponential_decay_schedule(
        init_value=1e-10,
        peak_value=config.lr,
        warmup_steps=config.warmup,
        transition_steps=config.lr_exponential_decay_transition_steps,
        decay_rate=config.lr_exponential_decay_rate,
        transition_begin=config.lr_decay_start_update,
        end_value=config.lr_end_value,
    )

    # ---------

    server_lr_schedule = {
        "warmup_step": lr_schedule_warmup_linear_decay,
        "warmup_exp": lr_schedule_warmup_exp_decay,
        "exp": lr_schedule_exp_decay,
        "step": create_step_decay_func(
            config.lr,
            decay_updates=config.lr_decay_updates,
            transition_begin=config.lr_decay_start_update,
            decay_factor=config.lr_step_decay_factor,
            end_value=config.lr_end_value,
        ),
        "constant": config.lr,
    }
    return server_lr_schedule.get(config.lr_schedule, None)
