#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import csv
import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
from loguru import logger
from tensorboardX import SummaryWriter

from pfl4asr.common import ConfigTrainingShared
from pfl4asr.modules.transformer_block import ConfigTransformerBlock
from pfl4asr.utils.counter import CounterValue
from pfl4asr.utils.data import (
    BLANK_TOKEN,
    construct_eng_char_trie_for_ctc,
    create_dataset,
    create_mfsc_func,
    get_next_sample_on_device,
)

try:
    from pfl4asr_internal.metrics import send_metrics
except:

    def send_metrics(log: dict, current_step: int, rank: int = 0, world_size: int = 1):
        pass


from pfl4asr.utils.distributed import init_distributed
from pfl4asr.utils.eval import compute_wer_ter_loss
from pfl4asr.utils.model import (
    create_model,
    init_model,
    load_model,
    load_state,
    load_other_state,
    save_model,
    save_state,
    save_other_state,
)
from pfl4asr.utils.optimizer import (
    ConfigServerOptimization,
    get_lr_schedule,
    get_optimizer,
)
from pfl4asr.utils.pfunctions import (
    create_ctc_loss_from_logits,
    flatten,
    pnorm_tree_func,
    psum_tree_func,
)
from pfl4asr.utils.train import create_step_and_eval_funcs, print_log


@dataclass
class ConfigCentralTraining:
    train: str = ""
    epochs: int = 10000000
    # Batch size given in seconds per 1 device
    batch_size: int = 300


if __name__ == "__main__":
    from simple_parsing import ArgumentGenerationMode, ArgumentParser

    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.NESTED,
        add_config_path_arg=True,
        config_path="configs/default_central.yaml",
    )
    parser.add_arguments(ConfigTrainingShared, dest="shared_config")
    parser.add_arguments(ConfigTransformerBlock, dest="block_config")
    parser.add_arguments(ConfigCentralTraining, dest="central_config")
    parser.add_arguments(ConfigServerOptimization, dest="server_optim_config")

    parser.set_defaults(
        central_config=ConfigCentralTraining(),
        shared_config=ConfigTrainingShared(),
        block_config=ConfigTransformerBlock(),
        server_optim_config=ConfigServerOptimization(optim_eps=1e-8),
    )
    all_args = parser.parse_args()
    central_config: ConfigCentralTraining = all_args.central_config
    shared_config: ConfigTrainingShared = all_args.shared_config
    block_config: ConfigTransformerBlock = all_args.block_config
    server_optim_config: ConfigServerOptimization = all_args.server_optim_config

    WORLD_SIZE, RANK = init_distributed(
        host_ip_address=shared_config.host_ip_address,
        distributed_port=shared_config.distributed_port,
        rank=shared_config.rank,
        world_size=shared_config.world_size,
    )
    num_local_devices = jax.local_device_count()
    if num_local_devices != 1:
        raise RuntimeError(
            "Please use one process per device -- this is more efficient"
        )
    local_devices = jax.local_devices()
    runname = Path(os.getenv("ARTIFACT_PREFIX", "")).joinpath(shared_config.runname)
    Path.mkdir(runname.parent, exist_ok=True, parents=True)
    logger.info(f"({RANK=}) Args: {all_args}")
    logger.info(f"({RANK=}) Devices: {num_local_devices}")
    logger.info(f"({RANK=}) Construct trie for CTC training with letters")
    # trie construction
    trie = construct_eng_char_trie_for_ctc(shared_config.additional_chars)
    BLANK_INDEX = trie.search(BLANK_TOKEN).id
    logger.info(f"({RANK=}) nlabel: {trie.num_keys()}")

    # model creation
    specaug, model = create_model(
        shared_config.use_cape,
        block_config,
        trie,
        input_mel=shared_config.num_mels,
        n_blocks=shared_config.n_blocks,
    )

    model_rng_keys = ["params", "speech_tr_block", "dropout", "cape1d"]
    params_specaug = init_model(
        specaug, ["spec_augment"], 2, shared_config.num_mels, shared_config.seed
    )
    variables = init_model(
        model, model_rng_keys, 2, shared_config.num_mels, shared_config.seed
    )
    if len(variables.keys()) == 1:
        model_state, params = dict(), variables.pop("params")
    else:
        model_state, params = variables.pop("params")
    del variables
    nparams = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logger.info(
        f"({RANK=}) Model arch {jax.tree_util.tree_map(lambda x: x.shape, params)}"
    )

    logger.info(
        f"({RANK=}) Model state {jax.tree_util.tree_map(lambda x: x.shape, model_state)}"
    )
    logger.info(f"({RANK=}) Model with {nparams=}")

    if (
        shared_config.restore_checkpoint is not None
        and not runname.with_suffix(".bin").exists()
    ):
        # use pretrained seed model when model doesn't exist yet
        restore_model_path = (
            Path(os.getenv("ARTIFACT_PREFIX"))
            .joinpath(shared_config.restore_checkpoint)
            .with_suffix(".bin")
        )
        current_step = 0
        logger.info(
            f"({RANK=}) Restoring model from the seed model {restore_model_path}"
        )
    else:
        # recover current model
        restore_model_path = runname.with_suffix(".bin")
        current_step = load_other_state(restore_model_path.with_suffix(".json")).get(
            "current_step", 0
        )
        logger.info(
            f"({RANK=}) Restoring model from previous checkpoint {restore_model_path}"
        )

    params = load_model(params, restore_model_path)
    logger.info(f"({RANK=}) Model is restored from {restore_model_path}")
    logger.info(f"({RANK=}) Initial step is {current_step}")

    # Optimizer
    logger.info(f"({RANK=}) Create optimizer")
    lr_schedule = get_lr_schedule(server_optim_config)
    optimizer = get_optimizer(server_optim_config, lr_schedule)
    optim_state = optimizer.init(params)
    if shared_config.is_restore_optim and runname.with_suffix(".bin").exists():
        logger.info(
            f"({RANK=}) Restoring optimizer from {runname.with_suffix('.optim.bin')}"
        )
        optim_state = load_state(runname.with_suffix(".optim.bin"))
        if optim_state.count != current_step:
            raise RuntimeError("Optimizer is incorrectly loaded")
        logger.info(
            f"({RANK=}) Optimizer is restored from {runname.with_suffix('.optim.bin')}"
        )

    # multi-gpu
    ctc_loss_from_logits, pctc_loss_from_logits = create_ctc_loss_from_logits(
        BLANK_INDEX
    )
    mfsc_func = create_mfsc_func(
        num_mels=shared_config.num_mels,
        sample_rate_hz=shared_config.sample_rate_hz,
        frame_size_ms=shared_config.frame_size_ms,
        frame_stride_ms=shared_config.frame_stride_ms,
    )
    peval_step, ptrain_step = create_step_and_eval_funcs(
        model, specaug, optimizer, mfsc_func, ctc_loss_from_logits, model_rng_keys
    )
    optim_state = flax.jax_utils.replicate(optim_state)

    # Logging
    f_perf = None
    tb_writer = None
    if RANK % 8 == 0:
        f_perf = open(runname.with_suffix(f".{current_step}.csv"), "w")
        tb_writer = SummaryWriter(logdir=runname.parent)
    perf = None

    num_steps = 0
    total_loss = CounterValue()
    grad_norm = CounterValue()
    t0 = time.perf_counter()
    t_data = 0
    rng = jax.random.PRNGKey(shared_config.seed)
    total_eval_time = 0.0
    total_num_evals = 0
    total_train_time = 0.0

    logger.info(f"({RANK=}) Starting central training")
    # multi-gpu
    params = flax.jax_utils.replicate(params)
    model_state = flax.jax_utils.replicate(model_state)
    loss_contains_nan = False  # Keep track to not save model if nan found

    dataset = create_dataset(
        os.path.join(shared_config.lists_dir, central_config.train),
        trie=trie,
        batch_size_s=central_config.batch_size,
        max_target_length=shared_config.max_target_len,
        batching_buffer_size=shared_config.batching_buffer_size,
        n_threads=shared_config.threads,
        input_key=shared_config.input_key,
        target_key=shared_config.target_key,
        target_pad=(not shared_config.target_nopad),
        tar=shared_config.tar,
        sample_rate_hz=shared_config.sample_rate_hz,
        world_size=WORLD_SIZE,
        rank=RANK,
        loop=central_config.epochs,
    )
    _, next_sample = get_next_sample_on_device(dataset, trie, local_devices)
    while next_sample is not None:
        sample = next_sample
        t_data_tmp = time.perf_counter()
        _, next_sample = get_next_sample_on_device(dataset, trie, local_devices)
        t_data += time.perf_counter() - t_data_tmp
        x, y, x_length, y_length = sample
        rng, prng = jax.random.split(rng, 2)
        prng = jax.random.split(prng, num_local_devices)

        use_specaug = True if current_step > shared_config.start_saug else False
        (
            params,
            model_state,
            optim_state,
            loss_value,
            grads_l2_norm,
            per_layer_norms,
        ) = ptrain_step(
            params,
            model_state,
            optim_state,
            x,
            x_length,
            y,
            y_length,
            prng,
            None,  # fedprox spaceholder
            None,  # fedprox spaceholder
            use_specaug,
            True,
            server_optim_config.max_grad_norm,
            True,  # all reduce
        )
        if jnp.isnan(loss_value.sum()):
            logger.info(f"({RANK=}) BUG: Loss contains NaN")
            loss_contains_nan = True

        total_loss.add(float(loss_value[0]))
        grad_norm.add(float(grads_l2_norm[0]))
        num_steps += 1
        current_step += 1

        if len(model_state.keys()) > 0:
            state_tree_sum = (
                pnorm_tree_func(model_state)
                if num_steps == 1
                else psum_tree_func(state_tree_sum, pnorm_tree_func(model_state))
            )
        else:
            state_tree_sum = None

        if current_step % shared_config.report == 0:
            t = time.perf_counter() - t0
            log = {
                "step": current_step,
                "duration": str(datetime.timedelta(seconds=t)),
                "ms/step": t / num_steps * 1000.0,
                "ms/step data loader": t_data / num_steps * 1000.0,
                "loss": total_loss.get_average(),
                "total_grad_norm": grad_norm.get_average(),
            }
            if state_tree_sum is not None:
                for key, val in dict(flatten(state_tree_sum, None)).items():
                    log["state(" + key + ")"] = float(val[0]) / num_steps

            eval_paths = (
                []
                if shared_config.eval is None or shared_config.eval == ""
                else shared_config.eval.split(",")
            )
            for eval_name in eval_paths:
                dataset_eval = create_dataset(
                    name=os.path.join(shared_config.lists_dir, eval_name),
                    trie=trie,
                    batch_size_s=shared_config.valid_batch_size,
                    max_target_length=shared_config.max_target_len,
                    batching_buffer_size=shared_config.batching_buffer_size,
                    n_threads=shared_config.threads,
                    input_key=shared_config.input_key,
                    target_key=shared_config.target_key,
                    target_pad=(not shared_config.target_nopad),
                    tar=shared_config.tar,
                )
                (
                    log[eval_name + "-WER"],
                    log[eval_name + "-TER"],
                    log[eval_name + "-loss"],
                ) = compute_wer_ter_loss(
                    peval_step,
                    dataset_eval,
                    trie=trie,
                    params=params,
                    model_state=model_state,
                    pctc_loss_from_logits=pctc_loss_from_logits,
                    rng=rng,
                    local_devices=local_devices,
                    n_threads=shared_config.threads,
                )

            log["lr"] = float(optim_state.hyperparams["learning_rate"][0])
            send_metrics(
                log, current_step=current_step, rank=RANK, world_size=WORLD_SIZE
            )
            print_log(log)

            if RANK % 8 == 0:
                for key, val in log.items():
                    if key == "duration":
                        continue
                    tb_writer.add_scalar(
                        tag=key,
                        scalar_value=val,
                        global_step=current_step,
                        walltime=time.time(),
                    )
                tb_writer.flush()

                if perf is None:
                    perf = csv.DictWriter(
                        f_perf, list(log.keys()), delimiter=" ", quotechar='"'
                    )
                    perf.writeheader()

                perf.writerow(log)
                f_perf.flush()

                if not loss_contains_nan:
                    save_model(params, runname.with_suffix(".bin"))
                    save_model(model_state, runname.with_suffix(".state.bin"))
                    save_state(optim_state, runname.with_suffix(".optim.bin"))
                    save_other_state(
                        {"current_step": current_step}, runname.with_suffix(".json")
                    )

            t0 = time.perf_counter()
            t_data = 0
            num_steps = 0
            total_loss = CounterValue()
            grad_norm = CounterValue()
    if f_perf is not None:
        f_perf.close()
    if tb_writer is not None:
        tb_writer.close()
