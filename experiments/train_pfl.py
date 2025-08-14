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
from typing import Optional

import flax
import jax
import jax.numpy as jnp
import pandas
from loguru import logger
from tensorboardX import SummaryWriter

from pfl4asr.common import ConfigTrainingShared
from pfl4asr.modules.transformer_block import ConfigTransformerBlock
from pfl4asr.pfl.data import (
    check_clients_in_batch,
    create_dataset_fl,
)
from pfl4asr.pfl.server import Server
from pfl4asr.utils.counter import CounterArray, CounterValue
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
    ConfigLocalOptimization,
    ConfigServerOptimization,
    get_lr_schedule,
    get_optimizer,
)
from pfl4asr.utils.pfunctions import (
    create_ctc_loss_from_logits,
    pclip_grads,
    pclip_layer_grads,
    pclip_layer_grads_params,
    pl2_tree_func,
    pminus_tree_func,
    preduce_clients,
)
from pfl4asr.utils.train import create_step_and_eval_funcs, print_log


@dataclass
class ConfigDP:
    # Clip norm for deltas sent to the server from each device in FL. None avoids clipping.
    clip_deltas: Optional[float] = None
    # Clip norm for deltas sent to the server from each device in FL with uniform. None avoids clipping.
    clip_deltas_per_layer: Optional[float] = None
    # Uniform per layer clipping (True) or per dimension (False)
    use_uniform_per_layer_clip: bool = True
    # Clip norm for cohort aggregate deltas before central optimizer. None avoids clipping.
    clip_cohort_deltas: Optional[float] = None
    # Normalize cohort deltas by dividing them with their norm
    normalize_cohort_deltas: bool = False
    # sigma noise for DP added after average deltas is computed: dp_sigma = clipping * sigma_average (from the paper). If None - no DP
    dp_sigma: Optional[float] = None


@dataclass
class ConfigPFLTraining:
    # file with all data with client column
    train: str = ""
    client_key: str = "client_id"
    client_size_key: str = "client_size"
    client_data_key: str = "csv"
    # Cohort size for the federated learning version.
    cohort_size: int = 20
    # Batch size given in seconds per 1 client
    batch_size: int = 300
    # Number of server optimization steps
    server_steps: int = 2000


if __name__ == "__main__":
    from simple_parsing import ArgumentGenerationMode, ArgumentParser

    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.NESTED,
        add_config_path_arg=True,
        config_path="configs/default_fl.yaml",
    )
    parser.add_arguments(ConfigTrainingShared, dest="shared_config")
    parser.add_arguments(ConfigTransformerBlock, dest="block_config")
    parser.add_arguments(ConfigPFLTraining, dest="pfl_config")
    parser.add_arguments(ConfigServerOptimization, dest="server_optim_config")
    parser.add_arguments(ConfigLocalOptimization, dest="local_optim_config")
    parser.add_arguments(ConfigDP, dest="dp_config")

    parser.set_defaults(
        pfl_config=ConfigPFLTraining(),
        shared_config=ConfigTrainingShared(),
        block_config=ConfigTransformerBlock(),
        server_optim_config=ConfigServerOptimization(),
        local_optim_config=ConfigLocalOptimization(),
        dp_config=ConfigDP(),
    )
    all_args = parser.parse_args()
    pfl_config: ConfigPFLTraining = all_args.pfl_config
    shared_config: ConfigTrainingShared = all_args.shared_config
    block_config: ConfigTransformerBlock = all_args.block_config
    server_optim_config: ConfigServerOptimization = all_args.server_optim_config
    local_optim_config: ConfigLocalOptimization = all_args.local_optim_config
    dp_config: ConfigDP = all_args.dp_config

    WORLD_SIZE, RANK = init_distributed(
        host_ip_address=shared_config.host_ip_address,
        distributed_port=shared_config.distributed_port,
        rank=shared_config.rank,
        world_size=shared_config.world_size,
    )
    num_local_devices = jax.local_device_count()
    local_devices = jax.local_devices()
    if pfl_config.cohort_size % (WORLD_SIZE * num_local_devices) != 0:
        raise RuntimeError(
            "Cohort size should be divisible by processes * local devices for efficiency"
        )
    if num_local_devices != 1:
        raise RuntimeError(
            "Please use one process per device -- this is more efficient"
        )
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
        initial_server_step = 0
        logger.info(
            f"({RANK=}) Restoring model from the seed model {restore_model_path}"
        )
    else:
        # recover current model
        restore_model_path = runname.with_suffix(".bin")
        initial_server_step = load_other_state(
            restore_model_path.with_suffix(".json")
        ).get("current_step", 0)
        logger.info(
            f"({RANK=}) Restoring model from previous checkpoint {restore_model_path}"
        )

    params = load_model(params, restore_model_path)
    logger.info(f"({RANK=}) Model is restored from {restore_model_path}")
    logger.info(f"({RANK=}) Initial server step is {initial_server_step}")

    # Server optimizer
    logger.info(f"({RANK=}) Create optimizer")
    lr_schedule = get_lr_schedule(server_optim_config)
    server_optim = get_optimizer(server_optim_config, lr_schedule=lr_schedule)

    # Server
    server = Server(
        params=params,
        model_state=model_state,
        optim=server_optim,
        clip_cohort_deltas=dp_config.clip_cohort_deltas,
        normalize_cohort_deltas=dp_config.normalize_cohort_deltas,
        num_local_devices=num_local_devices,
        dp_sigma=dp_config.dp_sigma,
    )
    if shared_config.is_restore_optim and runname.with_suffix(".bin").exists():
        logger.info(
            f"({RANK=}) Restoring optimizer from {runname.with_suffix('.optim.bin')}"
        )
        server_optim_state = load_state(runname.with_suffix(".optim.bin"))
        if server_optim_state.count != initial_server_step:
            raise RuntimeError("Optimizer is loaded incorrectly")
        server.restore_from_checkpoint(optim_state=server_optim_state)
        logger.info(
            f"({RANK=}) Optimizer is restored from {runname.with_suffix('.optim.bin')}"
        )
    # Local optimizer
    local_optim = get_optimizer(local_optim_config, lr_schedule=None)

    @jax.jit
    def init_local_optimizer(params):
        return local_optim.init(params)

    pinit_local_optimizer = jax.pmap(init_local_optimizer, "xl")
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
        model, specaug, local_optim, mfsc_func, ctc_loss_from_logits, model_rng_keys
    )
    # Logging
    f_perf = None
    tb_writer = None
    if RANK % 8 == 0:
        f_perf = open(runname.with_suffix(f".{initial_server_step}.csv"), "w")
        tb_writer = SummaryWriter(logdir=runname.parent)
    perf = None

    t0 = time.perf_counter()
    t_data = 0
    start_t0 = time.perf_counter()

    rng = jax.random.PRNGKey(shared_config.seed)
    rng_noise = jax.random.PRNGKey(shared_config.seed + 42)

    server_loss_counter = CounterValue()
    average_client_average_grad_norm_counter = CounterValue()
    std_client_average_grad_norm_counter = CounterValue()
    server_l2_norm_counter = CounterValue()
    server_l2_norm_with_noise_counter = CounterValue()
    average_client_average_delta_norm_counter = CounterValue()
    average_client_average_clipped_delta_norm_counter = CounterValue()
    std_client_average_delta_norm_counter = CounterValue()

    logger.info(f"({RANK=}) Starting FL training")
    clients_pd = pandas.read_csv(
        os.path.join(shared_config.lists_dir, pfl_config.train), sep=" "
    )
    clients_names = {
        client_id: os.path.join(shared_config.lists_dir, path)
        for client_id, path in zip(
            clients_pd[pfl_config.client_key], clients_pd[pfl_config.client_data_key]
        )
    }
    clients_sizes = None
    if local_optim_config.num_steps < 0:
        # doing epochs not steps over each client
        clients_sizes = {
            client_id: size
            for client_id, size in zip(
                clients_pd[pfl_config.client_key],
                clients_pd[pfl_config.client_size_key],
            )
        }

    for current_server_step in range(initial_server_step, pfl_config.server_steps):
        server.reset_deltas()

        clients_datasets_iterator = create_dataset_fl(
            clients_names,
            clients_sizes,
            trie=trie,
            batch_size_s=pfl_config.batch_size,
            max_target_length=shared_config.max_target_len,
            batching_buffer_size=shared_config.batching_buffer_size,
            n_threads=shared_config.threads,
            input_key=shared_config.input_key,
            target_key=shared_config.target_key,
            target_pad=(not shared_config.target_nopad),
            tar=shared_config.tar,
            world_size=WORLD_SIZE,
            rank=RANK,
            sample_rate_hz=shared_config.sample_rate_hz,
            cohort_size=pfl_config.cohort_size,
            seed=shared_config.seed + current_server_step,
            local_epochs=local_optim_config.num_epochs,
            local_steps=local_optim_config.num_steps,
        )
        current_clients = None

        n_clients = jnp.zeros(shape=(1))
        server_step_loss_counter = CounterValue()
        average_grad_norm = CounterArray()
        average_delta_norm = CounterArray()
        average_delta_norm_clip = CounterArray()
        average_delta_norm_noise = CounterArray()

        for client_dataset in clients_datasets_iterator:
            # take num_local_devices clients to distribute over num_local_devices
            sample_cpu = None
            sample_device = None
            sample_cpu_next, sample_device_next = get_next_sample_on_device(
                client_dataset, trie, local_devices
            )
            local_steps = 0
            is_switch_to_new_clients, next_clients = check_clients_in_batch(
                sample_cpu_next=(
                    [sample_cpu_next] if sample_cpu_next is not None else None
                ),
                current_clients=current_clients,
                client_key=pfl_config.client_key,
            )

            while sample_cpu_next is not None and sample_device_next is not None:
                sample_cpu, sample_device = sample_cpu_next, sample_device_next
                current_clients = next_clients
                if (
                    local_optim_config.num_steps > 0
                    and local_steps >= local_optim_config.num_steps - 1
                ):
                    sample_cpu_next, sample_device_next = None, None
                else:
                    t_data_tmp = time.perf_counter()
                    sample_cpu_next, sample_device_next = get_next_sample_on_device(
                        client_dataset, trie, local_devices
                    )
                    t_data += time.perf_counter() - t_data_tmp
                if is_switch_to_new_clients:
                    # we didn't finish going through whole cohort
                    logger.info(f"({RANK=}) Training clients: {current_clients}")
                    # Every client must start with the same parameters (prior to the FL round).
                    params = server.current_params()
                    model_state = server.current_model_state()

                    # We must reset optimizer for each cohort.
                    local_optim_state = pinit_local_optimizer(params)
                    local_losses = [CounterValue() for _ in range(num_local_devices)]
                    local_grad_norms = [
                        CounterValue() for _ in range(num_local_devices)
                    ]

                x, y, x_length, y_length = sample_device
                rng, prng = jax.random.split(rng, 2)
                prng = jax.random.split(prng, num_local_devices)

                use_specaug = (
                    True if current_server_step > shared_config.start_saug else False
                )
                (
                    params,
                    model_state,
                    local_optim_state,
                    loss_value,
                    grads_l2_norm,
                    _,
                ) = ptrain_step(
                    params,
                    model_state,
                    local_optim_state,
                    x,
                    x_length,
                    y,
                    y_length,
                    prng,
                    None,  # fedprox spaceholder
                    None,  # fedprox spaceholder
                    use_specaug,
                    True,
                    local_optim_config.max_grad_norm,
                    False,  # no all reduce as every GPU = every client
                )
                if jnp.isnan(loss_value.sum()):
                    logger.info(f"({RANK=}) BUG: loss contains NaN")
                local_steps += 1
                for device_idx in range(num_local_devices):
                    logger.info(
                        f"({RANK=}) Local loss {float(loss_value[device_idx])} on device {device_idx} for {local_steps=} at server step={current_server_step}"
                    )
                    local_losses[device_idx].add(float(loss_value[device_idx]))
                    local_grad_norms[device_idx].add(float(grads_l2_norm[device_idx]))

                is_switch_to_new_clients, new_clients = check_clients_in_batch(
                    sample_cpu_next=(
                        [sample_cpu_next] if sample_cpu_next is not None else None
                    ),
                    current_clients=current_clients,
                    client_key=pfl_config.client_key,
                )
                if is_switch_to_new_clients:
                    for device_idx in range(num_local_devices):
                        # means that we have previous client training done
                        logger.info(
                            f"({RANK=}) Done client {current_clients[device_idx]} training; average client loss is {local_losses[device_idx].get_average()}, \
                            average client grad norm {local_grad_norms[device_idx].get_average()} for server step={current_server_step}"
                        )
                        average_grad_norm.add(
                            local_grad_norms[device_idx].get_average()
                        )
                        server_step_loss_counter.add(
                            local_losses[device_idx].get_average()
                        )
                    # we take old - new to get the grad exactly
                    delta_params = pminus_tree_func(server.current_params(), params)

                    if (
                        dp_config.clip_deltas is not None
                        or dp_config.clip_deltas_per_layer is not None
                    ):
                        logger.info(
                            f"({RANK=}) Clipping deltas before sending to server"
                        )
                        if dp_config.clip_deltas is not None:
                            delta_params, l2_norm, per_layer_l2_norm = pclip_grads(
                                delta_params,
                                flax.jax_utils.replicate(dp_config.clip_deltas),
                            )
                        elif (
                            dp_config.clip_deltas_per_layer is not None
                            and dp_config.use_uniform_per_layer_clip
                        ):
                            delta_params, l2_norm, per_layer_l2_norm = (
                                pclip_layer_grads(
                                    delta_params,
                                    flax.jax_utils.replicate(
                                        dp_config.clip_deltas_per_layer
                                    ),
                                )
                            )
                        elif (
                            dp_config.clip_deltas_per_layer is not None
                            and not dp_config.use_uniform_per_layer_clip
                        ):
                            delta_params, l2_norm, per_layer_l2_norm = (
                                pclip_layer_grads_params(
                                    delta_params,
                                    flax.jax_utils.replicate(
                                        dp_config.clip_deltas_per_layer
                                    ),
                                )
                            )
                        for i in range(num_local_devices):
                            jax.debug.print(
                                "delta_params norm before clipping on {device}: {l2_norm}",
                                device=i,
                                l2_norm=float(l2_norm[i]),
                            )
                            average_delta_norm_clip.add(float(l2_norm[i]))

                    delta_l2_norm = pl2_tree_func(delta_params)
                    for i in range(num_local_devices):
                        average_delta_norm.add(float(delta_l2_norm[i]))

                    server.add_deltas(delta_params)
                    n_clients += len(current_clients)

        logger.info(
            f"({RANK=}) Done training for all clients {n_clients=} in cohort for server step={current_server_step}"
        )

        # synchronize across devices
        n_clients = preduce_clients(n_clients)
        if int(n_clients[0]) != pfl_config.cohort_size:
            raise RuntimeError(
                f"({RANK=}) Number of processed clients in cohort is wrong, {n_clients=} != {pfl_config.cohort_size}"
            )

        average_client_average_grad_norm_counter.add(average_grad_norm.get_average())
        std_client_average_grad_norm_counter.add(average_grad_norm.get_std())
        average_client_average_delta_norm_counter.add(average_delta_norm.get_average())
        average_client_average_clipped_delta_norm_counter.add(
            average_delta_norm_clip.get_average()
        )
        std_client_average_delta_norm_counter.add(average_delta_norm.get_std())
        server_loss_counter.add(server_step_loss_counter.get_average())
        server_l2_norm, server_l2_norm_noise, rng_noise = server.step(rng_noise)
        server_l2_norm_counter.add(float(server_l2_norm[0]))
        server_l2_norm_with_noise_counter.add(float(server_l2_norm_noise[0]))
        logger.info(
            f"({RANK=}) Done updating parameters for server step={current_server_step}"
        )

        if (
            current_server_step % shared_config.report == 0 and current_server_step > 0
        ) or (current_server_step == pfl_config.server_steps - 1):
            t = time.perf_counter() - t0

            log = {
                "server-step": current_server_step,
                "duration": str(datetime.timedelta(seconds=t)),
                "ms/central-step": t / shared_config.report * 1000.0,
                "ms/central-step data loader": t_data / shared_config.report * 1000.0,
                "ms/client": t
                / shared_config.report
                * 1000.0
                / (pfl_config.cohort_size / (WORLD_SIZE * num_local_devices)),
                "ms/client-step(epoch)": t
                / shared_config.report
                * 1000.0
                / (pfl_config.cohort_size / (WORLD_SIZE * num_local_devices))
                / (
                    local_optim_config.num_epochs
                    if local_optim_config.num_steps < 0
                    else local_optim_config.num_steps
                ),
                "loss": server_loss_counter.get_average(),
                "cohort average grad norm": average_client_average_grad_norm_counter.get_average(),
                "cohort std grad norm": std_client_average_grad_norm_counter.get_average(),
                "cohort average delta norm": average_client_average_delta_norm_counter.get_average(),
                "cohort average delta norm before clipping": average_client_average_clipped_delta_norm_counter.get_average(),
                "cohort std delta norm": std_client_average_delta_norm_counter.get_average(),
                "server grad norm": server_l2_norm_counter.get_average(),
                "server grad norm after DP noise": server_l2_norm_with_noise_counter.get_average(),
                "lr": float(server.get_learning_rate()),
            }

            t_start_eval = time.perf_counter()
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
                    params=server.current_params(),
                    model_state=server.current_model_state(),
                    pctc_loss_from_logits=pctc_loss_from_logits,
                    rng=rng,
                    local_devices=local_devices,
                    n_threads=shared_config.threads,
                )
            log["s/eval"] = time.perf_counter() - t_start_eval

            send_metrics(
                log, current_step=current_server_step, rank=RANK, world_size=WORLD_SIZE
            )
            print_log(log)

            if RANK % 8 == 0:
                for key, val in log.items():
                    if key == "duration":
                        continue
                    tb_writer.add_scalar(
                        tag=key,
                        scalar_value=val,
                        global_step=current_server_step,
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

                save_model(server.current_params(), runname.with_suffix(".bin"))
                save_model(
                    server.current_model_state(), runname.with_suffix(".state.bin")
                )
                save_state(server.optim_state, runname.with_suffix(".optim.bin"))
                save_other_state(
                    {"current_step": current_server_step}, runname.with_suffix(".json")
                )

            t0 = time.perf_counter()
            t_data = 0

            server_loss_counter = CounterValue()
            average_client_average_grad_norm_counter = CounterValue()
            std_client_average_grad_norm_counter = CounterValue()
            server_l2_norm_counter = CounterValue()
            server_l2_norm_with_noise_counter = CounterValue()
            average_client_average_delta_norm_counter = CounterValue()
            average_client_average_clipped_delta_norm_counter = CounterValue()
            std_client_average_delta_norm_counter = CounterValue()

    if f_perf is not None:
        f_perf.close()
    if tb_writer is not None:
        tb_writer.close()
