#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import mlx.data.core as dxcore
import numpy as np
from loguru import logger

from ..utils.data import create_dataset


def create_dataset_fl(
    clients_names: dict,
    clients_sizes: dict | None,
    trie: dxcore.CharTrie,
    batch_size_s: int | float,
    max_target_length=500,
    n_threads=8,
    world_size=1,
    rank=0,
    # either csv file column name containing tar file where audio is stored,
    # or the tar file itself if all data are stored in it
    tar="",
    target_pad=False,
    input_key="file",
    target_key="transcription",
    sample_rate_hz=16000,
    pad_to_seconds=5,
    pad_to_batch=8,
    batching_buffer_size=64,
    shuffle_buffer_size=20000,
    cohort_size=1,
    local_epochs=1,
    local_steps=-1,
    seed=42,
    max_audio_len_s=30,
):
    np.random.seed(seed)
    clients = np.array(list(clients_names.keys()))
    chosen_clients_all = clients[np.random.permutation(len(clients))[:cohort_size]]
    if clients_sizes is None or local_steps > 0:
        chosen_clients = chosen_clients_all[rank::world_size]
    else:
        if local_epochs <= 0:
            raise RuntimeError(
                f"Local epochs {local_epochs} should be positive, otherwise use local steps"
            )
        # greedy algorithm on splitting clients in case of epochs training to improve efficiency of gpu load
        chosen_sizes_all = np.array(
            [clients_sizes[client] for client in chosen_clients_all]
        )
        indices = np.argsort(chosen_sizes_all)[::-1]
        chosen_sizes_all = chosen_sizes_all[indices]
        chosen_clients_all = chosen_clients_all[indices]
        sizes_per_rank = np.zeros(world_size)
        chosen_clients = []
        for client, size in zip(chosen_clients_all, chosen_sizes_all):
            current_idx = np.argmin(sizes_per_rank)
            sizes_per_rank[current_idx] += size
            if current_idx == rank:
                chosen_clients.append(client)
        largest_device_proxy = sizes_per_rank[np.argmax(sizes_per_rank)]
        current_device_proxy = sizes_per_rank[rank]
        missing_samples = (largest_device_proxy - current_device_proxy) * local_epochs
        proxy_wait_s = missing_samples / (batch_size_s / 10) * 0.3

        import time

        # slow down the process in case we have smaller number of batches in the cohort
        # this is due to hardcoded 10min communication timeout in jax for nccl
        # the best is to run only with 2/4 gpus training with epochs
        time.sleep(proxy_wait_s)

    current_batch = None
    next_batch = None
    for client in chosen_clients:
        logger.info(f"Creating client dataset from {clients_names[client]}")
        next_batch = create_dataset(
            name=clients_names[client],
            trie=trie,
            batch_size_s=batch_size_s,
            max_target_length=max_target_length,
            n_threads=n_threads,
            world_size=1,
            rank=0,
            tar=tar,
            target_pad=target_pad,
            input_key=input_key,
            target_key=target_key,
            sample_rate_hz=sample_rate_hz,
            pad_to_seconds=pad_to_seconds,
            pad_to_batch=pad_to_batch,
            batching_buffer_size=batching_buffer_size,
            shuffle_buffer_size=shuffle_buffer_size,
            # loop number of epochs or otherwise loop infinite to make specific steps
            loop=1,
            loop_batches=local_epochs if local_steps < 0 else -1,
            max_audio_len_s=max_audio_len_s,
        )
        if current_batch is not None:
            yield current_batch
        current_batch = next_batch
    if current_batch is not None:
        yield current_batch


def check_clients_in_batch(
    sample_cpu_next: list[dict] | None,
    current_clients: list[str] | None,
    client_key: str,
) -> tuple[bool, list[str]]:
    if sample_cpu_next is None:
        return True, None
    if current_clients is None:
        current_clients = [""] * len(sample_cpu_next)
    num_local_devices = len(sample_cpu_next)
    if len(sample_cpu_next) != len(current_clients):
        raise RuntimeError("Error in the batch")
    new_clients = []
    counter_not_previous_client = 0
    counter_not_previous_client_empty = 0
    for device_idx, (sample, current_client) in enumerate(
        zip(sample_cpu_next, current_clients)
    ):
        if sample is None:
            new_clients.append(current_client)
            continue
        client_ids = [
            "".join([chr(index) for index in client_id])
            for client_id in sample[client_key]
        ]
        if len(set(client_ids)) != 1:
            raise RuntimeError(
                f"Mix of clients are detected in the client batch on device {device_idx}"
            )
        if client_ids[0] != current_client:
            counter_not_previous_client += 1
            new_clients.append(client_ids[0])
            if current_client != "":
                counter_not_previous_client_empty += 1
        else:
            new_clients.append(current_client)

    if (
        counter_not_previous_client != 0
        and counter_not_previous_client != num_local_devices
    ):
        raise RuntimeError("Clients across devices are not aligned in steps")

    if counter_not_previous_client == num_local_devices and (
        counter_not_previous_client_empty != 0
        and counter_not_previous_client_empty != num_local_devices
    ):
        # we switch to new set of clients on all devices
        # but there is issue how we switch from previous clients if they were non empty
        raise RuntimeError(
            "Clients across devices are not aligned in steps for previous empty client"
        )
    is_switch_to_new_clients = counter_not_previous_client == num_local_devices
    if len(new_clients) != len(current_clients):
        raise RuntimeError("Error in updating clients")
    return is_switch_to_new_clients, new_clients
