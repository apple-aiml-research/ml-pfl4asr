#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import jax
from loguru import logger


def init_distributed(host_ip_address=None, distributed_port=None, world_size=1, rank=0):
    try:
        if world_size > 1:
            if host_ip_address is None or host_ip_address == "":
                host_ip_address = "127.0.0.1"
            if distributed_port is None or distributed_port == "":
                distributed_port = "8888"
            logger.info(
                f"Trying to initialize distributed training: {host_ip_address=} {distributed_port=} {world_size=} {rank=}"
            )
            jax.distributed.initialize(
                coordinator_address="{}:{}".format(host_ip_address, distributed_port),
                num_processes=world_size,
                process_id=rank,
            )
            logger.info(
                f"initialized distributed training with {world_size=} and {rank=}"
            )
        else:
            logger.info("Distributed task is not initialized, use 1 process training")
    except Exception as e:
        world_size = 1
        rank = 0
        logger.info(f"{e} Distributed task is not initialized, use 1 process training")

    logger.info(f"Total devices {jax.device_count()}")
    return world_size, rank
