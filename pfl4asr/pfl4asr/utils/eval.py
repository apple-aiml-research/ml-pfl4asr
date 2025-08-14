#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import jax
import jax.numpy as jnp
import mlx.data
import numpy as np
from einops import rearrange
from jiwer import wer
from joblib import Parallel, delayed

from .data import BLANK_TOKEN, get_next_sample_on_device


def wer_subprocess(y, y_pred, y_length, normalized=False) -> float:
    if normalized:
        return wer(y, y_pred)
    else:
        return wer(y, y_pred) * y_length


def compute_wer_batched(parallel: Parallel, y: list, y_pred: list, normalized=False):
    y_words_length = [len(y_i.split(" ")) for y_i in y]
    if parallel is not None:
        wer_batch = parallel(
            delayed(wer_subprocess)(y_i, y_pred_i, y_len, normalized=normalized)
            for y_i, y_pred_i, y_len in zip(y, y_pred, y_words_length, strict=True)
            if len(y_i) > 0
        )
    else:
        wer_batch = [
            wer_subprocess(y_i, y_pred_i, y_len, normalized=normalized)
            for y_i, y_pred_i, y_len in zip(y, y_pred, y_words_length, strict=True)
            if len(y_i) > 0
        ]

    if normalized:
        return np.sum(wer_batch), len(y), wer_batch
    else:
        return np.sum(wer_batch), np.sum(y_words_length), wer_batch


def y_to_str_subprocess(index_key, y, y_length, trie, use_word=False):
    txt = []
    for index in range(y_length):
        token = trie[y[index]]
        if not use_word:
            token = "_" if token == " " else token
        txt.append(token)
    if use_word:
        return (index_key, "".join(txt))
    else:
        return (index_key, " ".join(txt))


def y_to_string_batched(parallel, y, y_length, trie, use_word=False):
    if parallel is not None:
        result = parallel(
            delayed(y_to_str_subprocess)(
                (batch_idx, sample_idx),
                y[batch_idx][sample_idx, :],
                y_length[batch_idx][sample_idx],
                trie,
                use_word,
            )
            for batch_idx, batch in enumerate(y)
            for sample_idx in range(batch.shape[0])
        )
        return [elem[1] for elem in sorted(result, key=lambda tup: tup[0])]
    else:
        return [
            y_to_str_subprocess(
                0,
                y[batch_idx][sample_idx, :],
                y_length[batch_idx][sample_idx],
                trie,
                use_word,
            )[1]
            for batch_idx, batch in enumerate(y)
            for sample_idx in range(batch.shape[0])
        ]


def compute_wer_ter_loss(
    peval,
    dataset,
    trie,
    params,
    model_state,
    pctc_loss_from_logits,
    rng,
    local_devices,
    n_threads=1,
) -> tuple[float, float, float]:
    total_err = 0
    total_num_tokens = 0
    total_loss = 0
    num_iters = 0
    yp_all = []
    y_all = []
    yp_length_all = []
    y_length_all = []

    _, next_sample = get_next_sample_on_device(dataset, trie, local_devices)
    num_local_devices = len(local_devices)
    while next_sample is not None:
        try:
            sample = next_sample
            _, next_sample = get_next_sample_on_device(dataset, trie, local_devices)
        except:
            next_sample = None
        x, y, x_length, y_length = sample

        prng = jax.random.split(rng, num_local_devices)
        (yp, yp_length), model_state_updated = peval(
            params, model_state, x, x_length, prng, False, False
        )
        loss = pctc_loss_from_logits(yp, yp_length, y, y_length).mean().item()
        yp = jnp.argmax(yp, -1)

        yp = np.array(
            rearrange(
                yp,
                "devices b ... -> (devices b) ...",
                devices=num_local_devices,
            )
        ).astype(np.int64)
        yp_length = np.array(
            rearrange(
                yp_length,
                "devices b ... -> (devices b) ...",
                devices=num_local_devices,
            )
        ).astype(np.int64)
        y = np.array(
            rearrange(
                y,
                "devices b ... -> (devices b) ...",
                devices=num_local_devices,
            )
        ).astype(np.int64)
        y_length = np.array(
            rearrange(
                y_length,
                "devices b ... -> (devices b) ...",
                devices=num_local_devices,
            )
        ).astype(np.int64)

        yp, yp_length = mlx.data.core.uniq(yp, yp_length, -1, -1)
        BLANK_INDEX = trie.search(BLANK_TOKEN).id
        yp, yp_length = mlx.data.core.remove(
            yp, yp_length, -1, BLANK_INDEX, -1
        )  # remove blank
        err = mlx.data.core.levenshtein(yp, yp_length, y, y_length)

        yp_all.append(yp)
        y_all.append(y)
        yp_length_all.append(yp_length)
        y_length_all.append(y_length)

        total_err += err.sum()
        total_num_tokens += y_length.sum()
        total_loss += loss
        num_iters += 1

    # for some reason it is faster than C++ one (maybe overhead because of simple trie)
    trie_python = {index: trie.key_string(index) for index in range(trie.num_keys())}

    with Parallel(n_jobs=n_threads) as parallel:
        yp_strs = y_to_string_batched(
            parallel, yp_all, yp_length_all, trie_python, True
        )
        y_strs = y_to_string_batched(parallel, y_all, y_length_all, trie_python, True)
        total_wer, total_num_words, _ = compute_wer_batched(
            parallel, y_strs, yp_strs, normalized=False
        )
    return (
        total_wer / total_num_words * 100,
        total_err / total_num_tokens * 100,
        total_loss / num_iters,
    )
