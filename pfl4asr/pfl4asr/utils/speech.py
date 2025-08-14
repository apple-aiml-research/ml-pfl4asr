#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import enum
import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp


class WindowType(enum.Enum):
    Hamming = 0
    Hanning = 1


class FrequencyScale(enum.Enum):
    MEL = 0
    LOG10 = 1
    LINEAR = 2


def next_pow_2(n):
    return 2 ** math.ceil(math.log2(n))


def sliding_window_output_length(window, stride, input_length):
    if type(input_length) == int:
        return max(input_length - window, 0) // stride + 1
    else:
        return jnp.maximum(input_length - window, 0) // stride + 1


def num_sample_per_frame(sampling_freq, frame_size_ms):
    return round(sampling_freq * frame_size_ms / 1000.0)


def compute_derivative(input, windowlen):
    norm = (windowlen * (windowlen + 1) * (2 * windowlen + 1)) / 3.0
    input = jnp.pad(input, ((windowlen, windowlen), (0, 0)), mode="edge")
    kernel = jnp.arange(-windowlen, windowlen + 1)[:, None]
    return jsp.signal.correlate(input, kernel, mode="valid") / norm


def hertz_to_warped_scale(hz, freqscale):
    if freqscale == FrequencyScale.MEL:
        return 2595.0 * jnp.log10(1.0 + hz / 700.0)
    elif freqscale == FrequencyScale.LOG10:
        return jnp.log10(hz)
    elif freqscale == FrequencyScale.LINEAR:
        return hz
    else:
        raise ValueError("invalid freqscale")


def warped_to_hertz_scale(wrp, freqscale):
    if freqscale == FrequencyScale.MEL:
        return 700.0 * (10 ** (wrp / 2595.0) - 1)
    elif freqscale == FrequencyScale.LOG10:
        return jnp.pow(10, wrp)
    elif freqscale == FrequencyScale.LINEAR:
        return wrp
    else:
        raise ValueError("invalid freqscale")


def derivatives(windowlen, dblwindowlen=0):
    def apply(input):
        res = [input]
        if windowlen > 0:
            deltas = compute_derivative(input, windowlen)
            res.append(deltas)
            if dblwindowlen > 0:
                res.append(compute_derivative(deltas, dblwindowlen))
        return jnp.concatenate(res, -1)

    return apply


def tri_filterbank(
    numfilters,
    filterlen,
    samplingfreq,
    lowfreq=0,
    highfreq=-1,
    melfloor=0.0,
    freqscale=FrequencyScale.MEL,
):
    if highfreq <= 0:
        highfreq = samplingfreq // 2

    minwarpfreq = hertz_to_warped_scale(lowfreq, freqscale)
    maxwarpfreq = hertz_to_warped_scale(highfreq, freqscale)
    dwarp = (maxwarpfreq - minwarpfreq) / (numfilters + 1)

    f = jnp.arange(numfilters + 2)
    f = (
        warped_to_hertz_scale(f * dwarp + minwarpfreq, freqscale)
        * (filterlen - 1.0)
        * 2.0
        / samplingfreq
    )

    hislope = jnp.arange(filterlen)[:, None] - f[None, :]
    hislope = hislope / (jnp.roll(f, -1) - f)
    hislope = hislope[:, :numfilters]

    loslope = jnp.roll(f, -2)[None, :] - jnp.arange(filterlen)[:, None]
    loslope = loslope / (jnp.roll(f, -2) - jnp.roll(f, -1))
    loslope = loslope[:, :numfilters]

    H = jnp.maximum(jnp.minimum(hislope, loslope), 0.0)

    def apply(input):
        return jnp.maximum(input @ H, melfloor)

    return apply


def dither(coeff=0.1):
    def apply(input, random_key):
        return input + coeff * jax.random.normal(random_key, input.shape)

    return apply


def pre_emphasis(coeff=1.0):
    def apply(input):
        inputm1 = jnp.roll(input, 1, -1)
        inputm1 = inputm1.at[:, 0].set(input[:, 0])
        res = input - coeff * inputm1
        return res

    return apply


def sliding_window(window, stride):
    def apply(input):
        osz = sliding_window_output_length(window, stride, input.shape[0])
        idx = stride * jnp.arange(osz)[:, None] + jnp.arange(window)[None, :]
        return input[idx]

    return apply


def windowing(window, windowtype):
    if windowtype == WindowType.Hamming:
        coeffs = jnp.hamming(window)
    elif windowtype == WindowType.Hanning:
        coeffs = jnp.hanning(window)
    else:
        raise ValueError("invalid windowtype")

    def apply(input):
        return input * coeffs

    return apply


def power_spectrum(n_fft):
    n_fft_div2 = n_fft // 2 + 1

    def apply(input):
        out = jnp.abs(jnp.fft.fft(input, n_fft))
        out = out[:, :n_fft_div2]
        return out

    return apply


def mfsc(
    n_filterbank,
    sampling_freq,
    frame_size_ms=25,
    frame_stride_ms=10,
    dither_coeff=0.0,
    pre_emphasis_coeff=0.97,
    window_type=WindowType.Hamming,
    use_energy=False,
    use_energy_raw=False,
    use_power=False,
    low_freq=0,
    high_freq=-1,
    mel_floor=1.0,
    freq_scale=FrequencyScale.MEL,
    delta_window=0,
    ddelta_window=0,
    post_process=None,
):
    n_sample_per_frame = num_sample_per_frame(sampling_freq, frame_size_ms)
    n_sample_per_stride = num_sample_per_frame(sampling_freq, frame_stride_ms)
    slwin = sliding_window(n_sample_per_frame, n_sample_per_stride)
    n_fft = next_pow_2(n_sample_per_frame)
    di = dither(dither_coeff)
    pe = pre_emphasis(pre_emphasis_coeff)
    win = windowing(n_sample_per_frame, window_type)
    ps = power_spectrum(n_fft)
    tfb = tri_filterbank(
        n_filterbank,
        n_fft // 2 + 1,
        sampling_freq,
        low_freq,
        high_freq,
        mel_floor,
        freq_scale,
    )
    der = derivatives(delta_window, ddelta_window)

    if not post_process:

        def post_process(out, energy):
            if use_energy:
                out = jnp.concatenate([energy, out], -1)
            return out

    def apply(input, random_key=None):
        out = input * 32768.0
        out = slwin(out)
        energy = None
        if use_energy and use_energy_raw:
            energy = jnp.log(
                jnp.maximum(
                    jnp.sum(out * out, -1, keepdims=True), jnp.finfo(jnp.float32).tiny
                )
            )
        if dither_coeff > 0:
            out = di(out, random_key)
        out = pe(out)
        out = win(out)
        if use_energy and not use_energy_raw:
            energy = jnp.log(
                jnp.maximum(
                    jnp.sum(out * out, -1, keepdims=True), jnp.finfo(jnp.float32).tiny
                )
            )
        out = ps(out)
        if use_power:
            out = out * out
        out = tfb(out)
        out = jnp.log(jnp.maximum(out, jnp.finfo(jnp.float32).tiny))
        out = post_process(out, energy)
        out = der(out)
        return out

    return apply
