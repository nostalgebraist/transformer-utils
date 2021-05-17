from functools import partial

import torch
import numpy as np
import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa

from .hooks import make_lens_hooks
from .layer_names import make_layer_names


def collect_logits(model, input_ids, layer_names, decoder_layer_names):
    needs_forward = True
    if model._last_input_ids is not None:
        if model._last_input_ids.shape == input_ids.shape:
            needs_forward = not (model._last_input_ids == input_ids).cpu().all()

    if layer_names is None:
        layer_names = model._ordered_layer_names
    else:
        layers_instrumented = model._layer_logits_handles.keys()
        needs_forward = needs_forward or set(layer_names).difference(layers_instrumented) != set()

    needs_forward = needs_forward or decoder_layer_names != model._lens_decoder_layer_names

    model._last_resid = None

    if needs_forward:
        with torch.no_grad():
            out = model(input_ids)
        del out
        model._last_resid = None

    layer_logits = np.concatenate(
        [model._layer_logits[name] for name in layer_names],
        axis=0,
    )

    return layer_logits, layer_names


def postprocess_logits(layer_logits):
    layer_preds = layer_logits.argmax(axis=-1)

    layer_probs = scipy.special.softmax(layer_logits, axis=-1)

    return layer_preds, layer_probs


def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)


def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark


def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)


def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)


def kl_div(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)


def _plot_logit_lens(
    layer_logits,
    layer_preds,
    layer_probs,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    probs=False,
    ranks=False,
    kl=False,
):
    end_ix = start_ix + layer_logits.shape[1]

    final_preds = layer_preds[-1]

    aligned_preds = layer_preds

    if kl:
        clip = 1 / (10 * layer_probs.shape[-1])
        final_probs = layer_probs[-1]
        to_show = kl_div(final_probs, layer_probs, clip=clip)
    else:
        numeric_input = layer_probs if probs else layer_logits

        to_show = get_value_at_preds(numeric_input, final_preds)

        if ranks:
            to_show = (numeric_input >= to_show[:, :, np.newaxis]).sum(axis=-1)

    _num2tok = np.vectorize(
        partial(num2tok, tokenizer=tokenizer, quotemark="'"), otypes=[str]
    )
    aligned_texts = _num2tok(aligned_preds)

    to_show = to_show[::-1]

    aligned_texts = aligned_texts[::-1]

    fig = plt.figure(figsize=(1.5 * to_show.shape[1], 0.375 * to_show.shape[0]))

    plot_kwargs = {"annot": aligned_texts, "fmt": ""}
    if kl:
        vmin, vmax = None, None

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40_r",
                "vmin": vmin,
                "vmax": vmax,
                "annot": True,
                "fmt": ".1f",
            }
        )
    elif ranks:
        vmax = 2000
        plot_kwargs.update(
            {
                "cmap": "Blues",
                "norm": mpl.colors.LogNorm(vmin=1, vmax=vmax),
                "annot": True,
            }
        )
    elif probs:
        plot_kwargs.update({"cmap": "Blues_r", "vmin": 0, "vmax": 1})
    else:
        vmin = np.percentile(to_show.reshape(-1), 5)
        vmax = np.percentile(to_show.reshape(-1), 95)

        plot_kwargs.update(
            {
                "cmap": "cet_linear_protanopic_deuteranopic_kbw_5_98_c40",
                "vmin": vmin,
                "vmax": vmax,
            }
        )

    sns.heatmap(to_show, **plot_kwargs)

    ax = plt.gca()
    input_tokens_str = _num2tok(input_ids[0].cpu())
    ax.set_xticklabels(input_tokens_str[start_ix:end_ix], rotation=0)

    if layer_names is None:
        layer_names = ["Layer {}".format(n) for n in range(to_show.shape[0])]
    ylabels = layer_names[::-1]
    ax.set_yticklabels(ylabels, rotation=0)

    tick_locs = ax.get_xticks()

    ax_top = ax.twiny()
    padw = 0.5 / to_show.shape[1]
    ax_top.set_xticks(np.linspace(padw, 1 - padw, to_show.shape[1]))

    starred = [
        "* " + true if pred == true else " " + true
        for pred, true in zip(
            aligned_texts[0], input_tokens_str[start_ix + 1 : end_ix + 1]
        )
    ]
    ax_top.set_xticklabels(starred, rotation=0)


def plot_logit_lens(
    model,
    tokenizer,
    input_ids,
    start_ix: int,
    end_ix: int,
    probs=False,
    ranks=False,
    kl=False,
    block_step=1,
    include_input=True,
    force_include_output=True,
    include_subblocks=False,
    decoder_layer_names: list = ['final_layernorm', 'lm_head'],
    verbose=False
):
    layer_names = make_layer_names(
        model,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names
    )

    make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names,
                    decoder_layer_names=decoder_layer_names,
                    verbose=verbose)

    layer_logits, layer_names = collect_logits(
        model, input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names
    )

    layer_preds, layer_probs = postprocess_logits(layer_logits)

    _plot_logit_lens(
        layer_logits=layer_logits,
        layer_preds=layer_preds,
        layer_probs=layer_probs,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        probs=probs,
        ranks=ranks,
        kl=kl,
        layer_names=layer_names,
    )
