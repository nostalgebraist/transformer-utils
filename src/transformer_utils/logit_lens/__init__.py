from functools import partial

import torch
import torch.nn as nn
import numpy as np
import scipy.special
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet

from ..util.python_utils import make_print_if_verbose
from ..util.module_utils import get_child_module_by_names


def make_lens_hooks(
    model, layer_names: list = None, prefixes: list = ["transformer", "h"], verbose=True
):
    vprint = make_print_if_verbose(verbose)

    for attr in ["_layer_logits", "_layer_logits_handles"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    if not hasattr(model, "_last_input_ids"):
        model._last_input_ids = None
        model._last_input_ids_handle = None

    if not hasattr(model, "_ln_base"):
        # TODO: use
        model._ln_base = (
            nn.LayerNorm(model.config.hidden_size)
            .to(model.device)
            .requires_grad_(False)
        )

    if layer_names is None:
        h = get_child_module_by_names(model, prefixes)
        layer_names = list(range(len(h)))

    def _get_layer(name):
        return get_child_module_by_names(model, prefixes + [str(name)])

    def _make_record_logits_hook(name):
        model._layer_logits[name] = None

        def _record_logits_hook(module, input, output) -> None:
            del model._layer_logits[name]
            model._layer_logits[name] = model.lm_head(model.transformer.ln_f(output[0]))

        return _record_logits_hook

    def _record_input_ids_hook(module, input, output):
        model._last_input_ids = input[0]

    def _hook_already_there(name):
        handle = model._layer_logits_handles.get(name)
        if not handle:
            return False
        layer = _get_layer(name)
        return handle.id in layer._forward_hooks

    for name in layer_names:
        if _hook_already_there(name):
            vprint(f"skipping layer {name}, hook already exists")
            continue
        layer = _get_layer(name)
        handle = layer.register_forward_hook(_make_record_logits_hook(name))
        model._layer_logits_handles[name] = handle

    if model._last_input_ids_handle is None:
        handle = model.transformer.wte.register_forward_hook(_record_input_ids_hook)
        model._last_input_ids_handle = handle


def collect_logits(model, input_ids):
    needs_forward = True
    if model._last_input_ids is not None:
        if model._last_input_ids.shape == input_ids.shape:
            needs_forward = not (model._last_input_ids == input_ids).cpu().all()

    if needs_forward:
        with torch.no_grad():
            out = model(input_ids)
        del out

    layer_logits = torch.cat(
        [model._layer_logits[name] for name in sorted(model._layer_logits.keys())],
        dim=0,
    )

    layer_logits = layer_logits.cpu().numpy()

    return layer_logits


def postprocess_logits(layer_logits):
    layer_preds = layer_logits.argmax(axis=-1)

    layer_probs = scipy.special.softmax(layer_logits, axis=-1)

    return layer_preds, layer_probs


def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)


def num2tok(x, tokenizer):
    return str(tokenizer.decode([x]))


def _plot_logit_lens(
    layer_logits,
    tokenizer,
    input_ids,
    start_ix: int,
    end_ix: int,
    probs=False,
):
    layer_preds, layer_probs = postprocess_logits(layer_logits)

    final_preds = layer_preds[-1]

    if probs:
        to_show = get_value_at_preds(layer_probs, final_preds)
    else:
        to_show = get_value_at_preds(layer_logits, final_preds)

    to_show = to_show[:, start_ix:end_ix][::-1]

    aligned_preds = layer_preds[:, start_ix:end_ix][::-1]

    _num2tok = np.vectorize(partial(num2tok, tokenizer=tokenizer), otypes=[str])
    aligned_texts = _num2tok(aligned_preds)

    fig = plt.figure(figsize=(1.5 * to_show.shape[1], 10))

    cmap = "Blues_r" if probs else "cet_linear_protanopic_deuteranopic_kbw_5_98_c40"

    sns.heatmap(
        to_show,
        cmap=cmap,
        annot=aligned_texts,
        fmt="",
    )

    ax = plt.gca()
    input_tokens_str = _num2tok(input_ids[0].cpu())
    ax.set_xticklabels(input_tokens_str[start_ix:end_ix], rotation=0)

    tick_locs = ax.get_xticks()

    ax_top = ax.twiny()
    padw = 0.5 / (end_ix - start_ix)
    ax_top.set_xticks(np.linspace(padw, 1 - padw, end_ix - start_ix))

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
):
    make_lens_hooks(model, verbose=False)

    layer_logits = collect_logits(model, input_ids)

    _plot_logit_lens(
        layer_logits=layer_logits,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        end_ix=end_ix,
        probs=probs,
    )
