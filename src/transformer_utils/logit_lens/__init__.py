from functools import partial

import torch
import torch.nn as nn
import numpy as np
import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa

from ..util.python_utils import make_print_if_verbose
from ..util.module_utils import get_child_module_by_names


def final_layernorm_locator(model: nn.Module):
    # TODO: more principled way?
    names = ["ln_f", "layernorm"]
    for name in names:
        if hasattr(model.transformer, name):
            return lambda: getattr(model.transformer, name)
    return lambda: lambda x: x


def make_lens_hooks(
    model,
    layer_names: list = None,
    prefixes: list = ["transformer"],
    verbose=True,
    extra_call_before_decoder=lambda x: x,
    start_ix=None,
    end_ix=None
):
    vprint = make_print_if_verbose(verbose)

    _RESID_SUFFIXES = {".attn", ".mlp"}

    def _sqz(x):
        if isinstance(x, torch.Tensor):
            return x
        try:
            return x[0]
        except:
            return x

    def _opt_slice(x, start_ix, end_ix):
        if not start_ix:
            start_ix = 0
        if not end_ix:
            end_ix = x.shape[1]
        return x[:, start_ix:end_ix, :]

    for attr in ["_layer_logits", "_layer_logits_handles"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    if not hasattr(model, "_last_input_ids"):
        model._last_input_ids = None
        model._last_input_ids_handle = None

    if not hasattr(model, "_ln_f_getter"):
        model._ln_f_getter = final_layernorm_locator(model)

    if not hasattr(model, "_ln_base"):
        # TODO: use
        model._ln_base = (
            nn.LayerNorm(model.config.hidden_size)
            .to(model.device)
            .requires_grad_(False)
        )

    if layer_names is None:
        h = get_child_module_by_names(model, prefixes + ["h"])
        layer_names = [f"h.{i}" for i in range(len(h))]

    # TODO: better naming
    model._ordered_layer_names = layer_names

    def _get_layer(name):
        return get_child_module_by_names(model, prefixes + name.split("."))

    def _make_record_logits_hook(name):
        model._layer_logits[name] = None

        is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])

        def _record_logits_hook(module, input, output) -> None:
            del model._layer_logits[name]
            ln_f = model._ln_f_getter()

            if is_resid:
                decoder_in = model._last_resid + _sqz(output)
            else:
                decoder_in = _sqz(output)

            slice_in = extra_call_before_decoder(decoder_in)
            sliced = _opt_slice(slice_in, start_ix, end_ix)

            decoder_out = model.lm_head(ln_f(sliced))

            model._layer_logits[name] = decoder_out.cpu().numpy()
            model._last_resid = decoder_in

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
        handle = model.transformer.get_input_embeddings().register_forward_hook(
            _record_input_ids_hook
        )
        model._last_input_ids_handle = handle


def clear_lens_hooks(model):
    if hasattr(model, "_layer_logits_handles"):
        for k, v in model._layer_logits_handles.items():
            v.remove()

        ks = list(model._layer_logits_handles.keys())
        for k in ks:
            del model._layer_logits_handles[k]

    if hasattr(model, "_last_input_ids"):
        model._last_input_ids = None


def collect_logits(model, input_ids, layer_names=None):
    needs_forward = True
    if model._last_input_ids is not None:
        if model._last_input_ids.shape == input_ids.shape:
            needs_forward = not (model._last_input_ids == input_ids).cpu().all()

    model._last_resid = None

    if needs_forward:
        with torch.no_grad():
            out = model(input_ids)
        del out
        model._last_resid = None

    if layer_names is None:
        layer_names = model._ordered_layer_names

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


def _plot_logit_lens(
    layer_logits,
    layer_preds,
    layer_probs,
    tokenizer,
    input_ids,
    start_ix,
    probs=False,
    ranks=False,
    layer_names=None,
):
    end_ix = start_ix + layer_logits.shape[1]

    final_preds = layer_preds[-1]

    numeric_input = layer_probs if probs else layer_logits

    to_show = get_value_at_preds(numeric_input, final_preds)

    if ranks:
        to_show = (numeric_input >= to_show[:, :, np.newaxis]).sum(axis=-1)

    to_show = to_show[::-1]

    aligned_preds = layer_preds[::-1]

    _num2tok = np.vectorize(
        partial(num2tok, tokenizer=tokenizer, quotemark="'"), otypes=[str]
    )
    aligned_texts = _num2tok(aligned_preds)

    fig = plt.figure(figsize=(1.5 * to_show.shape[1], 10))

    if ranks:
        vmin, vmax = 1, 100
        cmap = "Blues"
        norm = mpl.colors.LogNorm()
    elif probs:
        vmin, vmax = 0, 1
        cmap = "Blues_r"
        norm = None
    else:
        vmin, vmax = 0, layer_logits[-1, :].max()
        cmap = "cet_linear_protanopic_deuteranopic_kbw_5_98_c40"
        norm = None

    sns.heatmap(
        to_show,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        annot=True if ranks else aligned_texts,
        fmt="",
    )

    ax = plt.gca()
    input_tokens_str = _num2tok(input_ids[0].cpu())
    ax.set_xticklabels(input_tokens_str[start_ix : end_ix], rotation=0)

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
    layer_names=None,
):
    make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, verbose=False)

    layer_logits, layer_names = collect_logits(model, input_ids, layer_names=layer_names)

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
        layer_names=layer_names,
    )
