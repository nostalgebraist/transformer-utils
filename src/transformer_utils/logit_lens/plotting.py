from functools import partial

import torch
import numpy as np
import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
# import colorcet  # noqa

from ..util.python_utils import make_print_if_verbose

from .hooks import make_lens_hooks
from .layer_names import make_layer_names


def collect_logits(model, input_ids, layer_names, decoder_layer_names):
    model._last_resid = None

    with torch.no_grad():
        out = model(input_ids)
    del out
    model._last_resid = None

    layer_logits = np.concatenate(
        [model._layer_logits[name] for name in layer_names],
        axis=0,
    )

    return layer_logits, layer_names


def postprocess_logits(layer_logits, tokenizer, topk: int):
    
    probs = torch.softmax(torch.tensor(layer_logits), dim=-1)

    topk_logits = torch.topk(probs, topk)
    layer_preds = layer_logits.argmax(axis=-1)
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    print('layer_preds', layer_preds.shape) # [, predicted_tokens]
    print('layer_probs', layer_probs.shape)
    nested_layer_preds = topk_logits.indices.tolist()
    nested_layer_probs = topk_logits.values.tolist()
    display_preds = []

    for l, layer in enumerate(nested_layer_preds):
        current_sequence = []
        for i, group in enumerate(layer):
            current_group = ''
            for j, pred in enumerate(group):
                current_group += f"('{num2tok(int(pred), tokenizer)}' {round(nested_layer_probs[l][i][j], 4)})\n"
            current_sequence.append(current_group)

        # if l < 2 and i < 2:
        #     print(f'layer {l} group {i} layer_preds', current_group)
        #     print('nested_layer_probs[l][i][0]', nested_layer_probs[l][i][0])
        display_preds.append(current_sequence)
    display_preds = np.array(display_preds, dtype=np.object)
    print('display_preds.shape', display_preds.shape, display_preds.dtype)
    # print(display_preds)
    return layer_preds, layer_probs, display_preds


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
    display_preds,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    probs=False,
    ranks=False,
    kl=False,
    top_down=False,
    topk: int = 1,
    cmap='nipy_spectral'
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

    # _num2tok = np.vectorize(
    #     partial(num2tok, tokenizer=tokenizer, quotemark="'"), otypes=[str]
    # )
    aligned_texts = aligned_preds #  _num2tok(aligned_preds)

    to_show = to_show[::-1]

    aligned_texts = aligned_texts[::-1]

    fig = plt.figure(figsize=(2.5 * to_show.shape[1], 0.375 * topk * to_show.shape[0]))

    plot_kwargs = {"annot": display_preds[::-1], "fmt": ""}
    if kl:
        vmin, vmax = None, None

        plot_kwargs.update(
            {
                "cmap": cmap,
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
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
            }
        )

    sns.heatmap(to_show, **plot_kwargs)

    ax = plt.gca()
    # input_ids shape like [[0,1,2]] for ids 0,1,2
    input_tokens_str = [num2tok(x, tokenizer) for x in input_ids[0].cpu()]
    print(input_ids, input_tokens_str) # should be the same as the input string as broken into tokens
    if layer_names is None:
      layer_names = ["Layer {}".format(n) for n in range(to_show.shape[0])]

    ylabels = layer_names[::-1]
    ax.set_yticklabels(ylabels, rotation=0)

    starred = [
          "* " + true if pred == true else " " + true
          for pred, true in zip(
              aligned_texts[0], input_tokens_str[start_ix + 1 : end_ix + 1]
          )
      ]
    padw = 0.5 / to_show.shape[1]
    ax_top = ax.twiny()
    
    if top_down:
      ax.invert_yaxis()
      ax.set_xticklabels(starred, rotation=0)
      ax_top.set_xticks(np.linspace(padw, 1 - padw, to_show.shape[1]))
      ax_top.set_xticklabels(input_tokens_str[start_ix:end_ix], rotation=0)

    else:
      ax.set_xticklabels(input_tokens_str[start_ix:end_ix], rotation=0)
      ax_top.set_xticks(np.linspace(padw, 1 - padw, to_show.shape[1]))
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
    verbose=False,
    top_down=False,
    topk: int = 1
):
    """
    Draws "logit lens" plots, and generalizations thereof.
    For background, see
        https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
        https://jalammar.github.io/hidden-states/
    `model`, `tokenizer` and `input_ids` should be familiar from the transformers library.  Other args are
     documented below.
    `model` should be a `transformers.PreTrainedModel` with an `lm_head`, e.g. `GPTNeoForCausalLM`.
    Note that using `start_ix` and `end_ix` is not equivalent to passed an `input_ids` sliced like `input_ids[start_ix:end_ix]`.  The LM will see the entire input you pass in as `input_ids`, no matter how you set `start_ix` and `end_ix`.  These "ix" arguments only control what is _displayed_.
    The boolean arguments `probs`, `ranks` and `kl` control the type of plot.  The options are:
        - Logits (the default plot type, if `probs`, `ranks` and `kl` are all False):
            - cell color: logit assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer
        - Probabilities:
            - cell color: probability assigned by each layer to the final layer's top-1 token prediction
            - cell text:  top-1 token prediction at each layer
        - Ranks:
            - cell color: ranking over the vocab assigned by each layer to the final layer's top-1 token prediction
            - cell text:  same as cell color
        - KL:
            - cell color: KL divergence of each layer's probability distribtion w/r/t the final layer's
            - cell text:  same as cell color
    `include_subblocks` and `decoder_layer_names` allow the creation of plots that go beyond what was done
    in the original blog post.  See below for details
    Arguments:
        probs:
            draw a "Probabilities" plot
        ranks:
            draw a "Ranks" plot (overrides `probs`)
        kl:
            draw a "KL" plot (overrides `probs`, `ranks`)
        block_step:
            stride when choosing blocks to plot, e.g. block_step=2 skips every other block
        include_input:
            whether to treat the input embeddings (before any blocks have been applied) as a "layer"
        force_include_output:
            whether to include the final layer in the plot, even if the passed `block_step` would otherwise skip it
        include_subblocks:
            if True, includes predictions after the only the attention part of each block, along with those after the
            full block
        decoder_layer_names:
            defines the subset of the model used to "decode" hidden states.
            The default value `['final_layernorm', 'lm_head']` corresponds to the ordinary "logit lens," where
            we decode each layer's output as though it were the output of the final block.
            Prepending one or more of the last layers of the model, e.g. `['h11', 'final_layernorm', 'lm_head']`
            for a 12-layer model, will treat these layers as part of the decoder.  In the general case, this is equivalent
            to dropping different subsets of interior layers and watching how the output varies.
        top_down:
            reverse order so that the input is on top in the plot, and the output on the bottom
        topk:
            number of top k tokens to display per layer. Default is 1, just the top prediction.
    """
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
        model, input_ids, layer_names=layer_names, decoder_layer_names=decoder_layer_names,
    )

    layer_preds, layer_probs, display_preds = postprocess_logits(layer_logits, tokenizer, topk)

    _plot_logit_lens(
        layer_logits=layer_logits,
        layer_preds=layer_preds,
        layer_probs=layer_probs,
        display_preds=display_preds,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_ix=start_ix,
        probs=probs,
        ranks=ranks,
        kl=kl,
        layer_names=layer_names,
        top_down=top_down,
        topk=topk
    )
