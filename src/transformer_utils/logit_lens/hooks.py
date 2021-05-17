import torch
import torch.nn as nn

from ..util.python_utils import make_print_if_verbose
from ..util.module_utils import get_child_module_by_names


def blocks_input_locator(model: nn.Module):
    """
    HF usually (always?) places a dropout after the input embeddings.
    TODO: avoid depending on this
    """
    dropouts_on_base_model = [
        mod for mod in model.base_model.children()
        if isinstance(mod, nn.Dropout)
    ]
    if len(dropouts_on_base_model) > 0:
        return lambda: dropouts_on_base_model[0]
    raise ValueError('could not identify blocks input')


def final_layernorm_locator(model: nn.Module):
    layernorms_on_base_model = [
        mod for mod in model.base_model.children()
        if isinstance(mod, nn.LayerNorm)
    ]
    if len(layernorms_on_base_model) > 0:
        return lambda: layernorms_on_base_model[0]
    raise ValueError('could not identify ln_f')


def _locate_special_modules(model):
    if not hasattr(model, "_blocks_input_getter"):
        model._blocks_input_getter = blocks_input_locator(model)

    if not hasattr(model, "_ln_f_getter"):
        model._ln_f_getter = final_layernorm_locator(model)


def _get_layer(model, name):
    if name == "input":
        return model._blocks_input_getter()
    if name == "final_layernorm":
        return model._ln_f_getter()
    return get_child_module_by_names(model.base_model, name.split("."))


def _sqz(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x[0]
    except:
        return x


def make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head']):
    _locate_special_modules(model)

    decoder_layers = [_get_layer(model, name) for name in decoder_layer_names]

    def _decoder(x):
        for layer in decoder_layers:
            x = _sqz(layer(_sqz(x)))
        return x
    return _decoder


def make_lens_hooks(
    model,
    layer_names: list,
    decoder_layer_names: list = ['final_layernorm', 'lm_head'],
    verbose=True,
    start_ix=None,
    end_ix=None,
):
    vprint = make_print_if_verbose(verbose)

    _RESID_SUFFIXES = {".attn", ".mlp"}

    def _opt_slice(x, start_ix, end_ix):
        if not start_ix:
            start_ix = 0
        if not end_ix:
            end_ix = x.shape[1]
        return x[:, start_ix:end_ix, :]

    _locate_special_modules(model)

    for attr in ["_layer_logits", "_layer_logits_handles"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    if not hasattr(model, "_last_input_ids"):
        model._last_input_ids = None
        model._last_input_ids_handle = None

    # TODO: better naming
    model._ordered_layer_names = layer_names

    model._lens_decoder = make_decoder(model, decoder_layer_names)

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

            decoder_out = model._lens_decoder(decoder_in)
            decoder_out = _opt_slice(decoder_out, start_ix, end_ix)

            model._layer_logits[name] = decoder_out.cpu().numpy()
            model._last_resid = decoder_in

        return _record_logits_hook

    def _record_input_ids_hook(module, input, output):
        model._last_input_ids = input[0]

    def _hook_already_there(name):
        handle = model._layer_logits_handles.get(name)
        if not handle:
            return False
        layer = _get_layer(model, name)
        return handle.id in layer._forward_hooks

    for name in layer_names:
        if _hook_already_there(name):
            vprint(f"skipping layer {name}, hook already exists")
            continue
        layer = _get_layer(model, name)
        handle = layer.register_forward_hook(_make_record_logits_hook(name))
        model._layer_logits_handles[name] = handle

    if model._last_input_ids_handle is None:
        handle = model.base_model.get_input_embeddings().register_forward_hook(
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
