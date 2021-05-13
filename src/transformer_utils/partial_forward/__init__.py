import torch
import torch.nn as nn

from ..util.python_utils import make_print_if_verbose


class AfterStoppingPointException(Exception):
    pass


def _named_modules_in_call_chain(module):
    names, mods = [], []
    for name, mod in module.named_modules():
        if not getattr(mod, "_call_order_index", None):
            continue
        names.append(name)
        mods.append(mod)
    return names, mods


def discover_call_order(
    model: nn.Module,
    example_input=[[0]],
    verbose=True
):
    vprint = make_print_if_verbose(verbose)

    if all([hasattr(mod, "_call_order_index") for _, mod in model.named_modules()]):
        vprint("call order already known, skipping forward pass")
        return

    model._call_counter = 0

    def _record_call_hook(module, input, output) -> None:
        current_count = model._call_counter
        module._call_order_index = current_count
        model._call_counter += 1

    record_call_handles = []

    for name, mod in model.named_modules():
        rc_handle = mod.register_forward_hook(_record_call_hook)
        record_call_handles.append(rc_handle)

    device = list(model.parameters())[0].device
    example_input_th = torch.as_tensor(example_input).to(device)

    model(example_input_th)

    del model._call_counter
    for handle in record_call_handles:
        handle.remove()


def add_stopping_point_hooks(model, verbose=True, debug=False):
    vprint = make_print_if_verbose(verbose)
    dprint = make_print_if_verbose(debug)

    discover_call_order(model, verbose=verbose)

    names, mods = _named_modules_in_call_chain(model)

    if all([hasattr(mod, "_after_stopping_point_handle") for mod in mods]):
        # not a complete check, but should cover normal situations
        vprint("stopping point hooks already there, skipping")
        return

    indices_to_names = {mod._call_order_index: name for name, mod in zip(names, mods)}

    def _record_to_sink_hook(module, input, output) -> None:
        if hasattr(model, "_output_sink_names"):
            this_name = indices_to_names[module._call_order_index]
            dprint(f'reached output of {repr(this_name)}')
            dprint(f'model._output_sink_names: {model._output_sink_names}')
            if this_name in model._output_sink_names:
                dprint(f'{repr(this_name)} in sink')
                model._output_sink[this_name] = output

    def _after_stopping_point_hook(module, input) -> None:
        if hasattr(model, "_stopping_point"):
            this_name = indices_to_names[module._call_order_index]
            dprint(f'reached input of {repr(this_name)}')
            dprint(f'_call_order_index {module._call_order_index} vs _stopping_point {model._stopping_point}')
            if module._call_order_index > model._stopping_point:
                dprint('stopping')
                raise AfterStoppingPointException

    for name, mod in zip(names, mods):
        if hasattr(mod, "_record_to_sink_handle"):
            vprint(f"clearing existing handle at {repr(name)}")
            mod._record_to_sink_handle.remove()

        rts_handle = mod.register_forward_hook(_record_to_sink_hook)
        mod._record_to_sink_handle = rts_handle

        if hasattr(mod, "_after_stopping_point_handle"):
            vprint(f"clearing existing handle at {repr(name)}")
            mod._after_stopping_point_handle.remove()

        asp_handle = mod.register_forward_pre_hook(_after_stopping_point_hook)
        mod._after_stopping_point_handle = asp_handle


def last_name_with_prefix(names_to_indices, prefix):
    if prefix in names_to_indices:
        return prefix

    last_ix = max(
        [ix for name, ix in names_to_indices.items() if name.startswith(prefix)]
    )

    indices_to_names = {v: k for k, v in names_to_indices.items()}
    return indices_to_names[last_ix]


def partial_forward(model, output_names, *args, verbose=False, might_need_hooks=True, **kwargs,):
    vprint = make_print_if_verbose(verbose)
    if might_need_hooks:
        add_stopping_point_hooks(model, verbose=verbose)

    names, mods = _named_modules_in_call_chain(model)

    names_to_indices = {name: mod._call_order_index for name, mod in zip(names, mods)}

    model._stopping_point = max([names_to_indices[name] for name in output_names])
    model._output_sink_names = output_names

    if hasattr(model, "_output_sink"):
        vprint("clearing existing _output_sink")
        for v in model._output_sink.values():
            del v
        del model._output_sink

    model._output_sink = {}

    try:
        model.forward(*args, **kwargs)
    except AfterStoppingPointException:
        pass

    del model._stopping_point

    return model._output_sink
