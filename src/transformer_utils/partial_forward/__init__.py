from .hooks import discover_call_order


import torch
import torch.nn as nn

from ..util.python_utils import make_print_if_verbose
from ..util.module_utils import get_leaves


class AfterStoppingPointException(Exception):
    pass


def discover_call_order(
    model: nn.Module,
    example_input=[[0]],
    verbose=True
):
    vprint = make_print_if_verbose(verbose)

    names, leaves = get_leaves(model)

    if all([hasattr(leaf, "_call_order_index") for leaf in leaves]):
        vprint("call order already known, skipping forward pass")
        return

    model._call_counter = 0

    def _record_call_hook(module, input, output) -> None:
        current_count = model._call_counter
        module._call_order_index = current_count
        model._call_counter += 1

    record_call_handles = []

    for name, leaf in zip(names, leaves):
        rc_handle = leaf.register_forward_hook(_record_call_hook)
        record_call_handles.append(rc_handle)

    device = list(model.parameters())[0].device
    example_input_th = torch.as_tensor(example_input).to(device)

    model.forward(example_input_th)

    del model._call_counter
    for handle in record_call_handles:
        handle.remove()


def add_stopping_point_hooks(model, verbose=True):
    vprint = make_print_if_verbose(verbose)

    names, leaves = get_leaves(model)

    discover_call_order(model, verbose=verbose)
    indices_to_names = {leaf._call_order_index: name for name, leaf in zip(names, leaves)}

    if hasattr(model, "_output_sink"):
        vprint("clearing existing _output_sink")
        for v in model._output_sink.values():
            del v
        del model._output_sink

    model._output_sink = {}

    def _record_to_sink_hook(module, input, output) -> None:
        if hasattr(model, "_output_sink_names"):
            this_name = indices_to_names[module._call_order_index]
            if this_name in model._output_sink_names:
                model._output_sink[this_name] = output

    def _after_stopping_point_hook(module, input, output) -> None:
        if hasattr(model, "_stopping_point"):
            if module._call_order_index > model._stopping_point:
                raise AfterStoppingPointException

    for name, leaf in zip(names, leaves):
        if hasattr(leaf, "_record_to_sink_handle"):
            vprint(f"clearing existing handle at {repr(name)}")
            leaf._record_to_sink_handle.remove()

        rts_handle = leaf.register_forward_hook(_record_to_sink_hook)
        leaf._record_to_sink_handle = rts_handle

        if hasattr(leaf, "_after_stopping_point_handle"):
            vprint(f"clearing existing handle at {repr(name)}")
            leaf._after_stopping_point_handle.remove()

        asp_handle = leaf.register_forward_hook(_after_stopping_point_hook)


def partial_forward(model, output_names, *args, **kwargs):
    names, leaves = get_leaves(model)

    names_to_indices = {name: leaf._call_order_index for name, leaf in zip(names, leaves)}
    model._stopping_point = max([names_to_indices[name] for name in output_names])

    model._output_sink_names = output_names

    try:
        model.forward(*args, **kwargs);
    except AfterStoppingPointException:
        pass
