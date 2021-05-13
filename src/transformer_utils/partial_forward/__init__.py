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

    model._call_counter = 0

    def _record_call_hook(module, input, output) -> None:
        current_count = model._call_counter
        module._call_order_index = current_count
        model._call_counter += 1

    def _after_stopping_point_hook(module, input, output) -> None:
        if hasattr(model, "_stopping_point"):
            if module._call_order_index == model._stopping_point:
                model._output_sink = output
            if module._call_order_index > model._stopping_point:
                raise AfterStoppingPointException

    record_call_handles = []

    for name, leaf in zip(names, leaves):
        rc_handle = leaf.register_forward_hook(_record_call_hook)
        record_call_handles.append(rc_handle)

        if hasattr(leaf, "_after_stopping_point_handle"):
            vprint(f"clearing existing handle at {repr(name)}")
            leaf._after_stopping_point_handle.remove()

    if all([hasattr(leaf, "_call_order_index") for leaf in leaves]):
        vprint("call order already known, skipping")
    else:
        device = list(model.parameters())[0].device
        example_input_th = torch.as_tensor(example_input).to(device)

        model.forward(example_input_th)

    del model._call_counter
    for handle in record_call_handles:
        handle.remove()

    for name, leaf in zip(names, leaves):
        asp_handle = leaf.register_forward_hook(_after_stopping_point_hook)
