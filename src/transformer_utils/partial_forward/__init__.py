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

    if all([hasattr(mod, "_call_order_index")
            and hasattr(mod, "_return_order_index")
            for _, mod in model.named_modules()]):
        vprint("call order already known, skipping forward pass")
        return

    model._call_counter = 0
    model._return_counter = 0

    def _record_call_hook(module, input) -> None:
        current_call_count = model._call_counter
        module._call_order_index = current_call_count
        model._call_counter += 1

    def _record_return_hook(module, input, output) -> None:
        current_return_count = model._return_counter
        module._return_order_index = current_return_count
        model._return_counter += 1

    record_call_handles = []
    record_return_handles = []

    for name, mod in model.named_modules():
        rc_handle = mod.register_forward_pre_hook(_record_call_hook)
        record_call_handles.append(rc_handle)

        rr_handle = mod.register_forward_hook(_record_return_hook)
        record_return_handles.append(rr_handle)

    device = list(model.parameters())[0].device
    example_input_th = torch.as_tensor(example_input).to(device)

    model(example_input_th)

    del model._call_counter
    del model._return_counter
    for handle in record_call_handles + record_return_handles:
        handle.remove()


def add_partial_forward_hooks(model, verbose=True, debug=False):
    vprint = make_print_if_verbose(verbose)
    dprint = make_print_if_verbose(debug)

    # discover_call_order(model, verbose=verbose)

    names_to_mods = {}
    indices_to_names = {}
    names, mods = [], []
    for i, (name, mod) in enumerate(model.named_modules()):
        mod._identifying_index = i
        indices_to_names[i] = name

        names.append(name)
        mods.append(mod)
        names_to_mods[name] = mod

    def _record_to_sink_hook(module, input, output) -> None:
        if hasattr(model, "_output_sink_names"):
            this_name = indices_to_names[module._identifying_index]
            dprint(f'reached output of {repr(this_name)}')
            dprint(f'model._output_sink_names: {model._output_sink_names}')
            if this_name in model._output_sink_names:
                dprint(f'{repr(this_name)} in sink')
                model._output_sink[this_name] = output

    def _after_stopping_point_hook(module, input) -> None:
        if hasattr(model, "_output_sink_names"):
            this_name = indices_to_names[module._identifying_index]
            dprint(f'reached input of {repr(this_name)}')
            if all([name in model._output_sink for name in model._output_sink_names]):
                dprint('have all model._output_sink_names, stopping')
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


def partial_forward(model, output_names, *args, verbose=False, might_need_hooks=True, **kwargs,):
    vprint = make_print_if_verbose(verbose)
    if might_need_hooks:
        add_partial_forward_hooks(model, verbose=verbose)

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

    del model._output_sink_names

    return model._output_sink
