from ..util.python_utils import make_print_if_verbose


class AfterStoppingPointException(Exception):
    pass


def add_partial_forward_hooks(model, verbose=True, debug=False):
    vprint = make_print_if_verbose(verbose)
    dprint = make_print_if_verbose(debug)

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
            dprint(f"reached output of {repr(this_name)}")
            dprint(f"model._output_sink_names: {model._output_sink_names}")

            if this_name in model._output_sink_names:
                dprint(f"{repr(this_name)} in sink")

                model._output_sink[this_name] = output

    def _after_stopping_point_hook(module, input) -> None:
        if hasattr(model, "_output_sink_names"):

            this_name = indices_to_names[module._identifying_index]
            dprint(f"reached input of {repr(this_name)}")

            if all([name in model._output_sink for name in model._output_sink_names]):
                dprint("have all model._output_sink_names, stopping")

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


def partial_forward(
    model,
    output_names,
    *args,
    verbose=False,
    debug=False,
    might_need_hooks=True,
    **kwargs,
):
    vprint = make_print_if_verbose(verbose)
    if might_need_hooks:
        add_partial_forward_hooks(model, verbose=verbose, debug=debug)

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

    return_val = model._output_sink
    del model._output_sink

    return return_val
