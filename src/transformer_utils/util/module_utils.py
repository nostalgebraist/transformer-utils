from .python_utils import make_print_if_verbose


def get_child_module_by_names(module, names):
    obj = module
    for getter in map(lambda name: lambda obj: getattr(obj, name), names):
        obj = getter(obj)
    return obj


def get_leaf_modules(module, verbose=False):
    vprint = make_print_if_verbose(verbose)

    names = []
    leaves = []
    handled = set()

    for param_name in dict(module.named_parameters()).keys():
        mod_name = param_name.rpartition(".")[0]
        mod = get_child_module_by_names(module, mod_name.split("."))

        if mod_name in handled:
            continue

        vprint((param_name, mod_name, mod))

        names.append(mod_name)
        leaves.append(mod)
        handled.add(mod_name)

    return names, leaves
