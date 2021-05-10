def make_print_if_verbose(verbose: bool):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return vprint


def reverse_dict(d: dict):
    # TODO: validation
    return {v: k for k, v in d.items()}
