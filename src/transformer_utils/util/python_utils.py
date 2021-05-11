def make_print_if_verbose(verbose: bool):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    return vprint
