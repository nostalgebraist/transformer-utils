import torch.nn

_TORCH_NN_ORIGINAL = torch.nn
_TORCH_NN_ORIGINAL_LINEAR = _TORCH_NN_ORIGINAL.Linear


class LazyLinearAPICompatible(torch.nn.LazyLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(out_features=out_features, bias=bias)


class LowMemoryLoadContext:
    def __enter__(self):
        torch.nn.Linear = LazyLinearAPICompatible

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.nn.Linear = _TORCH_NN_ORIGINAL_LINEAR
        return exc_type is not None
