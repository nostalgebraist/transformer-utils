import transformers

from .load import low_memory_load
from ..util.tfm_utils import huggingface_model_local_paths

_TFM_PRETRAINED_MODEL_FROM_PRETRAINED_ORIGINAL = transformers.modeling_utils.PreTrainedModel.from_pretrained


def low_memory_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
    config_path, model_path = huggingface_model_local_paths(pretrained_model_name_or_path)

    model = low_memory_load(config_path=config_path, model_path=model_path, verbose=False)

    return model


def enable_low_memory_load():
    transformers.modeling_utils.PreTrainedModel.from_pretrained = low_memory_from_pretrained


def disable_low_memory_load():
    transformers.modeling_utils.PreTrainedModel.from_pretrained = _TFM_PRETRAINED_MODEL_FROM_PRETRAINED_ORIGINAL
