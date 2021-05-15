import json

import transformers.file_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


def fix_config_with_missing_model_type(model_name, config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model_type = config.get('model_type')

    # cf https://github.com/huggingface/transformers/blob/v4.5.1/src/transformers/models/auto/configuration_auto.py#L403
    #
    # we reproduce that logic here, but save the fixed config to the json file
    # so it will work more robustly, i.e. even if you are not using `AutoConfig`
    if model_type is None:
        for pattern, config_class in CONFIG_MAPPING.items():
            if pattern in model_name:
                config['model_type'] = config_class.model_type

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)


def get_local_path_from_huggingface_cdn(key, filename):
    archive_file = transformers.file_utils.hf_bucket_url(
        key,
        filename=filename,
    )

    resolved_archive_file = transformers.file_utils.cached_path(
        archive_file,
    )
    return resolved_archive_file


def huggingface_model_local_paths(model_name):
    config_path = get_local_path_from_huggingface_cdn(model_name, "config.json")

    fix_config_with_missing_model_type(model_name, config_path)

    model_path = get_local_path_from_huggingface_cdn(model_name, "pytorch_model.bin")

    return config_path, model_path


def normalize_inconsistent_state_dict_keys(state_dict):
    normalized = {}

    for k in state_dict.keys():
        if k.startswith("transformer."):
            normalized[k] = state_dict[k]
        else:
            normalized["transformer." + k] = state_dict[k]
    return normalized
