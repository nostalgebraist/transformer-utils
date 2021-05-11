import transformers.file_utils


def get_local_path_fom_huggingface_cdn(key, filename):
    archive_file = transformers.file_utils.hf_bucket_url(
        key,
        filename=filename,
    )

    resolved_archive_file = transformers.file_utils.cached_path(
        archive_file,
    )
    return resolved_archive_file


def huggingface_model_local_paths(model_name):
    config_path = get_local_path_fom_huggingface_cdn(model_name, "config.json")

    model_path = get_local_path_fom_huggingface_cdn(model_name, "pytorch_model.bin")

    return config_path, model_path


def normalize_inconsistent_state_dict_keys(state_dict):
    normalized = {}

    for k in state_dict.keys():
        if k.startswith("transformer."):
            normalized[k] = state_dict[k]
        else:
            normalized["transformer." + k] = state_dict[k]
    return normalized
