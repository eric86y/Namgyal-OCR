import os
from glob import glob
from huggingface_hub import snapshot_download


MODEL_DICT = {
    "Woodblock": "BDRC/Woodblock",
    "UCHAN": "BDRC/BigUCHAN_v1",
    "DergeTenjur": "BDRC/DergeTenjur",
    "GoogleBooks_C": "BDRC/GoogleBooks_C_v1",
    "GoogleBooks_E": "BDRC/GoogleBooks_E_v1",
    "Norbuketaka_C": "BDRC/Norbuketaka_C_V1",
    "Norbuketaka_E": "BDRC/Norbuketaka_E_V1",
    "Drutsa-A_E": "BDRC/Drutsa-A_E_v1",
    "Namgyal": "Eric-23xd/EarlyTibetan-Manuscript-Uchan"
}

LAYOUT_COLORS = {
                "background": "0, 0, 0",
                "image": "45, 255, 0",
                "line": "255, 100, 0",
                "margin": "255, 0, 0",
                "caption": "255, 100, 243"
}


# download the line model: https://huggingface.co/BDRC/PhotiLines
def init_line_model() -> str:
    model_id = "BDRC/PhotiLines"
    model_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=f"Models/{model_id}",
    )
    model_config = f"{model_path}/config.json"
    assert os.path.isfile(model_config)

    return model_config


# download the layout model: https://huggingface.co/BDRC/Photi
def init_layout_model() -> str:
    model_id = "BDRC/Photi-v2"
    model_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=f"Models/{model_id}",
    )

    model_config = f"{model_path}/config.json"
    assert os.path.isfile(model_config)

    return model_config


def init_ocr_model(identifier: str) -> str:
    available_models = list(MODEL_DICT.keys())

    if identifier in available_models:
        model_id = MODEL_DICT[identifier]

        model_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_dir=f"Models/{model_id}",
        )

        model_config = glob(f"{model_path}/*.json")[0]
        assert os.path.isfile(model_config)

        return model_config
    else:
        print(f"Error: {identifier} is not available")
        return None
