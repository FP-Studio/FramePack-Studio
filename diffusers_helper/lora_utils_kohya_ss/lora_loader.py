# https://github.com/kohya-ss/FramePack-eichi/blob/4085a24baf08d6f1c25e2de06f376c3fc132a470/webui/lora_utils/lora_loader.py
# FramePack-eichi LoRA Loader
#
# LoRAモデルの読み込みと適用のための機能を提供します。

import os
import torch
from tqdm import tqdm
from .lora_utils import merge_lora_to_state_dict


def load_and_apply_lora(
    model_files: list[str], lora_paths: list[str], lora_scales=None, fp8_enabled=False, device=None
) -> dict[str, torch.Tensor]:
    """
    LoRA重みをロードして重みに適用する

    Args:
        model_files: List of model files to load
        lora_paths: List of LoRA file paths
        lora_scales: List if LoRA weight scales
        fp8_enabled: Whether to enable FP8 optimization. Default is False.
        device: Device used for loading the model. If None, defaults to CPU.

    Returns:
        State dictionary with LoRA weights applied.
    """
    if lora_paths is None:
        lora_paths = []

    if device is None:
        device = torch.device("cpu")  # CPU fall back

    for lora_path in lora_paths:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"LoRA file not found: {lora_path}"
            )

    if lora_scales is None:
        lora_scales = [0.8] * len(lora_paths)
    if len(lora_scales) > len(lora_paths):
        lora_scales = lora_scales[: len(lora_paths)]
    if len(lora_scales) < len(lora_paths):
        lora_scales += [0.8] * (len(lora_paths) - len(lora_scales))


    for lora_path, lora_scale in zip(lora_paths, lora_scales):
        print(
            f"LoRA loading: {os.path.basename(lora_path)} (scale: {lora_scale})"
        )


    print(f"Model architecture: HunyuanVideo")

    # Merge the LoRA weighs into the state dictionary
    merged_state_dict = merge_lora_to_state_dict(
        model_files, lora_paths, lora_scales, fp8_enabled, device
    )

    print(f"LoRA loading complete")
    return merged_state_dict


def check_lora_applied(model):
    from lora_check_helper import check_lora_applied as check_lora_applied_helper
    # passthrough to the helper function - this function should be removed
    return check_lora_applied_helper(model)