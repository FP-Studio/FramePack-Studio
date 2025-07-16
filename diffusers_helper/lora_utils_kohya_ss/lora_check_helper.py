# https://github.com/kohya-ss/FramePack-eichi/blob/4085a24baf08d6f1c25e2de06f376c3fc132a470/webui/lora_utils/lora_check_helper.py
# FramePack-eichi LoRA Check Helper
#
# LoRAの適用状態確認のための機能を提供します。

import torch


def check_lora_applied(model):
    """
    Check if the model has LoRA applied.
    This function checks if LoRA is applied to the model either through a direct flag or by checking for LoRA hooks in the model's modules.

    Args:
        model: Target model to check.

    Returns:
        (bool, str): If the model has a '_lora_applied' flag, it returns True and the source as 'direct_application'. If LoRA hooks are found in the model's named modules, it returns True and the source as 'hooks'. Otherwise, it returns False and 'none'.
    """

    has_flag = hasattr(model, "_lora_applied") and model._lora_applied

    if has_flag:
        return True, "direct_application"

    # Check the named modules of the model for LoRA hooks
    has_hooks = False
    for name, module in model.named_modules():
        if hasattr(module, "_lora_hooks"):
            has_hooks = True
            break

    if has_hooks:
        return True, "hooks"

    return False, "none"


def analyze_lora_application(model):
    """
    モデルのLoRA適用率と影響を詳細に分析

    Args:
        model: 分析対象のモデル

    Returns:
        dict: 分析結果の辞書
    """
    total_params = 0
    lora_affected_params = 0

    # トータルパラメータ数とLoRAの影響を受けるパラメータ数をカウント
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            param_count = module.weight.numel()
            total_params += param_count

            # LoRA適用されたモジュールかチェック
            if hasattr(module, "_lora_hooks") or hasattr(module, "_lora_applied"):
                lora_affected_params += param_count

    application_rate = 0.0
    if total_params > 0:
        application_rate = lora_affected_params / total_params * 100.0

    return {
        "total_params": total_params,
        "lora_affected_params": lora_affected_params,
        "application_rate": application_rate,
        "has_lora": lora_affected_params > 0,
    }


def print_lora_status(model):
    """
    モデルのLoRA適用状況を出力

    Args:
        model: 出力対象のモデル
    """
    has_lora, source = check_lora_applied(model)

    if has_lora:
        print("LoRA status: applied")
        print(f"LoRA model: {source}")

        # 詳細な分析
        analysis = analyze_lora_application(model)
        application_rate = analysis["application_rate"]

        print(
            f'LoRA conditions: {analysis["lora_affected_params"]}/{analysis["total_params"]} parameters ({application_rate:.2f}%)'
        )
    else:
        print("LoRA status: Not applicable")
        print("LoRA model: not applicable")
