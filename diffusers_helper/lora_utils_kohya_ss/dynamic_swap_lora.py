# FramePack-eichi Dynamic Swap LoRA
#
# このモジュールは後方互換性のために残されていますが、
# 実際にはdirect_applicationによるLoRA適用が使用されています。
# https://github.com/kohya-ss/FramePack-eichi/blob/4085a24baf08d6f1c25e2de06f376c3fc132a470/webui/lora_utils/dynamic_swap_lora.py

import os
import torch
import warnings

# 国際化対応
from locales.i18n_extended import translate as _


class DynamicSwapLoRAManager:
    """
    この旧式のLoRA管理クラスは後方互換性のために残されていますが、
    実際の処理では使用されません。代わりに直接的なLoRA適用が行われます。
    """

    def __init__(self):
        """初期化"""
        self.is_active = False
        self.lora_path = None
        self.lora_scale = 0.8
        warnings.warn(
            _(
                "DynamicSwapLoRAManagerは非推奨です。代わりにlora_loader.load_and_apply_lora()を使用してください。"
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    def load_lora(self, lora_path, is_diffusers=False):
        """
        LoRAファイルをロードする (実際には、パスの記録のみ)

        Args:
            lora_path: LoRAファイルのパス
            is_diffusers: 互換性のために残されたパラメータ（使用されない）
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                _("LoRAファイルが見つかりません: {0}").format(lora_path)
            )

        self.lora_path = lora_path
        self.is_active = True

        print(
            _("LoRAファイルがロードされました (非推奨インターフェース): {0}").format(
                lora_path
            )
        )
        print(
            _("注意: ")
            + _(
                "DynamicSwapLoRAManagerは非推奨です。代わりにlora_loader.load_and_apply_lora()を使用してください。"
            )
        )

    def set_scale(self, scale):
        """
        LoRA適用スケールを設定する

        Args:
            scale: LoRAの適用強度
        """
        self.lora_scale = scale

    def install_hooks(self, model):
        """
        モデルにLoRAフックをインストールする (実際には、直接適用を行う)

        Args:
            model: フックをインストールするモデル
        """
        # 直接適用モードを使用してLoRAを適用
        from .lora_loader import load_and_apply_lora

        print(
            _(
                "警告: DynamicSwapLoRAManagerは非推奨です。直接適用モードにリダイレクトします。"
            )
        )

        load_and_apply_lora(
            model,
            self.lora_path,
            self.lora_scale,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        print(_("LoRAは直接適用モードで適用されました。"))
