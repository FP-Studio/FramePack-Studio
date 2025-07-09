# Original https://github.com/kohya-ss/FramePack-LoRAReady/blob/3613b67366b0bbf4a719c85ba9c3954e075e0e57/utils/fp8_optimization_utils.py
# Updates for eichi https://github.com/kohya-ss/FramePack-eichi/blob/4085a24baf08d6f1c25e2de06f376c3fc132a470/webui/lora_utils/fp8_optimization_utils.py

import torch.nn.functional as F
import torch
import torch.nn as nn
import os

from tqdm import tqdm
from typing import Literal, cast

# Flags to track whether a warning message was displayed.
FP8_E4M3_WARNING_SHOWN = False
FP8_DIMENSIONS_WARNING_SHOWN = False

# cSpell: ignore maxval, dequantized


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1) -> float:
    """
    Calculates the maximum value that can be expressed in FP8 format.
    The default is E4M3 format (4-bit exponent, 3-bit mantissa, 1-bit sign).

    Args:
        exp_bits (int): Number of bits in exponent
        mantissa_bits (int): Number of bits in mantissa
        sign_bits (int): Number of bits in sign (0 or 1)

    Returns:
        float: Maximum value that can be expressed in FP8 format.
    """
    assert (
        exp_bits + mantissa_bits + sign_bits == 8
    ), f"The total number of bits for FP8 must be 8, but got {exp_bits + mantissa_bits + sign_bits} bits (E{exp_bits} M{mantissa_bits} S{sign_bits})"

    # Calculate the exponent bias
    bias: int = 2 ** (exp_bits - 1) - 1

    # Calculate the maximum mantissa value
    # Maybe this can be an int?
    mantissa_max: float = 1.0
    for i in range(mantissa_bits - 1):
        mantissa_max += 2 ** -(i + 1)

    # Calculate the maximum value
    max_value = mantissa_max * (2 ** (2**exp_bits - 1 - bias))

    return cast(float, max_value)


def quantize_tensor_to_fp8(
    tensor: torch.Tensor,
    scale: float | torch.Tensor,
    exp_bits: int = 4,
    mantissa_bits: int = 3,
    sign_bits: int = 1,
    max_value: float | None = None,
    min_value: float | None = None,
):
    """
    Quantize the tensor to FP8 format

    Args:
        tensor (torch.Tensor): The tensor to quantize.
        scale (float or torch.Tensor): Scale factor.
        exp_bits (int): Number of bits in exponent.
        mantissa_bits (int): Number of bits in mantissa.
        sign_bits (int): Number of bits in sign.
        max_value (float, optional): Maximum value (automatically calculated if None).
        min_value (float, optional): Minimum value (automatically calculated if None).

    Returns:
        tuple: (quantized tensor, scale factor)
    """
    # スケーリングされたテンソルを作成
    scaled_tensor = tensor / scale

    # FP8パラメータを計算
    bias: int = 2 ** (exp_bits - 1) - 1

    if max_value is None:
        # 最大値と最小値を計算
        max_value = calculate_fp8_maxval(exp_bits, mantissa_bits, sign_bits)
        min_value = -max_value if sign_bits > 0 else 0.0

    # テンソルを範囲内に制限
    clamped_tensor = torch.clamp(scaled_tensor, min_value, max_value)

    # 量子化プロセス
    abs_values = torch.abs(clamped_tensor)
    nonzero_mask = abs_values > 0

    # logFスケールを計算（非ゼロ要素のみ）
    log_scales = torch.zeros_like(clamped_tensor)
    if nonzero_mask.any():
        log_scales[nonzero_mask] = torch.floor(
            torch.log2(abs_values[nonzero_mask]) + bias
        ).detach()

    # logスケールを制限し、量子化係数を計算
    log_scales = torch.clamp(log_scales, min=1.0)
    quant_factor = 2.0 ** (log_scales - mantissa_bits - bias)

    # 量子化と逆量子化
    quantized = torch.round(clamped_tensor / quant_factor) * quant_factor

    return quantized, scale


def optimize_state_dict_with_fp8_on_the_fly(
    model_files,
    calc_device,
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits: Literal[4, 5] = 4,
    mantissa_bits: Literal[2, 3] = 3,
    move_to_device=False,
    weight_hook=None,
):
    """
    Optimize linear layer weights in model state dictionary to FP8 format

    Args:
        model_files (list): List of model files to optimize (updates as they are read)
        calc_device (str): Device to quantize tensors to
        target_layer_keys (list, optional): Pattern of layer keys to target (all linear layers if None)
        exclude_layer_keys (list, optional): Pattern of layer keys to exclude
        exp_bits (int): Number of exponent bits. Valid values are 4 or 5. If 4 then mantissa_bits must be 3, if 5 then mantissa_bits must be 2.
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Whether to move optimized tensors to compute device
        weight_hook (callable, optional): Weight hook function (None if not used), applied to all weights before FP8 optimization, regardless of whether they are FP8 optimized or not.

    Returns:
        dict: FP8 optimized state dictionary
    """
    # Select FP8 data type
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 formats: E{exp_bits} M{mantissa_bits}")

    # Calculate the maximum value of FP8
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # この関数は符号付きFP8のみサポート

    # Create an optimized state dictionary
    def is_target_key(key):
        # Check if weight key matches include pattern and doesn't match exclude pattern
        is_target = (
            target_layer_keys is None
            or any(pattern in key for pattern in target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(
            pattern in key for pattern in exclude_layer_keys
        )
        is_target = is_target and not is_excluded
        return is_target

    # Optimized layer counter
    optimized_count = 0

    from . import MemoryEfficientSafeOpen

    state_dict = {}

    # Process each model file using MemoryEfficientSafeOpen
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            keys = f.keys()
            for key in tqdm(
                keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"
            ):
                value = f.get_tensor(key)
                if weight_hook is not None:
                    # If a weight hook is specified, apply the hook
                    value = weight_hook(key, value)

                if not is_target_key(key):
                    state_dict[key] = value
                    continue

                # Preserve original device and data type
                original_device = value.device
                original_dtype = value.dtype

                # Move to specified compute device
                if calc_device is not None:
                    value = value.to(calc_device)

                # Calculate the scale factor
                scale = torch.max(torch.abs(value.flatten())) / max_value

                # Quantize weights to FP8
                quantized_weight, _ = quantize_tensor_to_fp8(
                    value, scale, exp_bits, mantissa_bits, 1, max_value, min_value
                )

                # Use original key for weight, new key for scale
                fp8_key = key
                scale_key = key.replace(".weight", ".scale_weight")

                # Convert to FP8 data type
                quantized_weight = quantized_weight.to(fp8_dtype)

                # If no device is specified, revert to original device
                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                # Create a scale tensor
                scale_tensor = torch.tensor(
                    [scale], dtype=original_dtype, device=quantized_weight.device
                )

                # Add to state dictionary
                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor

                optimized_count += 1

                # Periodically free up memory
                if calc_device is not None and optimized_count % 10 == 0:
                    torch.cuda.empty_cache()

    print(f"Optimized Linear Layer Count: {optimized_count}")
    return state_dict


def fp8_linear_forward_patch(
    self: nn.Linear,
    x: torch.Tensor,
    use_scaled_mm: bool = False,
    max_value: float | None = None,
):
    """
    Patched forward method for linear layers with FP8 weights

    Args:
        self: Linear layer instance
        x (torch.Tensor): Input tensor
        use_scaled_mm (bool): Whether to use scaled_mm for FP8 linear layers (requires SM 8.9+, RTX 40 series)
        max_value (float): Maximum FP8 quantization (if None, no quantization is applied to the input tensor)

    Returns:
        torch.Tensor: Result of the linear transformation
    """
    if use_scaled_mm:
        # If you use scaled_mm (only works with RTX >= 40 series GPUs)
        input_dtype = x.dtype
        original_weight_dtype = cast(torch.dtype, self.scale_weight.dtype)
        weight_dtype = self.weight.dtype
        target_dtype = torch.float8_e5m2

        # Falls back to normal method if not E4M3FN
        # scaled_mm is only compatible with E4M3FN format even in FP8, so cannot be used with other formats
        if weight_dtype != torch.float8_e4m3fn:
            # may be noisy
            print(
                f"WARNING: scaled_mm requires FP8 E4M3FN format but {weight_dtype} was detected, falling back to regular method."
            )

            # fallback to normal method
            return fp8_linear_forward_patch(self, x, False, max_value)

        # Check the dimensions of the input tensor
        # scaled_mm expects a 3-dimensional tensor (batch_size, seq_len, hidden_dim), otherwise it will not work
        if x.ndim != 3:
            # may be noisy
            print(
                f"Warning: scaled_mm expects 3D input but found {x.ndim} dimensions. Falling back to normal method."
            )

            # fallback to normal method
            return fp8_linear_forward_patch(self, x, False, max_value)

        if max_value is None:
            # No input quantization, use scale of 1.0
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # Calculate the scale factor of the input tensor
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # Quantize input tensors to FP8 (can be memory intensive)
            x, _ = quantize_tensor_to_fp8(x, scale_x, 5, 2, 1, max_value, -max_value)

        original_shape = x.shape
        # Change the shape of the tensor to 2D
        x = x.reshape(-1, x.shape[2]).to(target_dtype)

        # Transpose the weights
        weight = self.weight.t()
        scale_weight = cast(torch.Tensor, self.scale_weight.to(torch.float32))

        # separate processing with and without biasing
        if self.bias is not None:
            # If biased then float32 is not supported
            o = torch._scaled_mm(
                x,
                weight,
                out_dtype=original_weight_dtype,
                bias=self.bias,
                scale_a=scale_x,
                scale_b=scale_weight,
            )
        else:
            o = torch._scaled_mm(
                x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight
            )

        # Return original shape
        return o.reshape(original_shape[0], original_shape[1], -1).to(input_dtype)
    else:
        # calculate by inverse quantization of weights
        original_dtype = cast(torch.dtype, self.scale_weight.dtype)
        dequantized_weight = self.weight.to(original_dtype) * cast(
            torch.Tensor, self.scale_weight
        )

        # Perform a linear transformation
        if self.bias is not None:
            output = F.linear(x, dequantized_weight, self.bias)
        else:
            output = F.linear(x, dequantized_weight)

        return output


def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """

    # Calculate FP8 float8_e5m2 max value
    max_value = None

    # Find all scale keys to identify FP8-optimized layers
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # Enumerate patched layers
    patched_module_paths = set()
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

    patched_count = 0

    # Apply monkey patch to each layer with FP8 weights
    for name, module in model.named_modules():
        # Check if this module has a corresponding scale_weight
        has_scale = name in patched_module_paths

        # Apply patch if it's a Linear layer with FP8 scale
        if isinstance(module, nn.Linear) and has_scale:
            # register the scale_weight as a buffer to load the state_dict
            module.register_buffer(
                "scale_weight", torch.tensor(1.0, dtype=module.weight.dtype)
            )

            # Create a new forward method with the patched version.
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            # Bind method to module
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    print(f"Number of monkey-patched Linear layers: {patched_count}")
    setattr(model, "_fp8_optimized", True)
    return model


def check_fp8_support():
    """
    Checks if the current PyTorch version supports FP8 formats and scaled_mm.

    Returns:
        tuple[bool, bool, bool]: (E4M3 support, E5M2 support, scaled_mm support)
    """

    has_e4m3 = hasattr(torch, "float8_e4m3fn")
    has_e5m2 = hasattr(torch, "float8_e5m2")

    has_scaled_mm = hasattr(torch, "_scaled_mm")

    if has_e4m3 and has_e5m2:
        print("FP8 support detected: E4M3 and E5M2 formats available")
        if has_scaled_mm:
            print(
                "scaled_mm support detected: FP8 acceleration possible on RTX >=40 series GPUs"
            )
    else:
        print("WARNING: No FP8 support detected. PyTorch 2.1 or higher required")

    return has_e4m3, has_e5m2, has_scaled_mm
