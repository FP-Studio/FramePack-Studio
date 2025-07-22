import torch

from diffusers_helper.utils import save_bcthw_as_mp4


@torch.no_grad()
def combine_videos_sequentially_from_tensors(
    processed_input_frames_np, generated_frames_pt, output_path, target_fps, crf_value
):
    """
    Combines processed input frames (NumPy) with generated frames (PyTorch Tensor) sequentially
    and saves the result as an MP4 video using save_bcthw_as_mp4.

    Args:
        processed_input_frames_np: NumPy array of processed input frames (T_in, H, W_in, C), uint8.
        generated_frames_pt: PyTorch tensor of generated frames (B_gen, C_gen, T_gen, H, W_gen), float32 [-1,1].
                             (This will be history_pixels from worker.py)
        output_path: Path to save the combined video.
        target_fps: FPS for the output combined video.
        crf_value: CRF value for video encoding.

    Returns:
        Path to the combined video, or None if an error occurs.
    """
    try:
        # 1. Convert processed_input_frames_np to PyTorch tensor BCTHW, float32, [-1,1]
        # processed_input_frames_np shape: (T_in, H, W_in, C)
        input_frames_pt = (
            torch.from_numpy(processed_input_frames_np).float() / 127.5 - 1.0
        )  # (T,H,W,C)
        input_frames_pt = input_frames_pt.permute(3, 0, 1, 2)  # (C,T,H,W)
        input_frames_pt = input_frames_pt.unsqueeze(0)  # (1,C,T,H,W) -> BCTHW

        # Ensure generated_frames_pt is on the same device and dtype for concatenation
        input_frames_pt = input_frames_pt.to(
            device=generated_frames_pt.device, dtype=generated_frames_pt.dtype
        )

        # 2. Dimension Check (Heights and Widths should match)
        #    They should match, since the input frames should have been processed to match the generation resolution.
        #    But sanity check to ensure no mismatch occurs when the code is refactored.
        if input_frames_pt.shape[3:] != generated_frames_pt.shape[3:]:  # Compare (H,W)
            print(
                f"Warning: Dimension mismatch for sequential combination! Input: {input_frames_pt.shape[3:]}, Generated: {generated_frames_pt.shape[3:]}."
            )
            print(
                "Attempting to proceed, but this might lead to errors or unexpected video output."
            )
            # Potentially add resizing logic here if necessary, but for now, assume they match

        # 3. Concatenate Tensors along the time dimension (dim=2 for BCTHW)
        combined_video_pt = torch.cat([input_frames_pt, generated_frames_pt], dim=2)

        # 4. Save
        save_bcthw_as_mp4(combined_video_pt, output_path, fps=target_fps, crf=crf_value)
        print(f"Sequentially combined video (from tensors) saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error in combine_videos_sequentially_from_tensors: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
