import torch

from diffusers_helper.hunyuan import vae_decode
from .video_base_generator import VideoBaseModelGenerator


class VideoF1ModelGenerator(VideoBaseModelGenerator):
    """
    Model generator for the Video F1 (forward video) extension of the F1 HunyuanVideo model.
    These generators accept video input instead of a single image.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Video F1 model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "Video F1"
        self.model_path = "lllyasviel/FramePack_F1_I2V_HY_20250503"  # Same as F1
        self.model_repo_id_for_cache = (
            "models--lllyasviel--FramePack_F1_I2V_HY_20250503"  # Same as F1
        )

    def get_latent_paddings(self, total_latent_sections):
        """
        Get the latent paddings for the Video model.

        Args:
            total_latent_sections: The total number of latent sections

        Returns:
            A list of latent paddings
        """
        # RT_BORG: pftq didn't even use latent paddings in the forward Video model. Keeping it for consistency.
        # Any list the size of total_latent_sections should work, but may as well end with 0 as a marker for the last section.
        # Similar to F1 model uses a fixed approach with just 0 for last section and 1 for others
        return [1] * (total_latent_sections - 1) + [0]

    def video_f1_prepare_clean_latents_and_indices(
        self, latent_window_size, video_latents, history_latents, num_cleaned_frames=5
    ):
        """
        Combined method to prepare clean latents and indices for the Video model.

        Args:
            Work in progress - better not to pass in latent_paddings and latent_padding.

        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices, clean_latents, clean_latents_2x, clean_latents_4x)
        """
        # Get num_cleaned_frames from job_params if available, otherwise use default value of 5
        num_clean_frames = num_cleaned_frames if num_cleaned_frames is not None else 5

        # RT_BORG: Retaining this commented code for reference.
        # start_latent = history_latents[:, :, :1]  # Shape: (1, channels, 1, height//8, width//8)
        start_latent = video_latents[
            :, :, -1:
        ]  # Shape: (1, channels, 1, height//8, width//8)

        available_frames = history_latents.shape[2]  # Number of latent frames
        max_pixel_frames = min(
            latent_window_size * 4 - 3, available_frames * 4
        )  # Cap at available pixel frames
        adjusted_latent_frames = max(
            1, (max_pixel_frames + 3) // 4
        )  # Convert back to latent frames
        # Adjust num_clean_frames to match original behavior: num_clean_frames=2 means 1 frame for clean_latents_1x
        effective_clean_frames = (
            max(0, num_clean_frames - 1) if num_clean_frames > 1 else 0
        )
        effective_clean_frames = (
            min(effective_clean_frames, available_frames - 2)
            if available_frames > 2
            else 0
        )  # 20250507 pftq: changed 1 to 2 for edge case for <=1 sec videos
        num_2x_frames = (
            min(2, max(1, available_frames - effective_clean_frames - 1))
            if available_frames > effective_clean_frames + 1
            else 0
        )  # 20250507 pftq: subtracted 1 for edge case for <=1 sec videos
        num_4x_frames = (
            min(16, max(1, available_frames - effective_clean_frames - num_2x_frames))
            if available_frames > effective_clean_frames + num_2x_frames
            else 0
        )  # 20250507 pftq: Edge case for <=1 sec

        total_context_frames = num_4x_frames + num_2x_frames + effective_clean_frames
        total_context_frames = min(
            total_context_frames, available_frames
        )  # 20250507 pftq: Edge case for <=1 sec videos

        indices = torch.arange(
            0,
            sum(
                [
                    1,
                    num_4x_frames,
                    num_2x_frames,
                    effective_clean_frames,
                    adjusted_latent_frames,
                ]
            ),
        ).unsqueeze(
            0
        )  # 20250507 pftq: latent_window_size to adjusted_latent_frames for edge case for <=1 sec videos
        (
            clean_latent_indices_start,
            clean_latent_4x_indices,
            clean_latent_2x_indices,
            clean_latent_1x_indices,
            latent_indices,
        ) = indices.split(
            [
                1,
                num_4x_frames,
                num_2x_frames,
                effective_clean_frames,
                adjusted_latent_frames,
            ],
            dim=1,  # 20250507 pftq: latent_window_size to adjusted_latent_frames for edge case for <=1 sec videos
        )
        clean_latent_indices = torch.cat(
            [clean_latent_indices_start, clean_latent_1x_indices], dim=1
        )

        # 20250506 pftq: Split history_latents dynamically based on available frames
        fallback_frame_count = (
            2  # 20250507 pftq: Changed 0 to 2 Edge case for <=1 sec videos
        )
        context_frames = (
            history_latents[:, :, -total_context_frames:, :, :]
            if total_context_frames > 0
            else history_latents[:, :, :fallback_frame_count, :, :]
        )
        if total_context_frames > 0:
            split_sizes = [num_4x_frames, num_2x_frames, effective_clean_frames]
            split_sizes = [s for s in split_sizes if s > 0]  # Remove zero sizes
            if split_sizes:
                splits = context_frames.split(split_sizes, dim=2)
                split_idx = 0
                clean_latents_4x = (
                    splits[split_idx]
                    if num_4x_frames > 0
                    else history_latents[:, :, :fallback_frame_count, :, :]
                )
                if (
                    clean_latents_4x.shape[2] < 2
                ):  # 20250507 pftq: edge case for <=1 sec videos
                    clean_latents_4x = torch.cat(
                        [clean_latents_4x, clean_latents_4x[:, :, -1:, :, :]], dim=2
                    )[:, :, :2, :, :]
                split_idx += 1 if num_4x_frames > 0 else 0
                clean_latents_2x = (
                    splits[split_idx]
                    if num_2x_frames > 0 and split_idx < len(splits)
                    else history_latents[:, :, :fallback_frame_count, :, :]
                )
                if (
                    clean_latents_2x.shape[2] < 2
                ):  # 20250507 pftq: edge case for <=1 sec videos
                    clean_latents_2x = torch.cat(
                        [clean_latents_2x, clean_latents_2x[:, :, -1:, :, :]], dim=2
                    )[:, :, :2, :, :]
                split_idx += 1 if num_2x_frames > 0 else 0
                clean_latents_1x = (
                    splits[split_idx]
                    if effective_clean_frames > 0 and split_idx < len(splits)
                    else history_latents[:, :, :fallback_frame_count, :, :]
                )
            else:
                clean_latents_4x = clean_latents_2x = clean_latents_1x = (
                    history_latents[:, :, :fallback_frame_count, :, :]
                )
        else:
            clean_latents_4x = clean_latents_2x = clean_latents_1x = history_latents[
                :, :, :fallback_frame_count, :, :
            ]

        clean_latents = torch.cat(
            [start_latent.to(history_latents), clean_latents_1x], dim=2
        )

        return (
            clean_latent_indices,
            latent_indices,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
            clean_latents,
            clean_latents_2x,
            clean_latents_4x,
        )

    def update_history_latents(self, history_latents, generated_latents):
        """
        Forward Generation: Update the history latents with the generated latents for the Video F1 model.

        Args:
            history_latents: The history latents
            generated_latents: The generated latents

        Returns:
            The updated history latents
        """
        # For Video F1 model, we append the generated latents to the back of history latents
        # This matches the F1 implementation
        # It generates new sections forward in time, chunk by chunk
        return torch.cat(
            [history_latents, generated_latents.to(history_latents)], dim=2
        )

    def get_real_history_latents(self, history_latents, total_generated_latent_frames):
        """
        Get the real history latents for the backward Video model. For Video, this is the first
        `total_generated_latent_frames` frames of the history latents.

        Args:
            history_latents: The history latents
            total_generated_latent_frames: The total number of generated latent frames

        Returns:
            The real history latents
        """
        # Generated frames at the back. Note the difference in "-total_generated_latent_frames:".
        return history_latents[:, :, -total_generated_latent_frames:, :, :]

    def update_history_pixels(self, history_pixels, current_pixels, overlapped_frames):
        """
        Update the history pixels with the current pixels for the Video model.

        Args:
            history_pixels: The history pixels
            current_pixels: The current pixels
            overlapped_frames: The number of overlapped frames

        Returns:
            The updated history pixels
        """
        from diffusers_helper.utils import soft_append_bcthw

        # For Video F1 model, we append the current pixels to the history pixels
        # This matches the F1 model, history_pixels is first, current_pixels is second
        return soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

    def get_current_pixels(self, real_history_latents, section_latent_frames, vae):
        """
        Get the current pixels for the Video model.

        Args:
            real_history_latents: The real history latents
            section_latent_frames: The number of section latent frames
            vae: The VAE model

        Returns:
            The current pixels
        """
        # For forward Video mode, current pixels are at the back of history, like F1.
        return vae_decode(
            real_history_latents[:, :, -section_latent_frames:], vae
        ).cpu()

    def format_position_description(
        self, total_generated_latent_frames, current_pos, original_pos, current_prompt
    ):
        """
        Format the position description for the Video model.

        Args:
            total_generated_latent_frames: The total number of generated latent frames
            current_pos: The current position in seconds (includes input video time)
            original_pos: The original position in seconds
            current_prompt: The current prompt

        Returns:
            The formatted position description
        """
        # RT_BORG: Duplicated from F1. Is this correct?
        return (
            f"Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, "
            f"Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). "
            f"Current position: {current_pos:.2f}s. "
            f"using prompt: {current_prompt[:256]}..."
        )
