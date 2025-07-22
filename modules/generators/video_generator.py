import torch
import math

from diffusers_helper.hunyuan import vae_decode
from .video_base_generator import VideoBaseModelGenerator


class VideoModelGenerator(VideoBaseModelGenerator):
    """
    Generator for the Video (backward) extension of the Original HunyuanVideo model.
    These generators accept video input instead of a single image.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Video model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "Video"
        self.model_path = "lllyasviel/FramePackI2V_HY"  # Same as Original
        self.model_repo_id_for_cache = "models--lllyasviel--FramePackI2V_HY"

    def get_latent_paddings(self, total_latent_sections):
        """
        Get the latent paddings for the Video model.

        Args:
            total_latent_sections: The total number of latent sections

        Returns:
            A list of latent paddings
        """
        # Video model uses reversed latent paddings like Original
        if total_latent_sections > 4:
            return [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            return list(reversed(range(total_latent_sections)))

    def video_prepare_clean_latents_and_indices(
        self,
        end_frame_output_dimensions_latent,
        end_frame_weight,
        end_clip_embedding,
        end_of_input_video_embedding,
        latent_paddings,
        latent_padding,
        latent_padding_size,
        latent_window_size,
        video_latents,
        history_latents,
        num_cleaned_frames=5,
    ):
        """
        Combined method to prepare clean latents and indices for the Video model.

        Args:
            Work in progress - better not to pass in latent_paddings and latent_padding.
            num_cleaned_frames: Number of context frames to use from the video (adherence to video)

        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices, clean_latents, clean_latents_2x, clean_latents_4x)
        """
        # Get num_cleaned_frames from job_params if available, otherwise use default value of 5
        num_clean_frames = num_cleaned_frames if num_cleaned_frames is not None else 5

        # HACK SOME STUFF IN THAT SHOULD NOT BE HERE
        # Placeholders for end frame processing
        # Colin, I'm only leaving them for the moment in case you want separate models for
        # Video-backward and Video-backward-Endframe.
        # end_latent = None
        # end_of_input_video_embedding = None # Placeholder for end frame's CLIP embedding. SEE: 20250507 pftq: Process end frame if provided
        # end_clip_embedding = None # Placeholders for end frame processing. SEE: 20250507 pftq: Process end frame if provided
        # end_frame_weight = 0.0 # Placeholders for end frame processing. SEE: 20250507 pftq: Process end frame if provided
        # HACK MORE STUFF IN THAT PROBABLY SHOULD BE ARGUMENTS OR OTHWISE MADE AVAILABLE
        end_of_input_video_latent = video_latents[
            :, :, -1:
        ]  # Last frame of the input video (produced by video_encode in the PR)
        is_start_of_video = (
            latent_padding == 0
        )  # This refers to the start of the *generated* video part
        is_end_of_video = (
            latent_padding == latent_paddings[0]
        )  # This refers to the end of the *generated* video part (closest to input video) (better not to pass in latent_paddings[])
        # End of HACK STUFF

        # Dynamic frame allocation for context frames (clean latents)
        # This determines which frames from history_latents are used as input for the transformer.
        available_frames = (
            video_latents.shape[2] if is_start_of_video else history_latents.shape[2]
        )  # Use input video frames for first segment, else previously generated history
        effective_clean_frames = (
            max(0, num_clean_frames - 1) if num_clean_frames > 1 else 1
        )
        if is_start_of_video:
            effective_clean_frames = 1  # Avoid jumpcuts if input video is too different

        clean_latent_pre_frames = effective_clean_frames
        num_2x_frames = (
            min(2, max(1, available_frames - clean_latent_pre_frames - 1))
            if available_frames > clean_latent_pre_frames + 1
            else 1
        )
        num_4x_frames = (
            min(16, max(1, available_frames - clean_latent_pre_frames - num_2x_frames))
            if available_frames > clean_latent_pre_frames + num_2x_frames
            else 1
        )
        total_context_frames = num_2x_frames + num_4x_frames
        total_context_frames = min(
            total_context_frames, available_frames - clean_latent_pre_frames
        )

        # Prepare indices for the transformer's input (these define the *relative positions* of different frame types in the input tensor)
        # The total length is the sum of various frame types:
        # clean_latent_pre_frames: frames before the blank/generated section
        # latent_padding_size: blank frames before the generated section (for backward generation)
        # latent_window_size: the new frames to be generated
        # post_frames: frames after the generated section
        # num_2x_frames, num_4x_frames: frames for lower resolution context
        # 20250511 pftq: Dynamically adjust post_frames based on clean_latents_post
        post_frames = (
            1
            if is_end_of_video and end_frame_output_dimensions_latent is not None
            else effective_clean_frames
        )  # 20250511 pftq: Single frame for end_latent, otherwise padding causes still image
        indices = torch.arange(
            0,
            clean_latent_pre_frames
            + latent_padding_size
            + latent_window_size
            + post_frames
            + num_2x_frames
            + num_4x_frames,
        ).unsqueeze(0)
        (
            clean_latent_indices_pre,
            blank_indices,
            latent_indices,
            clean_latent_indices_post,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
        ) = indices.split(
            [
                clean_latent_pre_frames,
                latent_padding_size,
                latent_window_size,
                post_frames,
                num_2x_frames,
                num_4x_frames,
            ],
            dim=1,
        )
        clean_latent_indices = torch.cat(
            [clean_latent_indices_pre, clean_latent_indices_post], dim=1
        )  # Combined indices for 1x clean latents

        # Prepare the *actual latent data* for the transformer's context inputs
        # These are extracted from history_latents (or video_latents for the first segment)
        context_frames = (
            history_latents[
                :,
                :,
                -(
                    total_context_frames + clean_latent_pre_frames
                ) : -clean_latent_pre_frames,
                :,
                :,
            ]
            if total_context_frames > 0
            else history_latents[:, :, :1, :, :]
        )
        # clean_latents_4x: 4x downsampled context frames. From history_latents (or video_latents).
        # clean_latents_2x: 2x downsampled context frames. From history_latents (or video_latents).
        split_sizes = [num_4x_frames, num_2x_frames]
        split_sizes = [s for s in split_sizes if s > 0]
        if split_sizes and context_frames.shape[2] >= sum(split_sizes):
            splits = context_frames.split(split_sizes, dim=2)
            split_idx = 0
            clean_latents_4x = (
                splits[split_idx]
                if num_4x_frames > 0
                else history_latents[:, :, :1, :, :]
            )
            split_idx += 1 if num_4x_frames > 0 else 0
            clean_latents_2x = (
                splits[split_idx]
                if num_2x_frames > 0 and split_idx < len(splits)
                else history_latents[:, :, :1, :, :]
            )
        else:
            clean_latents_4x = clean_latents_2x = history_latents[:, :, :1, :, :]

        # clean_latents_pre: Latents from the *end* of the input video (if is_start_of_video), or previously generated history.
        # Its purpose is to provide a smooth transition *from* the input video.
        clean_latents_pre = video_latents[
            :, :, -min(effective_clean_frames, video_latents.shape[2]) :
        ].to(history_latents)

        # clean_latents_post: Latents from the *beginning* of the previously generated video segments.
        # Its purpose is to provide a smooth transition *to* the existing generated content.
        clean_latents_post = history_latents[
            :, :, : min(effective_clean_frames, history_latents.shape[2]), :, :
        ]

        # Special handling for the end frame:
        # If it's the very first segment being generated (is_end_of_video in terms of generation order),
        # and an end_latent was provided, force clean_latents_post to be that end_latent.
        if is_end_of_video:
            clean_latents_post = torch.zeros_like(end_of_input_video_latent).to(
                history_latents
            )  # Initialize to zero

        # RT_BORG: end_of_input_video_embedding and end_clip_embedding shouldn't need to be checked, since they should
        # always be provided if end_latent is provided. But bulletproofing before the release since test time will be short.
        if (
            end_frame_output_dimensions_latent is not None
            and end_of_input_video_embedding is not None
            and end_clip_embedding is not None
        ):
            # image_encoder_last_hidden_state: Weighted average of CLIP embedding of first input frame and end frame's CLIP embedding
            # This guides the overall content to transition towards the end frame.
            image_encoder_last_hidden_state = (
                1 - end_frame_weight
            ) * end_of_input_video_embedding + end_clip_embedding * end_frame_weight
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(
                self.transformer.dtype
            )

            if is_end_of_video:
                # For the very first generated segment, the "post" part is the end_latent itself.
                clean_latents_post = end_frame_output_dimensions_latent.to(
                    history_latents
                )[:, :, :1, :, :]  # Ensure single frame

        # Pad clean_latents_pre/post if they have fewer frames than specified by clean_latent_pre_frames/post_frames
        if clean_latents_pre.shape[2] < clean_latent_pre_frames:
            clean_latents_pre = clean_latents_pre.repeat(
                1,
                1,
                math.ceil(clean_latent_pre_frames / clean_latents_pre.shape[2]),
                1,
                1,
            )[:, :, :clean_latent_pre_frames]
        if clean_latents_post.shape[2] < post_frames:
            clean_latents_post = clean_latents_post.repeat(
                1, 1, math.ceil(post_frames / clean_latents_post.shape[2]), 1, 1
            )[:, :, :post_frames]

        # clean_latents: Concatenation of pre and post clean latents. These are the 1x resolution context frames.
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

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
        Backward Generation: Update the history latents with the generated latents for the Video model.

        Args:
            history_latents: The history latents
            generated_latents: The generated latents

        Returns:
            The updated history latents
        """
        # For Video model, we prepend the generated latents to the front of history latents
        # This matches the original implementation in video-example.py
        # It generates new sections backwards in time, chunk by chunk
        return torch.cat(
            [generated_latents.to(history_latents), history_latents], dim=2
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
        # Generated frames at the front.
        return history_latents[:, :, :total_generated_latent_frames, :, :]

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

        # For Video model, we prepend the current pixels to the history pixels
        # This matches the original implementation in video-example.py
        return soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

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
        # For backward Video mode, current pixels are at the front of history.
        return vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()

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
        # For Video model, current_pos already includes the input video time
        # We just need to display the total generated frames and the current position
        return (
            f"Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, "
            f"Generated video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). "
            f"Current position: {current_pos:.2f}s (remaining: {original_pos:.2f}s). "
            f"using prompt: {current_prompt[:256]}..."
        )
