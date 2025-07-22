"""
Video F1 pipeline class for FramePack Studio.
This pipeline handles the "Video F1" model type.
"""

from diffusers_helper.bucket_tools import find_nearest_bucket
from .base_pipeline import BasePipeline


class VideoF1Pipeline(BasePipeline):
    """Pipeline for Video F1 generation type."""

    def prepare_parameters(self, job_params):
        """
        Prepare parameters for the Video generation job.

        Args:
            job_params: Dictionary of job parameters

        Returns:
            Processed parameters dictionary
        """
        processed_params = job_params.copy()

        # Ensure we have the correct model type
        processed_params["model_type"] = "Video F1"

        return processed_params

    def validate_parameters(self, job_params):
        """
        Validate parameters for the Video generation job.

        Args:
            job_params: Dictionary of job parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required parameters
        required_params = ["prompt_text", "seed", "total_second_length", "steps"]
        for param in required_params:
            if param not in job_params:
                return False, f"Missing required parameter: {param}"

        # Validate numeric parameters
        if job_params.get("total_second_length", 0) <= 0:
            return False, "Video length must be greater than 0"

        if job_params.get("steps", 0) <= 0:
            return False, "Steps must be greater than 0"

        # Check for input video (stored in input_image for Video F1 model)
        if not job_params.get("input_image"):
            return False, "Input video is required for Video F1 model"

        # Check if combine_with_source is provided (optional)
        combine_with_source = job_params.get("combine_with_source")
        if combine_with_source is not None and not isinstance(
            combine_with_source, bool
        ):
            return False, "combine_with_source must be a boolean value"

        return True, None

    def preprocess_inputs(self, job_params):
        """
        Preprocess input video for the Video F1 generation type.

        Args:
            job_params: Dictionary of job parameters

        Returns:
            Processed inputs dictionary
        """
        processed_inputs = {}

        # Get the input video (stored in input_image for Video F1 model)
        input_video = job_params.get("input_image")
        if not input_video:
            raise ValueError("Input video is required for Video F1 model")

        # Store the input video
        processed_inputs["input_video"] = input_video

        # Note: The following code will be executed in the worker function:
        # 1. The worker will call video_encode on the generator to get video_latents and input_video_pixels
        # 2. Then it will store these values for later use:
        #    input_video_pixels = input_video_pixels.cpu()
        #    video_latents = video_latents.cpu()
        #
        # 3. If the generator has the set_full_video_latents method, it will store the video latents:
        #    if hasattr(current_generator, 'set_full_video_latents'):
        #        current_generator.set_full_video_latents(video_latents.clone())
        #        print(f"Stored full input video latents in VideoModelGenerator. Shape: {video_latents.shape}")
        #
        # 4. For the Video model, history_latents is initialized with the video_latents:
        #    history_latents = video_latents
        #    print(f"Initialized history_latents with video context. Shape: {history_latents.shape}")
        processed_inputs["input_files_dir"] = job_params.get("input_files_dir")

        # Pass through the combine_with_source parameter if it exists
        if "combine_with_source" in job_params:
            processed_inputs["combine_with_source"] = job_params.get(
                "combine_with_source"
            )
            print(
                f"Video F1 pipeline: combine_with_source = {processed_inputs['combine_with_source']}"
            )

        # Pass through the num_cleaned_frames parameter if it exists
        if "num_cleaned_frames" in job_params:
            processed_inputs["num_cleaned_frames"] = job_params.get(
                "num_cleaned_frames"
            )
            print(
                f"Video F1 pipeline: num_cleaned_frames = {processed_inputs['num_cleaned_frames']}"
            )

        # Get resolution parameters
        resolutionW = job_params.get("resolutionW", 640)
        resolutionH = job_params.get("resolutionH", 640)

        # Find nearest bucket size
        height, width = find_nearest_bucket(
            resolutionH, resolutionW, (resolutionW + resolutionH) / 2
        )

        # Store the dimensions
        processed_inputs["height"] = height
        processed_inputs["width"] = width

        return processed_inputs

    def handle_results(self, job_params, result):
        """
        Handle the results of the Video F1 generation.

        Args:
            job_params: The job parameters
            result: The generation result

        Returns:
            Processed result
        """
        # For Video F1 generation, we just return the result as-is
        return result

    # Using the centralized create_metadata method from BasePipeline
