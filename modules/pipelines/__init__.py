"""
Pipeline module for FramePack Studio.
This module provides pipeline classes for different generation types.
"""

from .base_pipeline import BasePipeline
from .original_pipeline import OriginalPipeline
from .f1_pipeline import F1Pipeline
from .original_with_endframe_pipeline import OriginalWithEndframePipeline
from .video_pipeline import VideoPipeline
from .video_f1_pipeline import VideoF1Pipeline


def create_pipeline(model_type, settings):
    """
    Create a pipeline instance for the specified model type.

    Args:
        model_type: The type of model to create a pipeline for
        settings: Dictionary of settings for the pipeline

    Returns:
        A pipeline instance for the specified model type
    """
    if model_type == "Original":
        return OriginalPipeline(settings)
    elif model_type == "F1":
        return F1Pipeline(settings)
    elif model_type == "Original with Endframe":
        return OriginalWithEndframePipeline(settings)
    elif model_type == "Video":
        return VideoPipeline(settings)
    elif model_type == "Video F1":
        return VideoF1Pipeline(settings)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    "BasePipeline",
    "OriginalPipeline",
    "F1Pipeline",
    "OriginalWithEndframePipeline",
    "VideoPipeline",
    "VideoF1Pipeline",
    "create_pipeline",
]
