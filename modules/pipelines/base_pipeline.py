"""
Base pipeline class for FramePack Studio.
All pipeline implementations should inherit from this class.
"""

from modules.pipelines.metadata_utils import create_metadata


class BasePipeline:
    """Base class for all pipeline implementations."""

    def __init__(self, settings):
        """
        Initialize the pipeline with settings.

        Args:
            settings: Dictionary of settings for the pipeline
        """
        self.settings = settings

    def prepare_parameters(self, job_params):
        """
        Prepare parameters for the job.

        Args:
            job_params: Dictionary of job parameters

        Returns:
            Processed parameters dictionary
        """
        # Default implementation just returns the parameters as-is
        return job_params

    def validate_parameters(self, job_params):
        """
        Validate parameters for the job.

        Args:
            job_params: Dictionary of job parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Default implementation assumes all parameters are valid
        return True, None

    def preprocess_inputs(self, job_params):
        """
        Preprocess input images/videos for the job.

        Args:
            job_params: Dictionary of job parameters

        Returns:
            Processed inputs dictionary
        """
        # Default implementation returns an empty dictionary
        return {}

    def handle_results(self, job_params, result):
        """
        Handle the results of the job.

        Args:
            job_params: Dictionary of job parameters
            result: The result of the job

        Returns:
            Processed result
        """
        # Default implementation just returns the result as-is
        return result

    def create_metadata(self, job_params, job_id):
        """
        Create metadata for the job.

        Args:
            job_params: Dictionary of job parameters
            job_id: The job ID

        Returns:
            Metadata dictionary
        """
        return create_metadata(job_params, job_id, self.settings)
