from .original_generator import OriginalModelGenerator


class OriginalWithEndframeModelGenerator(OriginalModelGenerator):
    """
    Model generator for the Original HunyuanVideo model with end frame support.
    This extends the Original model with the ability to guide generation toward a specified end frame.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Original with Endframe model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "Original with Endframe"
        # Inherits everything else from OriginalModelGenerator
