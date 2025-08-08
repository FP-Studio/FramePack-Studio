import logging
from enum import auto
from importlib.util import find_spec

from shared.StrEnum import StrEnum


class QuantizationFormat(StrEnum):
    normal_float_4bit = auto()
    integer_8bit = auto()
    brain_floating_point_16bit = auto()
    NONE = brain_floating_point_16bit
    DEFAULT = NONE

    __logger = logging.getLogger(__name__)

    @staticmethod
    def supported_values() -> list[str]:
        """Returns a list of all supported QuantizationFormat values."""
        return [loader.value for loader in QuantizationFormat]

    @staticmethod
    def safe_parse(value: "str | QuantizationFormat") -> "QuantizationFormat":
        if find_spec("bitsandbytes") is None:
            QuantizationFormat.__logger.error(
                "bitsandbytes not found, defaulting to no quantization. https://docs.framepackstudio.com/help"
            )
            # If bitsandbytes is not installed, we can not quantize, so return NONE
            return QuantizationFormat.NONE

        QuantizationFormat.__logger.debug(f"QuantizationFormat: {value}")
        if isinstance(value, QuantizationFormat):
            return value
        try:
            return QuantizationFormat(value)
        except ValueError:
            QuantizationFormat.__logger.exception(
                f"Invalid QuantizationFormat value: {value}, defaulting to {QuantizationFormat.DEFAULT}."
            )
            return QuantizationFormat.DEFAULT


class LoraLoader(StrEnum):
    DIFFUSERS = "diffusers"
    LORA_READY = "lora_ready"
    DEFAULT = LORA_READY

    __logger = logging.getLogger(__name__)

    @staticmethod
    def supported_values() -> list[str]:
        """Returns a list of all supported LoraLoader values."""
        return [loader.value for loader in LoraLoader]

    @staticmethod
    def safe_parse(value: "str | LoraLoader") -> "LoraLoader":
        if isinstance(value, LoraLoader):
            return value
        try:
            return LoraLoader(value)
        except ValueError:
            LoraLoader.__logger.exception(
                f"Invalid LoraLoader value: {value}, defaulting to {LoraLoader.DEFAULT}."
            )
            return LoraLoader.DEFAULT


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # Test the StrEnum functionality
    logger.info("diffusers:", LoraLoader.DIFFUSERS)  # Should print "diffusers"
    logger.info("lora_ready:", LoraLoader.LORA_READY)  # Should print "lora_ready"
    logger.info("default:", LoraLoader.DEFAULT)  # Should print "lora_ready"
    logger.info("supported_values:", LoraLoader.supported_values())
    try:
        logger.info("fail:", LoraLoader("invalid"))  # Should raise ValueError
    except ValueError as e:
        logger.exception("pass:", e)  # Prints: Invalid LoraLoader value: invalid
    try:
        logger.info(
            "pass:", LoraLoader("diffusers")
        )  # Should return LoraLoader.DIFFUSERS
    except ValueError as e:
        logger.exception("fail:", e)
    try:
        logger.info("type of LoraLoader.DEFAULT:", type(LoraLoader.DEFAULT))
        default = LoraLoader.DEFAULT
        logger.info("type of default:", type(default))  # Should be LoraLoader, not str
    except Exception as e:
        logger.exception("fail:", e)

    assert isinstance(LoraLoader("lora_ready"), StrEnum)
    assert isinstance(LoraLoader.DIFFUSERS, LoraLoader), (
        "DIFFUSERS should be an instance of LoraLoader"
    )
    assert LoraLoader.DEFAULT == LoraLoader.DIFFUSERS, (
        "Default loader should be DIFFUSERS"
    )
    assert LoraLoader.DIFFUSERS != LoraLoader.LORA_READY, (
        "DIFFUSERS should not equal LORA_READY"
    )

    assert LoraLoader.LORA_READY.value == "lora_ready", (
        "lora_ready string should equal LoraLoader.LORA_READY"
    )
