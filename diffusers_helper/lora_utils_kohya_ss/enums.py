import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
    # StrEnum is introduced in 3.11 while we support python 3.10
else:
    from enum import Enum, auto
    from typing import Any

    # Fallback for Python 3.10 and earlier
    class StrEnum(str, Enum):
        def __new__(cls, value, *args, **kwargs):
            if not isinstance(value, (str, auto)):
                raise TypeError(
                    f"Values of StrEnums must be strings: {value!r} is a {type(value)}"
                )
            return super().__new__(cls, value, *args, **kwargs)

        def __str__(self):
            return str(self.value)

        @staticmethod
        def _generate_next_value_(
            name: str, start: int, count: int, last_values: list[Any]
        ) -> str:
            return name


class LoraLoader(StrEnum):
    DIFFUSERS = "diffusers"
    LORA_READY = "lora_ready"
    DEFAULT = DIFFUSERS

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
            return LoraLoader.DEFAULT


if __name__ == "__main__":
    # Test the StrEnum functionality
    print("diffusers:", LoraLoader.DIFFUSERS)  # Should print "diffusers"
    print("lora_ready:", LoraLoader.LORA_READY)  # Should print "lora_ready"
    print("default:", LoraLoader.DEFAULT)  # Should print "lora_ready"
    print(  # Should print all unique supported values (excludes aliases like DEFAULT)
        "supported_values:", LoraLoader.supported_values()
    )
    try:
        print("fail:", LoraLoader("invalid"))  # Should raise ValueError
    except ValueError as e:
        print("pass:", e)  # Prints: Invalid LoraLoader value: invalid
    try:
        print("pass:", LoraLoader("diffusers"))  # Should return LoraLoader.DIFFUSERS
    except ValueError as e:
        print("fail:", e)
    try:
        print("type of LoraLoader.DEFAULT:", type(LoraLoader.DEFAULT))
        default = LoraLoader.DEFAULT
        print("type of default:", type(default))  # Should be LoraLoader, not str
    except Exception as e:
        print(f"fail: {e}")

    assert isinstance(LoraLoader("lora_ready"), StrEnum)
    assert isinstance(
        LoraLoader.DIFFUSERS, LoraLoader
    ), "DIFFUSERS should be an instance of LoraLoader"
    assert (
        LoraLoader.DEFAULT == LoraLoader.DIFFUSERS
    ), "Default loader should be DIFFUSERS"
    assert (
        LoraLoader.DIFFUSERS != LoraLoader.LORA_READY
    ), "DIFFUSERS should not equal LORA_READY"

    assert (
        LoraLoader.LORA_READY.value == "lora_ready"
    ), "lora_ready string should equal LoraLoader.LORA_READY"
