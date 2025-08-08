import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum as NativeStrEnum
    # StrEnum is introduced in 3.11 while we support python 3.10

    class StrEnum(NativeStrEnum):
        pass

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
