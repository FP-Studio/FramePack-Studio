import hashlib
import json
from typing import Optional, cast
from dataclasses import dataclass, field, asdict

DEFAULT_WEIGHT: float = 0.8


@dataclass(frozen=True, eq=True)
class ModelLoraSetting:
    """ Represents a LoRA (Low-Rank Adaptation) setting for a model.
    Attributes:
        name: The name of the LoRA.
        weight: The weight of the LoRA. Typically between 0.0 and 1.0, but may be used up to 2.0 (or potentially higher).
                Default is 1.0 (DEFAULT_WEIGHT).
        sequence: The sequence order of the LoRA when applied to the model.
                  Lower numbers indicate higher priority, loaded first.
                  The sequence is automatically assigned if not provided or if duplicates exist.
                  The order of application can affect the final model output depending on block interactions.
        exclude_blocks: The blocks to exclude from the LoRA represented as a single regex string.
        include_blocks: The blocks to include in the LoRA represented as a single regex string.
    """

    name: str
    weight: float = field(default=DEFAULT_WEIGHT)
    sequence: int = field(default=0)
    exclude_blocks: Optional[str] = field(kw_only=True, default=None)
    include_blocks: Optional[str] = field(kw_only=True, default=None)

    def __post_init__(self):
        if not self.name:
            raise ValueError("ModelLoraSetting requires a 'name' attribute.")
        if not isinstance(self.weight, float):
            raise ValueError(
                "ModelLoraSetting requires a 'weight' attribute with a type of float but got {0} of type {1}".format(self.weight, type(self.weight)))
        if not isinstance(self.sequence, int):
            raise ValueError("ModelLoraSetting requires a 'sequence' attribute with a type of int but got {0} of type {1}".format(
                self.sequence, type(self.sequence)))

    @staticmethod
    def parse_settings(settings: list["ModelLoraSetting"] | str | list[str] | dict[str, dict] | dict[str, float | int] | None = None, reverse_sequence: bool = False) -> list["ModelLoraSetting"]:
        """Parses LoRA settings from various input formats into a list of ModelLoraSetting instances.
        Args:
            lora_settings: The LoRA settings to parse, which can be a list of ModelLoraSetting instances,
                           a list of strings (names), or a dictionary mapping names to settings defining at least weight and sequence.
            reverse_sequence: Whether to sort the settings in reverse order based on their sequence. Default is False.
        Returns:
            A list of ModelLoraSetting instances.
        """
        if settings is None or not settings:
            return []

        parsed_settings: list[ModelLoraSetting] = []
        if (isinstance(settings, str)):
            parsed_settings = [ModelLoraSetting(name=settings, weight=DEFAULT_WEIGHT, sequence=0)]
        elif (isinstance(settings, ModelLoraSetting)):
            parsed_settings = [settings]
        elif isinstance(settings, list) and all(isinstance(setting, ModelLoraSetting) for setting in settings):
            parsed_settings = list(set(lora for lora in settings if isinstance(
                lora, ModelLoraSetting)) if settings else [])
        elif isinstance(settings, str):
            parsed_settings = [ModelLoraSetting(name=settings, weight=DEFAULT_WEIGHT, sequence=0)]
        elif isinstance(settings, list) and all(isinstance(setting, str) for setting in settings):
            parsed_settings = [ModelLoraSetting(name=name, weight=DEFAULT_WEIGHT, sequence=sequence)
                               for sequence, name in enumerate(settings) if isinstance(name, str)]
        elif isinstance(settings, dict):
            if all(isinstance(k, str) and isinstance(v, float | int) for k, v in settings.items()):
                parsed_settings = [
                    ModelLoraSetting(name=name, weight=float(cast(float, weight)), sequence=sequence)
                    for sequence, (name, weight) in enumerate(settings.items())
                ]
            if all(isinstance(k, str) and isinstance(v, dict) for k, v in settings.items()):
                parsed_settings = [
                    ModelLoraSetting(name=name, weight=float(cast(dict, details).get("weight", DEFAULT_WEIGHT)),
                                     sequence=int(cast(dict, details).get("sequence", sequence)))
                    for sequence, (name, details) in enumerate(settings.items())
                ]
            elif all(isinstance(v, str) for v in settings.values()):
                parsed_settings = [
                    ModelLoraSetting(name=name, weight=DEFAULT_WEIGHT, sequence=sequence)
                    for sequence, (name, _) in enumerate(settings.items())
                ]
            else:
                raise ValueError("Invalid lora_settings format")

        if not parsed_settings:
            return []

        # assign sequences to settings without valid sequence
        sequences = [setting.sequence for setting in parsed_settings if setting.sequence is not None and setting.sequence >= 0]
        unique_sequences: set[int] = set(sequences)
        unique_sequences_len = len(unique_sequences)
        magic_number = 1000 if unique_sequences_len != len(parsed_settings) else 0
        for setting_index, setting in enumerate(parsed_settings):
            if unique_sequences_len == 0:
                # no sequence set on any setting, assign based on index
                setting = ModelLoraSetting(name=setting.name, weight=float(setting.weight), sequence=setting_index,
                                           exclude_blocks=setting.exclude_blocks, include_blocks=setting.include_blocks)
            elif setting.sequence is None or setting.sequence < 0:
                # sequence invalid or not set, assign a new unique sequence based on index + magic_number
                setting = ModelLoraSetting(name=setting.name, weight=float(setting.weight), sequence=setting_index +
                                           magic_number, exclude_blocks=setting.exclude_blocks, include_blocks=setting.include_blocks)

        # update duplicate sequences
        sequences = [setting.sequence for setting in parsed_settings if setting.sequence is not None and setting.sequence >= 0]
        unique_sequences: set[int] = set(sequences)
        unique_sequences_len = len(unique_sequences)
        for setting_index, setting in enumerate(parsed_settings):
            if sequences.count(setting.sequence) > 1:
                # duplicate sequence, assign a new unique sequence based on max existing + 1
                max_sequence = max(unique_sequences, default=1000 + setting_index)
                setting = ModelLoraSetting(name=setting.name, weight=float(setting.weight), sequence=max_sequence + 1,
                                           exclude_blocks=setting.exclude_blocks, include_blocks=setting.include_blocks)

        del unique_sequences, unique_sequences_len
        settings_set: set[ModelLoraSetting] = set()
        for setting_index, setting in enumerate(parsed_settings):
            if setting.sequence is None or setting.sequence <= 0:
                new_setting = setting if not any(setting.sequence == s.sequence for s in settings_set) else None
                if new_setting:
                    settings_set.add(new_setting)
                else:
                    max_sequence = max((s.sequence for s in settings_set if getattr(s, 'sequence', setting_index) is not None),
                                       default=1000 + len(settings_set))
                    settings_set.add(ModelLoraSetting(name=setting.name, weight=float(setting.weight), sequence=max_sequence + 1,
                                     exclude_blocks=setting.exclude_blocks, include_blocks=setting.include_blocks))

                if not any(setting.sequence == setting_index for setting in parsed_settings):
                    new_sequence = setting_index
                    setting = ModelLoraSetting(name=setting.name, weight=float(setting.weight), sequence=new_sequence,
                                               exclude_blocks=setting.exclude_blocks, include_blocks=setting.include_blocks)

                else:
                    max_sequence = max((s.sequence for s in parsed_settings if hasattr(s, 'sequence')),
                                       default=1000 + setting_index)
                    setting = ModelLoraSetting(name=setting.name, weight=float(setting.weight), sequence=max_sequence + 1,
                                               exclude_blocks=setting.exclude_blocks, include_blocks=setting.include_blocks)

        settings = list(settings_set)
        del parsed_settings, settings_set

        # always sort by sequence ascending or descending before returning
        settings.sort(key=lambda x: x.sequence, reverse=reverse_sequence)
        return settings

    @staticmethod
    def from_names_and_weights(lora_names: list[str], lora_weights: Optional[list[float | int]] = None) -> list["ModelLoraSetting"]:
        """Creates a list of ModelLoraSetting instances from lists of names and weights.
        Args:
            lora_names (list[str]): The list of LoRA names.
            lora_weights (Optional[list[float | int]]): The list of LoRA weights. If None, defaults to an empty list.
        Returns:
            list[ModelLoraSetting]: A list of ModelLoraSetting instances.
        """

        if lora_weights is None:
            lora_weights = []
        if len(lora_names) != len(lora_weights):
            print(
                f"Warning: Mismatch in lengths of lora_names ({len(lora_names)}) and lora_weights ({len(lora_weights)}).")
            additional_weights = len(lora_names) - len(lora_weights)
            if additional_weights > 0:
                print(f"Filling missing weights with default value {DEFAULT_WEIGHT}.")
                lora_weights = (lora_weights) + [DEFAULT_WEIGHT] * additional_weights
            else:
                lora_weights = (lora_weights)[:len(lora_names)]

        lora_settings: list[ModelLoraSetting] = []
        for sequence, (name, weight) in enumerate(zip(lora_names, lora_weights or [DEFAULT_WEIGHT] * len(lora_names))):
            lora_settings.append(ModelLoraSetting(name=name, weight=float(weight), sequence=sequence))
        return lora_settings


@dataclass
class ModelSettings():
    lora_settings: list[ModelLoraSetting] = field(default_factory=list)

    def add_lora_setting(self, setting: ModelLoraSetting) -> None:
        max_sequence = max((s.sequence for s in self.lora_settings if hasattr(s, 'sequence')), default=-1)
        new_sequence = max_sequence + 1
        self.lora_settings.append(ModelLoraSetting(name=setting.name, weight=setting.weight, sequence=new_sequence,
                                  include_blocks=setting.include_blocks, exclude_blocks=setting.exclude_blocks))


@dataclass
class ModelConfiguration:
    model_name: str
    settings: ModelSettings = field(default_factory=ModelSettings)

    @property
    def _hash(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def add_lora_setting(self, setting: ModelLoraSetting) -> None:
        self.settings.add_lora_setting(setting)

    def add_lora(self, name: str, weight: float = DEFAULT_WEIGHT) -> None:
        self.add_lora_setting(ModelLoraSetting(name=name, weight=weight))

    def validate(self) -> bool:
        total_weights = sum([setting.weight for setting in self.settings.lora_settings])
        valid = 2 > total_weights > 0
        if not valid:
            print("Warning: total weight for all LoRA may not perform well with the model ({0}). Total weight: {1}".format(
                self.model_name, total_weights))
        return valid

    @staticmethod
    def from_settings(model_name: str, settings: ModelSettings | dict | None):
        model_settings: ModelSettings | None = None
        if settings is None:
            model_settings = ModelSettings()
        elif isinstance(settings, ModelSettings):
            model_settings = settings
        elif isinstance(settings, dict):
            model_settings = ModelSettings(lora_settings=ModelLoraSetting.parse_settings(settings))

        if model_settings is None:
            raise ValueError("Invalid config type for ModelConfiguration")

        return ModelConfiguration(model_name=model_name, settings=model_settings)

    @staticmethod
    def from_lora_names_and_weights(model_name: str, lora_names: list[str], lora_weights: Optional[list[float | int]] = None) -> "ModelConfiguration":
        weights: list[float] = [float(weight) for weight in (lora_weights or [])]
        lora_settings = ModelLoraSetting.from_names_and_weights(lora_names, lora_weights=weights)
        model_settings = ModelSettings(lora_settings=lora_settings)
        del weights, lora_settings
        return ModelConfiguration.from_settings(model_name=model_name, settings=model_settings)

    def set_model_name(self, model_name: str) -> "ModelConfiguration":
        self.model_name = model_name
        return self

    def set_settings(self, settings: ModelSettings) -> "ModelConfiguration":
        self.settings = settings
        return self

    def update_lora_setting(self, lora_settings: list[ModelLoraSetting] | str | list[str] | dict[str, dict]) -> "ModelConfiguration":
        self.settings.lora_settings = ModelLoraSetting.parse_settings(lora_settings)
        return self


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    config: ModelConfiguration = ModelConfiguration.from_lora_names_and_weights(
        model_name="test-model",
        lora_names=["lora1", "lora2", "lora3"],
        lora_weights=[0.5, 0.5, 0.5]
    )
    logger.info(f"Model Name: {config.model_name}")
    logger.info(f"LoRA Settings: {config.settings.lora_settings}")
    logger.debug(json.dumps(asdict(config), indent=4))
    logger.debug("hash: {0}".format(config._hash))
    config.model_name = "changed"
    logger.debug("hash: {0}".format(config._hash))
    config.settings.lora_settings = ModelLoraSetting.from_names_and_weights(
        lora_names=["lora_A", "lora_B", "lora_C"],
        lora_weights=[1, 1.5, 2.5]
    )
    logger.debug("hash: {0}".format(config._hash))
    config.settings.lora_settings.append(ModelLoraSetting(name="lora_D", weight=0.75))
    logger.debug(json.dumps(asdict(config), indent=4))
    logger.debug("hash: {0}".format(config._hash))
    config.add_lora("lora_E")
    logger.debug(json.dumps(asdict(config), indent=4))
    logger.debug("hash: {0}".format(config._hash))
    valid = config.validate()
    logger.debug("Config validation result: {0}".format(valid))
