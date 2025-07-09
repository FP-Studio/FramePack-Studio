from typing import (
    # Generator,
    Optional,
    Union,
)

from modules.generators.model_configuration import ModelConfiguration
from .generators import (
    BaseModelGenerator,
    VideoBaseModelGenerator
)
from .settings import Settings
from .video_queue import VideoJobQueue

# cSpell: ignore loras


class ModelState:
    """
    Class to track the state of model configurations.
    This class keeps track of the previous model configuration and its hash.
    """
    previous_model_configuration: Optional[ModelConfiguration]
    previous_model_hash: str
    active_model_configuration: Optional[ModelConfiguration]
    active_model_hash: str

    def __init__(self):
        self.previous_model_configuration: Optional[ModelConfiguration] = None
        self.previous_model_hash: str = 'previous_model_hash'
        self.active_model_configuration: Optional[ModelConfiguration] = None
        self.active_model_hash: str = 'active_model_hash'

    def is_reload_required(self, model_name: str, current_generator: Optional[BaseModelGenerator], selected_loras: list[str], lora_values: list[float], lora_loaded_names: list[str]) -> bool:
        selected_lora_values = [lora_values[lora_loaded_names.index(
            name)] for name in selected_loras if name in lora_loaded_names]

        active_model_configuration: ModelConfiguration = ModelConfiguration.from_lora_names_and_weights(
            model_name=model_name,
            lora_names=selected_loras,
            lora_weights=selected_lora_values
        )

        return active_model_configuration._hash != self.active_model_hash

    def update_model_state(self, model_name: str, current_generator: Optional[BaseModelGenerator], selected_loras: list[str], lora_values: list[float], lora_loaded_names: list[str]) -> None:
        """ Update the model state with the current configuration.
        This method checks if the current model configuration is different from the previous one.
        If it is, it updates the model state and returns True.
        If the configuration is unchanged, it returns False.
        """

        self.previous_model_configuration = self.active_model_configuration
        self.previous_model_hash = self.active_model_hash
        if current_generator is not None and not self.is_reload_required(model_name, current_generator, selected_loras, lora_values, lora_loaded_names):
            print("Model configuration unchanged, skipping reload.")
            return

        selected_lora_values = [lora_values[lora_loaded_names.index(
            name)] for name in selected_loras if name in lora_loaded_names]
        active_model_configuration: ModelConfiguration = ModelConfiguration.from_lora_names_and_weights(
            model_name=model_name,
            lora_names=selected_loras,
            lora_weights=selected_lora_values
        )

        computed_model_hash = active_model_configuration._hash

        self.active_model_configuration = active_model_configuration
        self.active_model_hash = computed_model_hash


class StudioManager:
    """
    Manages the current model instance.
    This class provides methods to set and get the current model.
    """
    _instance: Optional['StudioManager'] = None
    __current_generator: Optional[Union[BaseModelGenerator, VideoBaseModelGenerator]] = None
    job_queue: VideoJobQueue = VideoJobQueue()
    settings: Settings = Settings()
    model_state: ModelState = ModelState()

    def __new__(cls):
        if cls._instance is None:
            print('Creating the StudioManager instance')
            cls._instance = super(StudioManager, cls).__new__(cls)
        return cls._instance

    @property
    def current_generator(self) -> Optional[Union[BaseModelGenerator, VideoBaseModelGenerator]]:
        """
        Property to get the current model generator instance.
        Returns None if no generator is set.
        """
        return self.__current_generator

    @current_generator.setter
    def current_generator(self, generator: Union[BaseModelGenerator, VideoBaseModelGenerator]) -> None:
        """
        Property to set the current model generator instance.
        Raises TypeError if the generator is not an instance of BaseModelGenerator or VideoBaseModelGenerator.
        """
        if not isinstance(generator, BaseModelGenerator):
            raise TypeError("Expected generator to be an instance of BaseModelGenerator")

        self.__current_generator = generator

    def unset_current_generator(self) -> None:
        """
        Delete the current model generator instance.
        This will set the current generator to None.
        """
        self.__current_generator = None  # Reset the current generator
        self.model_state = ModelState()  # Reset the model state

    def is_reload_required(self, model_name: str, selected_loras: list[str], lora_values: list[float], lora_loaded_names: list[str]) -> bool:
        return self.current_generator is None or self.model_state.is_reload_required(
            model_name=model_name,
            current_generator=self.__current_generator,
            selected_loras=selected_loras,
            lora_values=lora_values,
            lora_loaded_names=lora_loaded_names
        )

    def update_model_state(self, model_name: str, selected_loras: list[str], lora_values: list[float], lora_loaded_names: list[str]) -> None:
        self.model_state.update_model_state(
            model_name=model_name,
            current_generator=self.__current_generator,
            selected_loras=selected_loras,
            lora_values=lora_values,
            lora_loaded_names=lora_loaded_names
        )
