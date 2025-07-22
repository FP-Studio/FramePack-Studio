import logging
from typing import (
    Optional,
    Union,
)

from diffusers_helper.lora_utils_kohya_ss.enums import LoraLoader
from modules.generators.model_configuration import ModelConfiguration
from .generators import BaseModelGenerator, VideoBaseModelGenerator
from .settings import Settings
from .video_queue import VideoJobQueue

# cSpell: ignore loras


class ModelState:
    """
    Class to track the state of model configurations.
    This class keeps track of the previous model configuration and its hash.
    """

    previous_model_configuration: Optional[ModelConfiguration]
    active_model_configuration: Optional[ModelConfiguration]
    __logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self):
        self.previous_model_configuration: Optional[ModelConfiguration] = None
        self.active_model_configuration: Optional[ModelConfiguration] = None

    @property
    def active_model_hash(self) -> str:
        """
        Returns the hash of the active model configuration.
        If no active model configuration is set, returns an empty string.
        """
        return (
            self.active_model_configuration._hash
            if self.active_model_configuration
            else "active_model_hash"
        )

    @property
    def previous_model_hash(self) -> str:
        """
        Returns the hash of the previous model configuration.
        If no previous model configuration is set, returns an empty string.
        """
        return (
            self.previous_model_configuration._hash
            if self.previous_model_configuration
            else "previous_model_hash"
        )

    def is_reload_required(
        self,
        model_name: str,
        selected_loras: list[str],
        lora_values: list[float],
        lora_loaded_names: list[str],
        lora_loader: str | LoraLoader,
    ) -> bool:
        """
        Check if a reload is required based on the current model state.
        This method checks if the current model configuration is different from the previous one.
        Args:
            model_name: The name of the model to check.
            selected_loras: List of selected LoRA names.
            lora_values: List of LoRA values corresponding to the selected LoRAs.
            lora_loaded_names: List of names of LoRAs that are currently loaded.
        Returns:
            True if a reload is required, otherwise False.
        """
        # queue load only sends the exact selected_loras and lora_values while other functions may send lora_values for all lora_loaded_names
        # remove the condition if we update to always send the matching selected_loras and lora_values
        selected_lora_values = (
            lora_values
            if len(selected_loras) == len(lora_values)
            else [
                lora_values[lora_loaded_names.index(name)]
                for name in selected_loras
                if name in lora_loaded_names
            ]
        )

        active_model_configuration: ModelConfiguration = (
            ModelConfiguration.from_lora_names_and_weights(
                model_name=model_name,
                lora_names=selected_loras,
                lora_weights=selected_lora_values,
                lora_loader=lora_loader,
            )
        )

        return active_model_configuration._hash != self.active_model_hash

    def update_model_state(
        self,
        current_generator: Optional[BaseModelGenerator],
        selected_loras: list[str],
        lora_values: list[float],
        lora_loaded_names: list[str],
    ) -> None:
        """Update the model state with the current configuration.
        This method checks if the current model configuration is different from the previous one.
        If it is, it updates the model state and returns True.
        If the configuration is unchanged, it returns False.
        """

        assert current_generator is not None, (
            "current_generator must be set when updating model state"
        )
        self.previous_model_configuration = self.active_model_configuration

        if not self.is_reload_required(
            current_generator.model_name,
            selected_loras,
            lora_values,
            lora_loaded_names,
            current_generator.settings.lora_loader,
        ):
            self.__logger.debug("Model configuration unchanged, skipping reload.")
            return

        # queue load only sends the exact selected_loras and lora_values while other functions may send lora_values for all lora_loaded_names
        # remove the condition if we update to always send the matching selected_loras and lora_values
        selected_lora_values = (
            lora_values
            if len(selected_loras) == len(lora_values)
            else [
                lora_values[lora_loaded_names.index(name)]
                for name in selected_loras
                if name in lora_loaded_names
            ]
        )
        active_model_configuration: ModelConfiguration = (
            ModelConfiguration.from_lora_names_and_weights(
                model_name=current_generator.model_name,
                lora_names=selected_loras,
                lora_weights=selected_lora_values,
                lora_loader=current_generator.settings.lora_loader,
            )
        )

        self.active_model_configuration = active_model_configuration


class StudioManager:
    """
    Singleton class to manage the current model instance and its state.
    """

    _instance: Optional["StudioManager"] = None
    __current_generator: Optional[
        Union[BaseModelGenerator, VideoBaseModelGenerator]
    ] = None
    job_queue: VideoJobQueue = VideoJobQueue()
    settings: Settings = Settings()
    model_state: ModelState = ModelState()
    __logger: logging.Logger = logging.getLogger(__name__)

    def __new__(cls):
        if cls._instance is None:
            cls.__logger.debug("Creating the StudioManager instance")
            cls._instance = super(StudioManager, cls).__new__(cls)
        return cls._instance

    @property
    def current_generator(
        self,
    ) -> Optional[Union[BaseModelGenerator, VideoBaseModelGenerator]]:
        """
        Property to get the current model generator instance.
        Returns None if no generator is set.
        """
        return self.__current_generator

    @current_generator.setter
    def current_generator(self, generator: BaseModelGenerator) -> None:
        """
        Property to set the current model generator instance.
        Raises TypeError if the generator is not an instance of BaseModelGenerator or VideoBaseModelGenerator.
        """
        assert isinstance(generator, BaseModelGenerator), (
            "Expected generator to be an instance of BaseModelGenerator"
        )

        self.__current_generator = generator

    def unset_current_generator(self) -> None:
        """
        Delete the current model generator instance.
        This will set the current generator to None.
        """
        self.__current_generator = None  # Reset the current generator
        self.model_state = ModelState()  # Reset the model state

    def is_reload_required(
        self,
        model_name: str,
        selected_loras: list[str],
        lora_values: list[float],
        lora_loaded_names: list[str],
    ) -> bool:
        """
        Check if a reload is required based on the current model state.
        This method checks if the current model generator is None or if the settings require a reload.
        It also checks if the model state has changed based on the provided parameters.
        Currently it does not check against the base model, so it will always return True if the model type has changed at all.

        Args:
            model_name: The name of the model to check.
            selected_loras: List of selected LoRA names.
            lora_values: List of LoRA values corresponding to the selected LoRAs.
            lora_loaded_names: List of names of LoRAs that are currently loaded.
        Returns:
            True if a reload is required, otherwise False.
        """
        if self.current_generator is None:
            self.__logger.debug("No current generator set, reload is required.")
            return True
        if not self.settings.reuse_model_instance:
            self.__logger.debug("Model instance reuse is disabled, reload is required.")
            return True

        return self.model_state.is_reload_required(
            model_name=model_name,
            selected_loras=selected_loras,
            lora_values=lora_values,
            lora_loaded_names=lora_loaded_names,
            lora_loader=self.settings.lora_loader,
        )

    def update_model_state(
        self,
        selected_loras: list[str],
        lora_values: list[float],
        lora_loaded_names: list[str],
    ) -> None:
        assert self.current_generator is not None, (
            "current_generator must be set when updating model state"
        )
        self.model_state.update_model_state(
            current_generator=self.__current_generator,
            selected_loras=selected_loras,
            lora_values=lora_values,
            lora_loaded_names=lora_loaded_names,
        )
