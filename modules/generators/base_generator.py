import torch
import os  # required for os.path
from abc import ABC, abstractmethod
from dataclasses import asdict
from diffusers_helper import lora_utils
from typing import List, Optional, cast
from pathlib import Path

from ..settings import Settings
from .model_configuration import ModelConfiguration


class BaseModelGenerator(ABC):
    """
    Base class for model generators.
    This defines the common interface that all model generators must implement.
    """

    def __init__(self,
                 text_encoder,
                 text_encoder_2,
                 tokenizer,
                 tokenizer_2,
                 vae,
                 image_encoder,
                 feature_extractor,
                 high_vram=False,
                 prompt_embedding_cache=None,
                 settings: Settings | None = None,
                 offline=False):  # NEW: offline flag
        """
        Initialize the base model generator.

        Args:
            text_encoder: The text encoder model
            text_encoder_2: The second text encoder model
            tokenizer: The tokenizer for the first text encoder
            tokenizer_2: The tokenizer for the second text encoder
            vae: The VAE model
            image_encoder: The image encoder model
            feature_extractor: The feature extractor
            high_vram: Whether high VRAM mode is enabled
            prompt_embedding_cache: Cache for prompt embeddings
            settings: Application settings
            offline: Whether to run in offline mode for model loading
        """
        self.model_name: str
        self.model_path: str
        self.model_repo_id_for_cache: str

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.vae = vae
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.high_vram = high_vram
        self.prompt_embedding_cache = prompt_embedding_cache or {}
        self.settings: Settings = settings if settings is not None else Settings()
        self.offline = offline
        self.transformer = None
        self.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        self.previous_model_hash: str = ""
        self.previous_model_configuration: ModelConfiguration | None = None

    @abstractmethod
    def load_model(self) -> torch.nn.Module:
        """
        Load the transformer model.
        This method should be implemented by each specific model generator.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model.
        This method should be implemented by each specific model generator.
        """
        pass

    @staticmethod
    def _get_snapshot_hash_from_refs(model_repo_id_for_cache: str) -> str | None:
        """
        Reads the commit hash from the refs/main file for a given model in the HF cache.
        Args:
            model_repo_id_for_cache (str): The model ID formatted for cache directory names
                                           (e.g., "models--lllyasviel--FramePackI2V_HY").
        Returns:
            str: The commit hash if found, otherwise None.
        """
        hf_home_dir = os.environ.get('HF_HOME')
        if not hf_home_dir:
            print("Warning: HF_HOME environment variable not set. Cannot determine snapshot hash.")
            return None

        refs_main_path = os.path.join(hf_home_dir, 'hub', model_repo_id_for_cache, 'refs', 'main')
        if os.path.exists(refs_main_path):
            try:
                with open(refs_main_path, 'r') as f:
                    print(f"Offline mode: Reading snapshot hash from: {refs_main_path}")
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read snapshot hash from {refs_main_path}: {e}")
                return None
        else:
            print(f"Warning: refs/main file not found at {refs_main_path}. Cannot determine snapshot hash.")
            return None

    def _get_offline_load_path(self) -> str:
        """
        Returns the local snapshot path for offline loading if available.
        Falls back to the default self.model_path if local snapshot can't be found.
        Relies on self.model_repo_id_for_cache and self.model_path being set by subclasses.
        """
        # Ensure necessary attributes are set by the subclass
        if not hasattr(self, 'model_repo_id_for_cache') or not self.model_repo_id_for_cache:
            print(
                f"Warning: model_repo_id_for_cache not set in {self.__class__.__name__}. Cannot determine offline path.")
            # Fallback to model_path if it exists, otherwise None
            return str(getattr(self, 'model_path', None))

        if not hasattr(self, 'model_path') or not self.model_path:
            print(
                f"Warning: model_path not set in {self.__class__.__name__}. Cannot determine fallback for offline path.")
            return None

        snapshot_hash = self._get_snapshot_hash_from_refs(self.model_repo_id_for_cache)
        hf_home = os.environ.get('HF_HOME')

        if snapshot_hash and hf_home:
            specific_snapshot_path = os.path.join(
                hf_home, 'hub', self.model_repo_id_for_cache, 'snapshots', snapshot_hash
            )
            if os.path.isdir(specific_snapshot_path):
                return specific_snapshot_path

        # If snapshot logic fails or path is not a dir, fallback to the default model path
        return self.model_path

    def unload_loras(self):
        """
        Unload all LoRAs from the transformer model.
        """
        if self.transformer is not None:
            print(f"Unloading all LoRAs from {self.get_model_name()} model")
            self.transformer = lora_utils.unload_all_loras(self.transformer)
            self.verify_lora_state("After unloading LoRAs")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def verify_lora_state(self, label=""):
        """
        Debug function to verify the state of LoRAs in the transformer model.
        """
        if self.transformer is None:
            print(f"[{label}] Transformer is None, cannot verify LoRA state")
            return

        has_loras = False
        if hasattr(self.transformer, 'peft_config'):
            adapter_names = list(self.transformer.peft_config.keys()) if self.transformer.peft_config else []
            if adapter_names:
                has_loras = True
                print(f"[{label}] Transformer has LoRAs: {', '.join(adapter_names)}")
            else:
                print(f"[{label}] Transformer has no LoRAs in peft_config")
        else:
            print(f"[{label}] Transformer has no peft_config attribute")

        # Check for any LoRA modules
        for name, module in self.transformer.named_modules():
            if hasattr(module, 'lora_A') and module.lora_A:
                has_loras = True
                # print(f"[{label}] Found lora_A in module {name}")
            if hasattr(module, 'lora_B') and module.lora_B:
                has_loras = True
                # print(f"[{label}] Found lora_B in module {name}")

        if not has_loras:
            print(f"[{label}] No LoRA components found in transformer")

    def move_lora_adapters_to_device(self, target_device):
        """
        Move all LoRA adapters in the transformer model to the specified device.
        This handles the PEFT implementation of LoRA.
        """
        if self.transformer is None:
            return

        print(f"Moving all LoRA adapters to {target_device}")

        # First, find all modules with LoRA adapters
        lora_modules = []
        for name, module in self.transformer.named_modules():
            if hasattr(module, 'active_adapter') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules.append((name, module))

        # Now move all LoRA components to the target device
        for name, module in lora_modules:
            # Get the active adapter name
            active_adapter = module.active_adapter

            # Move the LoRA layers to the target device
            if active_adapter is not None:
                if isinstance(module.lora_A, torch.nn.ModuleDict):
                    # Handle ModuleDict case (PEFT implementation)
                    for adapter_name in list(module.lora_A.keys()):
                        # Move lora_A
                        if adapter_name in module.lora_A:
                            module.lora_A[adapter_name] = module.lora_A[adapter_name].to(target_device)

                        # Move lora_B
                        if adapter_name in module.lora_B:
                            module.lora_B[adapter_name] = module.lora_B[adapter_name].to(target_device)

                        # Move scaling
                        if hasattr(module, 'scaling') and isinstance(module.scaling, dict) and adapter_name in module.scaling:
                            if isinstance(module.scaling[adapter_name], torch.Tensor):
                                module.scaling[adapter_name] = module.scaling[adapter_name].to(target_device)
                else:
                    # Handle direct attribute case
                    if hasattr(module, 'lora_A') and module.lora_A is not None:
                        module.lora_A = module.lora_A.to(target_device)
                    if hasattr(module, 'lora_B') and module.lora_B is not None:
                        module.lora_B = module.lora_B.to(target_device)
                    if hasattr(module, 'scaling') and module.scaling is not None:
                        if isinstance(module.scaling, torch.Tensor):
                            module.scaling = module.scaling.to(target_device)

        print(f"Moved all LoRA adapters to {target_device}")

    @abstractmethod
    def get_latent_paddings(self, total_latent_sections) -> list[int]:
        raise NotImplementedError(
            "get_latent_paddings must be implemented by the specific model generator subclass.")

    @abstractmethod
    def format_position_description(self, total_generated_latent_frames, current_pos, original_pos, current_prompt) -> str:
        raise NotImplementedError(
            "format_position_description must be implemented by the specific model generator subclass.")

    @abstractmethod
    def get_real_history_latents(self, history_latents: torch.Tensor, total_generated_latent_frames: int) -> torch.Tensor:
        raise NotImplementedError(
            "get_real_history_latents must be implemented by the specific model generator subclass.")

    @abstractmethod
    def update_history_latents(self, history_latents: torch.Tensor, generated_latents: torch.Tensor) -> torch.Tensor:
        """
        Update the history latents with the generated latents.
        This method should be implemented by each specific model generator.

        Args:
            history_latents: The history latents
            generated_latents: The generated latents

        Returns:
            The updated history latents
        """
        raise NotImplementedError(
            "update_history_latents must be implemented by the specific model generator subclass.")

    def __compute_lora_state_hash(self, lora_config: ModelConfiguration) -> str:
        """
        Compute a simple hash representing the current state of LoRA adapters in the transformer.
        This can be used to detect changes in loaded LoRAs.
        """
        import hashlib
        # md5 should be sufficient for this purpose
        m = hashlib.md5()

        if self.transformer is None:
            # Should not happen - return a unique value
            print("Warning: Transformer is None when computing LoRA state hash.")
            from time import time
            m.update(str(time() * 1000).encode('utf-8'))

        import json
        m.update(json.dumps(asdict(lora_config), sort_keys=True).encode('utf-8'))
        return m.hexdigest()

    def load_loras(self, selected_loras: List[str], lora_folder: str, lora_loaded_names: List[str], lora_values: Optional[List[float]] = None):
        """
        Load LoRAs into the transformer model and applies their weights.

        Args:
            selected_loras: List of LoRA base names to load (e.g., ["lora_A", "lora_B"]).
            lora_folder: Path to the folder containing the LoRA files.
            lora_loaded_names: The master list of ALL available LoRA names, used for correct weight indexing.
            lora_values: A list of strength values corresponding to lora_loaded_names.
        """
        if not selected_loras:
            # Only unload at this point if no LoRAs are selected
            self.unload_loras()
            print("No LoRAs selected, skipping loading.")
            return

        if self.transformer is None:
            print("Transformer model is None, cannot load LoRAs.")
            return

        if lora_values is None:
            lora_values = []

        selected_lora_values = [lora_values[lora_loaded_names.index(
            name)] for name in selected_loras if name in lora_loaded_names]
        print(f"Loading LoRAs: {selected_loras} with values: {selected_lora_values}")

        active_model_configuration: ModelConfiguration = ModelConfiguration.from_lora_names_and_weights(
            self.get_model_name(), selected_loras, selected_lora_values)

        active_model_hash = self.__compute_lora_state_hash(active_model_configuration)
        if active_model_hash == self.previous_model_hash:
            # will never short circuit with current architecture since we create a new
            print("Model configuration unchanged, skipping reload.")
            return

        print(
            f"Previous LoRA config: {self.previous_model_configuration}, Current LoRA config: {active_model_configuration}")
        print(f"Previous LoRA hash: {self.previous_model_hash}, Current LoRA hash: {active_model_hash}")

        self.previous_model_hash = active_model_hash
        self.previous_model_configuration = active_model_configuration

        lora_dir = Path(lora_folder)

        if self.settings.get("kohya_ss_lora_support", False):
            from diffusers_helper.lora_utils_kohya_ss.lora_loader import load_and_apply_lora
            print(f"Loading LoRAs using kohya_ss loader from {lora_dir}")

            def _find_model_files(model_path):
                """Get state dictionary file from specified model path
                This is undesirable as it depends on Diffusers implementation."""
                import glob
                model_root = os.environ['HF_HOME']  # './hf_download'?
                subdir = os.path.join(model_root, 'hub', 'models--' + model_path.replace('/', '--'))
                model_files = glob.glob(os.path.join(subdir, '**', '*.safetensors'), recursive=True)
                model_files.sort()
                return model_files
            try:
                model_files = _find_model_files(self.model_path)
                print(f"LoRA -> Found model files: {model_files}")
                lora_paths = [
                    # not sure why the full path is not passed around and potentially trimmed for the interface display
                    str(lora_dir / f"{lora_setting.name}.safetensors")
                    if Path(lora_dir / f"{lora_setting.name}.safetensors").exists()
                    else str(lora_dir / f"{lora_setting.name}.pt")  # hopefully .pt is the correct extension.
                    for lora_setting in active_model_configuration.settings.lora_settings
                ]
                lora_scales: list[float] = [lora_setting.weight for lora_setting in active_model_configuration.settings.lora_settings]
                print(f'Lora paths: {lora_paths}')
                if not lora_paths:
                    raise ValueError("No valid LoRA paths found for the selected LoRAs.")

                state_dict = load_and_apply_lora(
                    model_files=model_files,
                    lora_paths=lora_paths,
                    lora_scales=lora_scales,
                    fp8_enabled=cast(bool, self.settings.get("fp8", False)),
                    device=self.gpu if torch.cuda.is_available() else self.cpu
                )
                print("Loading state dict into transformer...")
                missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, assign=True, strict=True)

                if missing_keys:
                    print(f"Warning: Missing keys when loading LoRA state dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys when loading LoRA state dict: {unexpected_keys}")

                state_dict_size: int = 0
                try:
                    state_dict_size = sum(param.numel() * param.element_size()
                                          for param in state_dict.values() if hasattr(param, 'numel'))
                    print(f"State dictionary size: {state_dict_size / (1024**3):.2f} GB")
                except:
                    pass

                try:
                    del state_dict
                    import gc
                    gc.collect()
                    print(f"Freed state dictionary size: {state_dict_size / (1024**3):.2f} GB")
                except:
                    print("Could not free state dictionary from memory.")

            except Exception as e:
                import traceback
                print(f"Error loading LoRAs with kohya_ss loader: {e}")
                traceback.print_exc()
        else:
            self.unload_loras()
            adapter_names = []
            strengths = []

            for idx, lora_base_name in enumerate(selected_loras):
                lora_file = None
                for ext in (".safetensors", ".pt"):
                    candidate_path_relative = f"{lora_base_name}{ext}"
                    candidate_path_full = lora_dir / candidate_path_relative
                    if candidate_path_full.is_file():
                        lora_file = candidate_path_relative
                        break

                if not lora_file:
                    print(f"Warning: LoRA file for base name '{lora_base_name}' not found; skipping.")
                    continue

                print(f"Loading LoRA from '{lora_file}'...")

                self.transformer, adapter_name = lora_utils.load_lora(self.transformer, lora_dir, lora_file)
                adapter_names.append(adapter_name)

                weight = 1.0
                if lora_values:
                    try:
                        master_list_idx = lora_loaded_names.index(lora_base_name)
                        if master_list_idx < len(lora_values):
                            weight = float(lora_values[master_list_idx])
                        else:
                            print(f"Warning: Index mismatch for '{lora_base_name}'. Defaulting to 1.0.")
                    except ValueError:
                        print(f"Warning: LoRA '{lora_base_name}' not found in master list. Defaulting to 1.0.")

                strengths.append(weight)

            if adapter_names:
                print(f"Activating adapters: {adapter_names} with strengths: {strengths}")
                lora_utils.set_adapters(self.transformer, adapter_names, strengths)

        self.verify_lora_state("After completing load_loras")
