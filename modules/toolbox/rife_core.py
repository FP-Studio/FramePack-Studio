import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path
import os
import gc
from huggingface_hub import snapshot_download

from .RIFE.RIFE_HDv3 import Model as RIFEBaseModel
from .message_manager import MessageManager
import devicetorch

# Get the directory of the current script (rife_core.py)
_MODULE_DIR = Path(
    os.path.dirname(os.path.abspath(__file__))
)  # __file__ gives path to current script

# MODEL_RIFE_PATH = "model_rife" # OLD - this is relative to CWD
MODEL_RIFE_PATH = _MODULE_DIR / "model_rife"  # NEW - relative to this script's location
RIFE_MODEL_FILENAME = "flownet.pkl"


class RIFEHandler:
    def __init__(self, message_manager: MessageManager = None):
        self.message_manager = message_manager if message_manager else MessageManager()
        self.model_dir = Path(
            MODEL_RIFE_PATH
        )  # Path() constructor handles Path objects correctly
        self.model_file_path = self.model_dir / RIFE_MODEL_FILENAME
        self.rife_model = None

    def _log(self, message, level="INFO"):
        # Helper for logging using the MessageManager
        if level.upper() == "ERROR":
            self.message_manager.add_error(f"RIFEHandler: {message}")
        elif level.upper() == "WARNING":
            self.message_manager.add_warning(f"RIFEHandler: {message}")
        else:
            self.message_manager.add_message(f"RIFEHandler: {message}")

    def _ensure_model_downloaded_and_loaded(self) -> bool:
        if self.rife_model is not None:
            self._log("RIFE model already loaded.")
            return True

        # self.model_dir is now an absolute path
        if not self.model_dir.exists():
            os.makedirs(self.model_dir, exist_ok=True)
            self._log(f"Created RIFE model directory: {self.model_dir}")

        # self.model_file_path is now an absolute path
        if not self.model_file_path.exists():
            self._log("RIFE model weights not found. Downloading...")
            try:
                snapshot_download(
                    repo_id="AlexWortega/RIFE",
                    allow_patterns=["*.pkl", "*.pth"],
                    local_dir=self.model_dir,  # Pass the absolute path
                    local_dir_use_symlinks=False,
                )
                if self.model_file_path.exists():
                    self._log("RIFE model weights downloaded successfully.")
                else:
                    self._log(
                        f"RIFE model download completed, but {RIFE_MODEL_FILENAME} not found in {self.model_dir}. Check allow_patterns and repo structure.",
                        "ERROR",
                    )
                    return False
            except Exception as e:
                self._log(f"Failed to download RIFE model weights: {e}", "ERROR")
                return False

        if not self.model_file_path.exists():
            self._log(
                f"RIFE model file {self.model_file_path} does not exist. Cannot load model.",
                "ERROR",
            )
            return False

        try:
            self._log(
                f"Loading RIFE model from {self.model_dir}..."
            )  # self.model_dir is absolute
            #current_device_str = devicetorch.get(torch)
            self.rife_model = RIFEBaseModel(local_rank=-1)

            self.rife_model.load_model(
                str(self.model_dir), -1
            )  # str(self.model_dir) is absolute
            self.rife_model.eval()
            self._log("RIFE model loaded successfully to its determined device.")
            return True
        except Exception as e:
            self._log(f"Failed to load RIFE model: {e}", "ERROR")
            import traceback

            self._log(f"Traceback: {traceback.format_exc()}", "ERROR")
            self.rife_model = None
            return False

    def unload_model(self):
        if self.rife_model is not None:
            self._log("Unloading RIFE model...")
            del self.rife_model
            self.rife_model = None
            devicetorch.empty_cache(torch)
            gc.collect()
            self._log("RIFE model unloaded and memory cleared.")
        else:
            self._log("RIFE model not loaded, no need to unload.")

    def interpolate_between_frames(
        self, frame1_np: np.ndarray, frame2_np: np.ndarray
    ) -> np.ndarray | None:
        if self.rife_model is None:
            self._log(
                "RIFE model not loaded. Call _ensure_model_downloaded_and_loaded() before interpolation.",
                "ERROR",
            )
            return None

        try:
            img0_tensor = to_tensor(frame1_np).unsqueeze(0)
            img1_tensor = to_tensor(frame2_np).unsqueeze(0)

            img0 = devicetorch.to(torch, img0_tensor)
            img1 = devicetorch.to(torch, img1_tensor)

            required_multiple = 32
            h_orig, w_orig = img0.shape[2], img0.shape[3]
            pad_h = (required_multiple - h_orig % required_multiple) % required_multiple
            pad_w = (required_multiple - w_orig % required_multiple) % required_multiple

            if pad_h > 0 or pad_w > 0:
                img0 = torch.nn.functional.pad(
                    img0, (0, pad_w, 0, pad_h), mode="replicate"
                )
                img1 = torch.nn.functional.pad(
                    img1, (0, pad_w, 0, pad_h), mode="replicate"
                )

            with torch.no_grad():
                middle_frame_tensor = self.rife_model.inference(img0, img1, scale=1.0)

            if pad_h > 0 or pad_w > 0:
                middle_frame_tensor = middle_frame_tensor[:, :, :h_orig, :w_orig]

            middle_frame_pil = to_pil_image(middle_frame_tensor.squeeze(0).cpu())
            return np.array(middle_frame_pil)

        except Exception as e:
            self._log(f"Error during RIFE frame interpolation: {e}", "ERROR")
            import traceback

            self._log(f"Traceback: {traceback.format_exc()}", "ERROR")
            if "out of memory" in str(e).lower():
                devicetorch.empty_cache(torch)
            return None
