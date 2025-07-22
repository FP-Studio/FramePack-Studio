import numpy as np
import torch

from diffusers_helper.models.mag_cache_ratios import MAG_RATIOS_DB


class MagCache:
    """
    Implements the MagCache algorithm for skipping transformer steps during video generation.
    MagCache: Fast Video Generation with Magnitude-Aware Cache
    Zehong Ma, Longhui Wei, Feng Wang, Shiliang Zhang, Qi Tian
    https://arxiv.org/abs/2506.09045
    https://github.com/Zehong-Ma/MagCache
    PR Demo defaults were threshold=0.1, max_consectutive_skips=3, retention_ratio=0.2
    Changing defauults to threshold=0.1, max_consectutive_skips=2, retention_ratio=0.25 for quality vs speed tradeoff.
    """

    def __init__(
        self,
        model_family,
        height,
        width,
        num_steps,
        is_enabled=True,
        is_calibrating=False,
        threshold=0.1,
        max_consectutive_skips=2,
        retention_ratio=0.25,
    ):
        self.model_family = model_family
        self.height = height
        self.width = width
        self.num_steps = num_steps

        self.is_enabled = is_enabled
        self.is_calibrating = is_calibrating

        self.threshold = threshold
        self.max_consectutive_skips = max_consectutive_skips
        self.retention_ratio = retention_ratio

        # total cache statistics for all sections in the entire generation
        self.total_cache_requests = 0
        self.total_cache_hits = 0

        self.mag_ratios = self._determine_mag_ratios()

        self._init_for_every_section()

    def _init_for_every_section(self):
        self.step_index = 0
        self.steps_skipped_list = []
        # Error accumulation state
        self.accumulated_ratio = 1.0
        self.accumulated_steps = 0
        self.accumulated_err = 0
        # Statistics for calibration
        self.norm_ratio, self.norm_std, self.cos_dis = [], [], []

        self.hidden_states = None
        self.previous_residual = None

        if self.is_calibrating and self.total_cache_requests > 0:
            print(
                "WARNING: Resetting MagCache calibration stats for new section. Typically you only want one section per calibration job. Discarding calibration from previsou section."
            )

    def should_skip(self, hidden_states):
        """
        Expected to be called once per step during the forward pass, for the numer of initialized steps.
        Determines if the current step should be skipped based on estimated accumulated error.
        If the step is skipped, the hidden_states should be replaced with the output of estimate_predicted_hidden_states().

        Args:
            hidden_states: The current hidden states tensor from the transformer model.
        Returns:
            True if the step should be skipped, False otherwise
        """
        if self.step_index == 0 or self.step_index >= self.num_steps:
            self._init_for_every_section()
        self.total_cache_requests += 1
        self.hidden_states = hidden_states.clone()  # Is clone needed?

        if self.is_calibrating:
            print(
                "######################### Calibrating MagCache #########################"
            )
            return False

        should_skip_forward = False
        if (
            self.step_index >= int(self.retention_ratio * self.num_steps)
            and self.step_index >= 1
        ):  # keep first retention_ratio steps
            cur_mag_ratio = self.mag_ratios[self.step_index]
            self.accumulated_ratio = self.accumulated_ratio * cur_mag_ratio
            cur_skip_err = np.abs(1 - self.accumulated_ratio)
            self.accumulated_err += cur_skip_err
            self.accumulated_steps += 1
            # RT_BORG: Per my conversation with Zehong Ma, this 0.06 could potentially be exposed as another tunable param.
            if (
                self.accumulated_err <= self.threshold
                and self.accumulated_steps <= self.max_consectutive_skips
                and np.abs(1 - cur_mag_ratio) <= 0.06
            ):
                should_skip_forward = True
            else:
                self.accumulated_ratio = 1.0
                self.accumulated_steps = 0
                self.accumulated_err = 0

        if should_skip_forward:
            self.total_cache_hits += 1
            self.steps_skipped_list.append(self.step_index)
        # Increment for next step
        self.step_index += 1
        if self.step_index == self.num_steps:
            self.step_index = 0

        return should_skip_forward

    def estimate_predicted_hidden_states(self):
        """
        Should be called if and only if should_skip() returned True for the current step.
        Estimates the hidden states for the current step based on the previous hidden states and residual.

        Returns:
            The estimated hidden states tensor.
        """
        return self.hidden_states + self.previous_residual

    def update_hidden_states(self, model_prediction_hidden_states):
        """
        If and only if should_skip() returned False for the current step, the denoising layers should have been run,
        and this function should be called to compute and store the residual for future steps.

        Args:
            model_prediction_hidden_states: The hidden states tensor output from running the denoising layers.
        """

        current_residual = model_prediction_hidden_states - self.hidden_states
        if self.is_calibrating:
            self._update_calibration_stats(current_residual)

        self.previous_residual = current_residual

    def _update_calibration_stats(self, current_residual):
        if self.step_index >= 1:
            norm_ratio = (
                (
                    current_residual.norm(dim=-1) / self.previous_residual.norm(dim=-1)
                ).mean()
            ).item()
            norm_std = (
                (current_residual.norm(dim=-1) / self.previous_residual.norm(dim=-1))
                .std()
                .item()
            )
            cos_dis = (
                (
                    1
                    - torch.nn.functional.cosine_similarity(
                        current_residual, self.previous_residual, dim=-1, eps=1e-8
                    )
                )
                .mean()
                .item()
            )
            self.norm_ratio.append(round(norm_ratio, 5))
            self.norm_std.append(round(norm_std, 5))
            self.cos_dis.append(round(cos_dis, 5))
            # print(f"time: {self.step_index}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")

        self.step_index += 1
        if self.step_index == self.num_steps:
            print("norm ratio")
            print(self.norm_ratio)
            print("norm std")
            print(self.norm_std)
            print("cos_dis")
            print(self.cos_dis)
            self.step_index = 0

    def _determine_mag_ratios(self):
        """
        Determines the magnitude ratios by finding the closest resolution and step count
        in the pre-calibrated database.

        Returns:
            A numpy array of magnitude ratios for the specified configuration, or None if not found.
        """
        if self.is_calibrating:
            return None
        try:
            # Find the closest available resolution group for the given model family
            resolution_groups = MAG_RATIOS_DB[self.model_family]
            available_resolutions = list(resolution_groups.keys())
            if not available_resolutions:
                raise ValueError("No resolutions defined for this model family.")

            avg_resolution = (self.height + self.width) / 2.0
            closest_resolution_key = min(
                available_resolutions, key=lambda r: abs(r - avg_resolution)
            )

            # Find the closest available step count for the given model/resolution
            steps_group = resolution_groups[closest_resolution_key]
            available_steps = list(steps_group.keys())
            if not available_steps:
                raise ValueError(
                    f"No step counts defined for resolution {closest_resolution_key}."
                )
            closest_steps = min(available_steps, key=lambda x: abs(x - self.num_steps))
            base_ratios = steps_group[closest_steps]
            if closest_steps == self.num_steps:
                print(
                    f"MagCache: Found ratios for {self.model_family}, resolution group {closest_resolution_key} ({self.width}x{self.height}), {self.num_steps} steps."
                )
                return base_ratios
            print(
                f"MagCache: Using ratios from {self.model_family}, resolution group {closest_resolution_key} ({self.width}x{self.height}), {closest_steps} steps and interpolating to {self.num_steps} steps."
            )
            return self._nearest_step_interpolation(base_ratios, self.num_steps)
        except KeyError:
            # This will catch if model_family is not in MAG_RATIOS_DB
            print(
                f"Warning: MagCache not calibrated for model family '{self.model_family}'. MagCache will not be used."
            )
            self.is_enabled = False
        except (ValueError, TypeError) as e:
            # This will catch errors if resolution keys or step keys are not numbers, or if groups are empty.
            print(
                f"Warning: Error processing MagCache DB for model family '{self.model_family}': {e}. MagCache will not be used."
            )
            self.is_enabled = False
        return None

    # Nearest interpolation function for MagCache mag_ratios
    @staticmethod
    def _nearest_step_interpolation(src_array, target_length):
        src_length = len(src_array)
        if target_length == 1:
            return np.array([src_array[-1]])

        scale = (src_length - 1) / (target_length - 1)
        mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
        return src_array[mapped_indices]

    def append_calibration_to_file(self, output_file):
        """
        Appends tab delimited calibration data (model_family,width,height,norm_ratio) to output_file.
        """
        if not self.is_calibrating or not self.norm_ratio:
            print("Calibration data can only be appended after calibration.")
            return False
        try:
            with open(output_file, "a") as f:
                # Format the data as a string
                calibration_set = f"{self.model_family}\t{self.width}\t{self.height}\t{self.num_steps}"
                # data_string = f"{calibration_set}\t{self.norm_ratio}"
                entry_string = f"{calibration_set}\t{self.num_steps}: np.array([1.0] + {self.norm_ratio}),"
                # Append the data to the file
                f.write(entry_string + "\n")
            print(f"Calibration data appended to {output_file}")
            return True
        except Exception as e:
            print(f"Error appending calibration data: {e}")
            return False
