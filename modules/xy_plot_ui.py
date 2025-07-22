import gradio as gr
import numpy as np
import re
import itertools

from modules.video_queue import JobType

# --- Helper Dictionaries & Functions ---

xy_plot_axis_options = {
    # "type": [
    #     "dropdown(checkboxGroup), textbox or number",
    #     "empty if textbox, dtype if number, [] if dropdown",
    #     "standard values",
    #     "True if multi axis - like prompt replace, False is only on one axis - like steps"
    # ],
    "Nothing": ["nothing", "", "", True],
    "Model type": ["dropdown", ["Original", "F1"], ["Original", "F1"], False],
    "End frame influence": ["number", "float", "0.05-0.95[3]", False],
    "Latent type": [
        "dropdown",
        ["Black", "White", "Noise", "Green Screen"],
        ["Black", "Noise"],
        False,
    ],
    "Prompt add": ["textbox", "", "", True],
    "Prompt replace": ["textbox", "", "", True],
    "Blend sections": ["number", "int", "3-7 [3]", False],
    "Steps": ["number", "int", "15-30 [3]", False],
    "Seed": ["number", "int", "1000-10000 [3]", False],
    "Use teacache": ["dropdown", [True, False], [True, False], False],
    "TeaCache steps": ["number", "int", "5-25 [3]", False],
    "TeaCache rel_l1_thresh": ["number", "float", "0.01-0.3 [3]", False],
    "Use MagCache": ["dropdown", [True, False], [True, False], False],
    "MagCache Threshold": ["number", "float", "0.01-1.0 [3]", False],
    "MagCache Max Consecutive Skips": ["number", "int", "1-5 [3]", False],
    "MagCache Retention Ratio": ["number", "float", "0.0-1.0 [3]", False],
    # "CFG": ["number", "float", "", False],
    "Distilled CFG Scale": ["number", "float", "5-15 [3]", False],
    # "RS": ["number", "float", "", False],
    # "Use weighted embeddings": ["dropdown", [True, False], [True, False], False],
}

text_to_base_keys = {
    "Model type": "model_type",
    "End frame influence": "end_frame_strength_original",
    "Latent type": "latent_type",
    "Prompt add": "prompt",
    "Prompt replace": "prompt",
    "Blend sections": "blend_sections",
    "Steps": "steps",
    "Seed": "seed",
    "Use teacache": "use_teacache",
    "TeaCache steps": "teacache_num_steps",
    "TeaCache rel_l1_thresh": "teacache_rel_l1_thresh",
    "Use MagCache": "use_magcache",
    "MagCache Threshold": "magcache_threshold",
    "MagCache Max Consecutive Skips": "magcache_max_consecutive_skips",
    "MagCache Retention Ratio": "magcache_retention_ratio",
    "Latent window size": "latent_window_size",
    # "CFG": "",
    "Distilled CFG Scale": "gs",
    # "RS": "",
    # "Use weighted embeddings": "",
}


def xy_plot_parse_input(text):
    text = text.strip()
    if "," in text:
        return [x.strip() for x in text.split(",")]
    match = re.match(r"^\s*(-?\d*\.?\d*)\s*-\s*(-?\d*\.?\d*)\s*\[\s*(\d+)\s*\]$", text)
    if match:
        start, end, count = map(float, match.groups())
        result = np.linspace(start, end, int(count))
        if np.allclose(result, np.round(result)):
            result = np.round(result).astype(int)
        return result.tolist()
    return []


def xy_plot_process(
    job_queue,
    settings,  # Added explicit dependencies
    model_type,
    input_image,
    end_frame_image_original,
    end_frame_strength_original,
    latent_type,
    prompt,
    blend_sections,
    steps,
    total_second_length,
    resolutionW,
    resolutionH,
    seed,
    randomize_seed,
    use_teacache,
    teacache_num_steps,
    teacache_rel_l1_thresh,
    use_magcache,
    magcache_threshold,
    magcache_max_consecutive_skips,
    magcache_retention_ratio,
    latent_window_size,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    mp4_crf,
    axis_x_switch,
    axis_x_value_text,
    axis_x_value_dropdown,
    axis_y_switch,
    axis_y_value_text,
    axis_y_value_dropdown,
    axis_z_switch,
    axis_z_value_text,
    axis_z_value_dropdown,
    selected_loras,
    *lora_slider_values,
):
    # print(model_type, input_image, latent_type,
    #     prompt, blend_sections, steps, total_second_length,
    #     resolutionW, resolutionH, seed, randomize_seed, use_teacache,
    #     latent_window_size, cfg, gs, rs, gpu_memory_preservation,
    #     mp4_crf,
    #     axis_x_switch, axis_x_value_text, axis_x_value_dropdown,
    #     axis_y_switch, axis_y_value_text, axis_y_value_dropdown,
    #     axis_z_switch, axis_z_value_text, axis_z_value_dropdown, sep=", ")
    if (
        axis_x_switch == "Nothing"
        and axis_y_switch == "Nothing"
        and axis_z_switch == "Nothing"
    ):
        return "Not selected any axis for plot", gr.update()
    if (
        axis_x_switch == "Nothing" or axis_y_switch == "Nothing"
    ) and axis_z_switch != "Nothing":
        return "For using Z axis, first use X and Y axis", gr.update()
    if axis_x_switch == "Nothing" and axis_y_switch != "Nothing":
        return "For using Y axis, first use X axis", gr.update()
    if (
        xy_plot_axis_options[axis_x_switch][0] == "dropdown"
        and len(axis_x_value_dropdown) < 1
    ):
        return "No values for axis X", gr.update()
    if (
        xy_plot_axis_options[axis_y_switch][0] == "dropdown"
        and len(axis_y_value_dropdown) < 1
    ):
        return "No values for axis Y", gr.update()
    if (
        xy_plot_axis_options[axis_z_switch][0] == "dropdown"
        and len(axis_z_value_dropdown) < 1
    ):
        return "No values for axis Z", gr.update()
    if not xy_plot_axis_options[axis_x_switch][3]:
        if axis_x_switch == axis_y_switch:
            return (
                'Axis type on X and Y axis are same, you can\'t do that generation.<br>Multi axis supported only for "Prompt add" and "Prompt replace".',
                gr.update(),
            )
        if axis_x_switch == axis_z_switch:
            return (
                'Axis type on X and Z axis are same, you can\'t do that generation.<br>Multi axis supported only for "Prompt add" and "Prompt replace".',
                gr.update(),
            )
    if not xy_plot_axis_options[axis_y_switch][3]:
        if axis_y_switch == axis_z_switch:
            return (
                'Axis type on Y and Z axis are same, you can\'t do that generation.<br>Multi axis supported only for "Prompt add" and "Prompt replace".',
                gr.update(),
            )

    base_generator_vars = {
        "model_type": model_type,
        "input_image": input_image,
        "end_frame_image": None,
        "end_frame_strength": 1.0,
        "input_video": None,
        "end_frame_image_original": end_frame_image_original,
        "end_frame_strength_original": end_frame_strength_original,
        "prompt_text": prompt,
        "n_prompt": "",
        "seed": seed,
        "total_second_length": total_second_length,
        "latent_window_size": latent_window_size,
        "steps": steps,
        "cfg": cfg,
        "gs": gs,
        "rs": rs,
        "use_teacache": use_teacache,
        "teacache_num_steps": teacache_num_steps,
        "teacache_rel_l1_thresh": teacache_rel_l1_thresh,
        "use_magcache": use_magcache,
        "magcache_threshold": magcache_threshold,
        "magcache_max_consecutive_skips": magcache_max_consecutive_skips,
        "magcache_retention_ratio": magcache_retention_ratio,
        "has_input_image": True if input_image is not None else False,
        "save_metadata_checked": True,
        "blend_sections": blend_sections,
        "latent_type": latent_type,
        "selected_loras": selected_loras,
        "resolutionW": resolutionW,
        "resolutionH": resolutionH,
        "lora_loaded_names": lora_names,
        "lora_values": lora_slider_values,
    }

    def xy_plot_convert_values(type, value_textbox, value_dropdown):
        retVal = []
        if type[0] == "dropdown":
            retVal = value_dropdown
        elif type[0] == "textbox":
            retVal = xy_plot_parse_input(value_textbox)
        elif type[0] == "number":
            if type[1] == "int":
                retVal = [int(float(x)) for x in xy_plot_parse_input(value_textbox)]
            else:
                retVal = [float(x) for x in xy_plot_parse_input(value_textbox)]
        return retVal

    prompt_replace_initial_values = {}
    all_axis_values = {
        axis_x_switch + " -> X": xy_plot_convert_values(
            xy_plot_axis_options[axis_x_switch],
            axis_x_value_text,
            axis_x_value_dropdown,
        )
    }
    if axis_x_switch == "Prompt replace":
        prompt_replace_initial_values["X"] = all_axis_values[axis_x_switch + " -> X"][0]
        if prompt_replace_initial_values["X"] not in base_generator_vars["prompt_text"]:
            return (
                "Prompt for replacing in X axis not present in generation prompt",
                gr.update(),
            )
    if axis_y_switch != "Nothing":
        all_axis_values[axis_y_switch + " -> Y"] = xy_plot_convert_values(
            xy_plot_axis_options[axis_y_switch],
            axis_y_value_text,
            axis_y_value_dropdown,
        )
        if axis_y_switch == "Prompt replace":
            prompt_replace_initial_values["Y"] = all_axis_values[
                axis_y_switch + " -> Y"
            ][0]
            if (
                prompt_replace_initial_values["Y"]
                not in base_generator_vars["prompt_text"]
            ):
                return (
                    "Prompt for replacing in Y axis not present in generation prompt",
                    gr.update(),
                )
    if axis_z_switch != "Nothing":
        all_axis_values[axis_z_switch + " -> Z"] = xy_plot_convert_values(
            xy_plot_axis_options[axis_z_switch],
            axis_z_value_text,
            axis_z_value_dropdown,
        )
        if axis_z_switch == "Prompt replace":
            prompt_replace_initial_values["Z"] = all_axis_values[
                axis_z_switch + " -> Z"
            ][0]
            if (
                prompt_replace_initial_values["Z"]
                not in base_generator_vars["prompt_text"]
            ):
                return (
                    "Prompt for replacing in Z axis not present in generation prompt",
                    gr.update(),
                )

    active_axes = list(all_axis_values.keys())
    value_lists = [all_axis_values[axis] for axis in active_axes]
    output_generator_vars = []

    combintion_plot = itertools.product(*value_lists)
    for combo in combintion_plot:
        vars_copy = base_generator_vars.copy()
        for axis, value in zip(active_axes, combo):
            splitted_axis_name = axis.split(" -> ")
            if splitted_axis_name[0] == "Prompt add":
                vars_copy["prompt_text"] = vars_copy["prompt_text"] + " " + str(value)
            elif splitted_axis_name[0] == "Prompt replace":
                orig_copy_prompt_text = vars_copy["prompt_text"]
                vars_copy["prompt_text"] = orig_copy_prompt_text.replace(
                    prompt_replace_initial_values[splitted_axis_name[1]], str(value)
                )
            else:
                vars_copy[text_to_base_keys[splitted_axis_name[0]]] = value
            vars_copy[splitted_axis_name[1] + "_axis_on_plot"] = str(value)

        worker_params = {
            k: v
            for k, v in vars_copy.items()
            if k not in ["X_axis_on_plot", "Y_axis_on_plot", "Z_axis_on_plot"]
        }
        output_generator_vars.append(worker_params)
    # print("----- BEFORE GENERATED VIDS VARS START -----")
    # for v in output_generator_vars:
    #     print(v)
    # print("------ BEFORE GENERATED VIDS VARS END ------")

    job_queue.add_job(
        params=base_generator_vars,
        job_type=JobType.GRID,
        child_job_params_list=output_generator_vars,
    )
    return "Grid job added to the queue.", gr.update(visible=False)
    # print("----- GENERATED VIDS VARS START -----")
    # for v in output_generator_vars:
    #     print(v)
    # print("------ GENERATED VIDS VARS END ------")

    # -------------------------- connect with settings --------------------------
    # Ensure settings is available in this scope or passed in.
    # Assuming 'settings' object is available from create_interface's scope.
    output_dir_setting = settings.get("output_dir", "outputs")
    mp4_crf_setting = settings.get("mp4_crf", 16)  # Default CRF if not in settings
    # -------------------------- connect with settings --------------------------


def create_xy_plot_ui(lora_names, default_prompt, DUMMY_LORA_NAME):
    """
    Creates the Gradio UI for the XY Plot functionality.
    Returns a dictionary of key components to be used by the main interface.
    """
    with gr.Group(visible=False) as xy_group:  # The original was visible=False
        with gr.Row():
            xy_plot_model_type = gr.Radio(
                ["Original", "F1"],
                label="Model Type",
                value="F1",
                info="Select which model to use for generation",
            )
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1):
                    xy_plot_input_image = gr.Image(
                        sources="upload",
                        type="numpy",
                        label="Image (optional)",
                        height=420,
                        image_mode="RGB",
                        elem_classes="contain-image",
                    )
                with gr.Column(scale=1):
                    xy_plot_end_frame_image_original = gr.Image(
                        sources="upload",
                        type="numpy",
                        label="End Frame (Optional)",
                        height=420,
                        elem_classes="contain-image",
                        image_mode="RGB",
                        show_download_button=False,
                        show_label=True,
                        container=True,
                    )
            with gr.Group():
                xy_plot_end_frame_strength_original = gr.Slider(
                    label="End Frame Influence",
                    minimum=0.05,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    info="Controls how strongly the end frame guides the generation. 1.0 is full influence.",
                )
        with gr.Accordion("Latent Image Options", open=False):
            xy_plot_latent_type = gr.Dropdown(
                ["Black", "White", "Noise", "Green Screen"],
                label="Latent Image",
                value="Black",
                info="Used as a starting point if no image is provided",
            )
        xy_plot_prompt = gr.Textbox(label="Prompt", value=default_prompt)
        with gr.Accordion("Prompt Parameters", open=False):
            xy_plot_blend_sections = gr.Slider(
                minimum=0,
                maximum=10,
                value=4,
                step=1,
                label="Number of sections to blend between prompts",
            )
        with gr.Accordion("Generation Parameters", open=True):
            with gr.Row():
                xy_plot_steps = gr.Slider(
                    label="Steps", minimum=1, maximum=100, value=5, step=1
                )
                xy_plot_total_second_length = gr.Slider(
                    label="Video Length (Seconds)",
                    minimum=0.1,
                    maximum=120,
                    value=1,
                    step=0.1,
                )
            with gr.Row():
                xy_plot_seed = gr.Number(label="Seed", value=31337, precision=0)
                xy_plot_randomize_seed = gr.Checkbox(
                    label="Randomize",
                    value=False,
                    info="Generate a new random seed for each job",
                )
            with gr.Row("LoRAs"):
                xy_plot_lora_selector = gr.Dropdown(
                    choices=lora_names,
                    label="Select LoRAs to Load",
                    multiselect=True,
                    value=[],
                    info="Select one or more LoRAs to use for this job",
                )
                xy_plot_lora_sliders = {}
                for lora in lora_names:
                    xy_plot_lora_sliders[lora] = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.01,
                        label=f"{lora} Weight",
                        visible=False,
                        interactive=True,
                    )
        with gr.Accordion("Advanced Parameters", open=False):
            with gr.Row("TeaCache"):
                xy_plot_use_teacache = gr.Checkbox(
                    label="Use TeaCache",
                    value=True,
                    info="Faster speed, but often makes hands and fingers slightly worse.",
                )
                xy_plot_teacache_num_steps = gr.Slider(
                    label="TeaCache steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=25,
                    visible=True,
                    info="How many intermediate sections to keep in the cache",
                )
                xy_plot_teacache_rel_l1_thresh = gr.Slider(
                    label="TeaCache rel_l1_thresh",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=0.15,
                    visible=True,
                    info="Relative L1 Threshold",
                )
            with gr.Row("MagCache"):
                xy_plot_use_magcache = gr.Checkbox(
                    label="Use MagCache",
                    value=False,
                    info="Faster speed, but may introduce artifacts. Uses pre-calibrated ratios.",
                )
                xy_plot_magcache_threshold = gr.Slider(
                    label="MagCache Threshold",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=0.1,
                    visible=False,
                    info="Error tolerance for skipping steps. Lower = more skips, higher = fewer skips.",
                )
                xy_plot_magcache_max_consecutive_skips = gr.Slider(
                    label="MagCache Max Consecutive Skips",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=2,
                    visible=False,
                    info="Maximum number of consecutive steps that can be skipped.",
                )
                xy_plot_magcache_retention_ratio = gr.Slider(
                    label="MagCache Retention Ratio",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.25,
                    visible=False,
                    info="Ratio of initial steps to always calculate (not skip).",
                )

            # Mutual exclusivity logic for TeaCache and MagCache in XY Plot UI
            xy_plot_use_teacache.change(
                lambda enabled: (
                    gr.update(visible=enabled),
                    gr.update(visible=enabled),
                    gr.update(value=not enabled),
                ),
                inputs=xy_plot_use_teacache,
                outputs=[
                    xy_plot_teacache_num_steps,
                    xy_plot_teacache_rel_l1_thresh,
                    xy_plot_use_magcache,
                ],
            )
            xy_plot_use_magcache.change(
                lambda enabled: (
                    gr.update(visible=enabled),
                    gr.update(visible=enabled),
                    gr.update(visible=enabled),
                    gr.update(value=not enabled),
                ),
                inputs=xy_plot_use_magcache,
                outputs=[
                    xy_plot_magcache_threshold,
                    xy_plot_magcache_max_consecutive_skips,
                    xy_plot_magcache_retention_ratio,
                    xy_plot_use_teacache,
                ],
            )

            xy_plot_latent_window_size = gr.Slider(
                label="Latent Window Size",
                minimum=1,
                maximum=33,
                value=9,
                step=1,
                visible=True,
                info="Change at your own risk, very experimental",
            )
            xy_plot_cfg = gr.Slider(
                label="CFG Scale",
                minimum=1.0,
                maximum=32.0,
                value=1.0,
                step=0.01,
                visible=False,
            )
            xy_plot_gs = gr.Slider(
                label="Distilled CFG Scale",
                minimum=1.0,
                maximum=32.0,
                value=10.0,
                step=0.01,
            )
            xy_plot_rs = gr.Slider(
                label="CFG Re-Scale",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                visible=False,
            )
            xy_plot_gpu_memory_preservation = gr.Slider(
                label="GPU Inference Preserved Memory (GB) (larger means slower)",
                minimum=1,
                maximum=128,
                value=6,
                step=0.1,
                info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.",
            )
        with gr.Accordion("Output Parameters", open=False):
            xy_plot_mp4_crf = gr.Slider(
                label="MP4 Compression",
                minimum=0,
                maximum=100,
                value=16,
                step=1,
                info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ",
            )
        with gr.Accordion("Plot Parameters", open=True):

            def xy_plot_axis_change(updated_value_type):
                if (
                    xy_plot_axis_options[updated_value_type][0] == "textbox"
                    or xy_plot_axis_options[updated_value_type][0] == "number"
                ):
                    return gr.update(
                        visible=True, value=xy_plot_axis_options[updated_value_type][2]
                    ), gr.update(visible=False, value=[], choices=[])
                elif xy_plot_axis_options[updated_value_type][0] == "dropdown":
                    return gr.update(visible=False), gr.update(
                        visible=True,
                        value=xy_plot_axis_options[updated_value_type][2],
                        choices=xy_plot_axis_options[updated_value_type][1],
                    )
                else:
                    return gr.update(visible=False), gr.update(
                        visible=False, value=[], choices=[]
                    )

            with gr.Row():
                xy_plot_axis_x_switch = gr.Dropdown(
                    label="X axis type for plotting",
                    choices=list(xy_plot_axis_options.keys()),
                )
                xy_plot_axis_x_value_text = gr.Textbox(
                    label="X axis comma separated text", visible=False
                )
                xy_plot_axis_x_value_dropdown = gr.CheckboxGroup(
                    label="X axis values", visible=False
                )  # , multiselect=True)
            with gr.Row():
                xy_plot_axis_y_switch = gr.Dropdown(
                    label="Y axis type for plotting",
                    choices=list(xy_plot_axis_options.keys()),
                )
                xy_plot_axis_y_value_text = gr.Textbox(
                    label="Y axis comma separated text", visible=False
                )
                xy_plot_axis_y_value_dropdown = gr.CheckboxGroup(
                    label="Y axis values", visible=False
                )  # , multiselect=True)
            with gr.Row(visible=False):  # not implemented Z axis
                xy_plot_axis_z_switch = gr.Dropdown(
                    label="Z axis type for plotting",
                    choices=list(xy_plot_axis_options.keys()),
                )
                xy_plot_axis_z_value_text = gr.Textbox(
                    label="Z axis comma separated text", visible=False
                )
                xy_plot_axis_z_value_dropdown = gr.CheckboxGroup(
                    label="Z axis values", visible=False
                )  # , multiselect=True)

        xy_plot_status = gr.HTML("")
        xy_plot_output = gr.Video(
            autoplay=True, loop=True, sources=[], height=256, visible=False
        )
        # --- ADD THE PROCESS BUTTON HERE ---
        # This button is logically part of the XY plot group but will be controlled
        # from interface.py. We place it here so it's encapsulated.
        xy_plot_process_btn = gr.Button("Submit", visible=False)

    # --- Internal Event Handlers ---
    xy_plot_use_teacache.change(
        lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)),
        inputs=xy_plot_use_teacache,
        outputs=[xy_plot_teacache_num_steps, xy_plot_teacache_rel_l1_thresh],
    )
    xy_plot_axis_x_switch.change(
        fn=xy_plot_axis_change,
        inputs=[xy_plot_axis_x_switch],
        outputs=[xy_plot_axis_x_value_text, xy_plot_axis_x_value_dropdown],
    )
    xy_plot_axis_y_switch.change(
        fn=xy_plot_axis_change,
        inputs=[xy_plot_axis_y_switch],
        outputs=[xy_plot_axis_y_value_text, xy_plot_axis_y_value_dropdown],
    )
    xy_plot_axis_z_switch.change(
        fn=xy_plot_axis_change,
        inputs=[xy_plot_axis_z_switch],
        outputs=[xy_plot_axis_z_value_text, xy_plot_axis_z_value_dropdown],
    )

    def xy_plot_update_lora_sliders(selected_loras):
        updates = []
        actual_selected_loras_for_display = [
            lora for lora in selected_loras if lora != DUMMY_LORA_NAME
        ]
        updates.append(gr.update(value=actual_selected_loras_for_display))

        for lora_name_key in lora_names:
            if lora_name_key == DUMMY_LORA_NAME:
                updates.append(gr.update(visible=False))
            else:
                updates.append(
                    gr.update(
                        visible=(lora_name_key in actual_selected_loras_for_display)
                    )
                )
        return updates

    xy_plot_lora_selector.change(
        fn=xy_plot_update_lora_sliders,
        inputs=[xy_plot_lora_selector],
        outputs=[xy_plot_lora_selector]
        + [
            xy_plot_lora_sliders[lora]
            for lora in lora_names
            if lora in xy_plot_lora_sliders
        ],
    )

    # --- Component Dictionary for Export ---
    components = {
        "group": xy_group,
        "status": xy_plot_status,
        "output": xy_plot_output,
        "process_btn": xy_plot_process_btn,
        # --- Inputs for the process button ---
        "model_type": xy_plot_model_type,
        "input_image": xy_plot_input_image,
        "end_frame_image_original": xy_plot_end_frame_image_original,
        "end_frame_strength_original": xy_plot_end_frame_strength_original,
        "latent_type": xy_plot_latent_type,
        "prompt": xy_plot_prompt,
        "blend_sections": xy_plot_blend_sections,
        "steps": xy_plot_steps,
        "total_second_length": xy_plot_total_second_length,
        "seed": xy_plot_seed,
        "randomize_seed": xy_plot_randomize_seed,
        "use_teacache": xy_plot_use_teacache,
        "teacache_num_steps": xy_plot_teacache_num_steps,
        "teacache_rel_l1_thresh": xy_plot_teacache_rel_l1_thresh,
        "use_magcache": xy_plot_use_magcache,
        "magcache_threshold": xy_plot_magcache_threshold,
        "magcache_max_consecutive_skips": xy_plot_magcache_max_consecutive_skips,
        "magcache_retention_ratio": xy_plot_magcache_retention_ratio,
        "latent_window_size": xy_plot_latent_window_size,
        "cfg": xy_plot_cfg,
        "gs": xy_plot_gs,
        "rs": xy_plot_rs,
        "gpu_memory_preservation": xy_plot_gpu_memory_preservation,
        "mp4_crf": xy_plot_mp4_crf,
        "axis_x_switch": xy_plot_axis_x_switch,
        "axis_x_value_text": xy_plot_axis_x_value_text,
        "axis_x_value_dropdown": xy_plot_axis_x_value_dropdown,
        "axis_y_switch": xy_plot_axis_y_switch,
        "axis_y_value_text": xy_plot_axis_y_value_text,
        "axis_y_value_dropdown": xy_plot_axis_y_value_dropdown,
        "axis_z_switch": xy_plot_axis_z_switch,
        "axis_z_value_text": xy_plot_axis_z_value_text,
        "axis_z_value_dropdown": xy_plot_axis_z_value_dropdown,
        "lora_selector": xy_plot_lora_selector,
        "lora_sliders": xy_plot_lora_sliders,
    }
    return components
