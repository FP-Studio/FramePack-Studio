import gradio as gr
import os
import json
import random
import functools
import logging
from PIL import Image
import numpy as np

from diffusers_helper.bucket_tools import find_nearest_bucket
from modules.xy_plot_ui import create_xy_plot_ui
from modules.llm_enhancer import enhance_prompt
from modules.llm_captioner import caption_image

PRESET_FILE = os.path.join(".framepack", "generation_presets.json")


def load_presets(model_type):
    if not os.path.exists(PRESET_FILE) or not os.path.isfile(PRESET_FILE):
        return []
    try:
        with open(PRESET_FILE, "r") as f:
            data = json.load(f)
        return list(data.get(model_type, {}).keys())
    except (json.JSONDecodeError, IOError):
        return []


def create_generate_ui(
    lora_names, default_prompt, DUMMY_LORA_NAME, get_latents_display_top, settings
):
    with gr.Row(visible=get_latents_display_top()) as top_preview_row:
        top_preview_image = gr.Image(
            label="Next Latents (Top Display)",
            height=150,
            visible=True,
            type="numpy",
            interactive=False,
            elem_classes="contain-image",
            image_mode="RGB",
        )

    with gr.Row():
        with gr.Column(scale=2):
            model_type = gr.Radio(
                choices=[
                    ("Original", "Original"),
                    ("Original with Endframe", "Original with Endframe"),
                    ("F1", "F1"),
                    ("Video", "Video"),
                    ("Video with Endframe", "Video with Endframe"),
                    ("Video F1", "Video F1"),
                ],
                value="Original",
                label="Generation Type",
            )
            with gr.Accordion(
                "Original Presets", open=False, visible=True
            ) as preset_accordion:
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="Select Preset",
                        choices=load_presets("Original"),
                        interactive=True,
                        scale=2,
                    )
                    delete_preset_button = gr.Button(
                        "🗑️ Delete", variant="stop", scale=1
                    )
                with gr.Row():
                    preset_name_textbox = gr.Textbox(
                        label="Preset Name",
                        placeholder="Enter a name for your preset",
                        scale=2,
                    )
                    save_preset_button = gr.Button(
                        "💾 Save", variant="primary", scale=1
                    )
                with gr.Row(visible=False) as confirm_delete_row:
                    gr.Markdown("### Are you sure you want to delete this preset?")
                    confirm_delete_yes_btn = gr.Button("🗑️ Yes, Delete", variant="stop")
                    confirm_delete_no_btn = gr.Button("↩️ No, Go Back")
            with gr.Accordion(
                "Basic Parameters", open=True, visible=True
            ) as basic_parameters_accordion:
                with gr.Group():
                    total_second_length = gr.Slider(
                        label="Video Length (Seconds)",
                        minimum=1,
                        maximum=120,
                        value=6,
                        step=0.1,
                    )
                    with gr.Row("Resolution"):
                        resolutionW = gr.Slider(
                            label="Width",
                            minimum=128,
                            maximum=768,
                            value=640,
                            step=32,
                            info="Nearest valid width will be used.",
                        )
                        resolutionH = gr.Slider(
                            label="Height",
                            minimum=128,
                            maximum=768,
                            value=640,
                            step=32,
                            info="Nearest valid height will be used.",
                        )
                    resolution_text = gr.Markdown(
                        value="<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: 640 x 640</div>",
                        label="",
                        show_label=False,
                    )

            xy_plot_components = create_xy_plot_ui(
                lora_names=lora_names,
                default_prompt=default_prompt,
                DUMMY_LORA_NAME=DUMMY_LORA_NAME,
            )

            with gr.Group(visible=True) as standard_generation_group:
                with gr.Group(visible=True) as image_input_group:
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_image = gr.Image(
                                sources="upload",
                                type="numpy",
                                label="Start Frame (optional)",
                                elem_classes="contain-image",
                                image_mode="RGB",
                                show_download_button=False,
                                show_label=True,
                                container=True,
                            )

                with gr.Group(visible=False) as video_input_group:
                    input_video = gr.Video(
                        sources="upload",
                        label="Video Input",
                        height=420,
                        show_label=True,
                    )
                    combine_with_source = gr.Checkbox(
                        label="Combine with source video",
                        value=True,
                        info="If checked, the source video will be combined with the generated video",
                        interactive=True,
                    )
                    num_cleaned_frames = gr.Slider(
                        label="Number of Context Frames (Adherence to Video)",
                        minimum=2,
                        maximum=10,
                        value=5,
                        step=1,
                        interactive=True,
                        info="Expensive. Retain more video details. Reduce if memory issues or motion too restricted (jumpcut, ignoring prompt, still).",
                    )

                with gr.Column(scale=1, visible=False) as end_frame_group_original:
                    end_frame_image_original = gr.Image(
                        sources="upload",
                        type="numpy",
                        label="End Frame (Optional)",
                        elem_classes="contain-image",
                        image_mode="RGB",
                        show_download_button=False,
                        show_label=True,
                        container=True,
                    )

                with gr.Group(visible=False) as end_frame_slider_group:
                    end_frame_strength_original = gr.Slider(
                        label="End Frame Influence",
                        minimum=0.05,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        info="Controls how strongly the end frame guides the generation. 1.0 is full influence.",
                    )

                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", value=default_prompt, scale=10)
                with gr.Row():
                    enhance_prompt_btn = gr.Button("✨ Enhance", scale=1)
                    caption_btn = gr.Button("✨ Caption", scale=1)

                with gr.Accordion("Prompt Parameters", open=False):
                    n_prompt = gr.Textbox(
                        label="Negative Prompt", value="", visible=True
                    )
                    blend_sections = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=4,
                        step=1,
                        label="Number of sections to blend between prompts",
                    )
                with gr.Accordion("Batch Input", open=False):
                    batch_input_images = gr.File(
                        label="Batch Images (Upload one or more)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    batch_input_gallery = gr.Gallery(
                        label="Selected Batch Images",
                        visible=False,
                        columns=5,
                        object_fit="contain",
                        height="auto",
                    )
                    add_batch_to_queue_btn = gr.Button(
                        "🚀 Add Batch to Queue", variant="primary"
                    )
                with gr.Accordion("Generation Parameters", open=True):
                    steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=25, step=1
                    )
                    seed = gr.Number(label="Seed", value=2500, precision=0)
                    randomize_seed = gr.Checkbox(
                        label="Randomize",
                        value=True,
                        info="Generate a new random seed for each job",
                    )
                with gr.Accordion("LoRAs", open=False):
                    with gr.Row():
                        lora_selector = gr.Dropdown(
                            choices=lora_names,
                            label="Select LoRAs to Load",
                            multiselect=True,
                            value=[],
                            info="Select one or more LoRAs to use for this job",
                            scale=10,
                        )
                    lora_names_states = gr.State(lora_names)
                    lora_sliders = {
                        lora: gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.01,
                            label=f"{lora} Weight",
                            visible=False,
                            interactive=True,
                        )
                        for lora in lora_names
                    }
                    with gr.Row():
                        refresh_loras_button = gr.Button(
                            "🔄 Refresh", variant="secondary"
                        )
                with gr.Accordion("Latent Image Options", open=False):
                    latent_type = gr.Dropdown(
                        ["Noise", "White", "Black", "Green Screen"],
                        label="Latent Image",
                        value="Noise",
                        info="Used as a starting point if no image is provided",
                    )
                with gr.Accordion("Advanced Parameters", open=False):
                    gr.Markdown("#### Motion Model")
                    gr.Markdown("Settings for precise control of the motion model")
                    with gr.Group(elem_classes="control-group"):
                        latent_window_size = gr.Slider(
                            label="Latent Window Size",
                            minimum=1,
                            maximum=33,
                            value=9,
                            step=1,
                            info="Change at your own risk, very experimental",
                        )
                        gs = gr.Slider(
                            label="Distilled CFG Scale",
                            minimum=1.0,
                            maximum=32.0,
                            value=10.0,
                            step=0.5,
                        )
                    gr.Markdown("#### CFG Scale")
                    gr.Markdown(
                        "Much better prompt following. Warning: Modifying these values from their defaults will almost double generation time. ⚠️"
                    )
                    with gr.Group(elem_classes="control-group"):
                        cfg = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                        )
                        rs = gr.Slider(
                            label="CFG Re-Scale",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                    gr.Markdown("#### Cache Options")
                    gr.Markdown(
                        "Using a cache will speed up generation. May affect quality, fine or even coarse details, and may change or inhibit motion. You can choose at most one."
                    )
                    with gr.Group(elem_classes="control-group"):
                        cache_type = gr.Radio(
                            ["MagCache", "TeaCache", "None"],
                            value="MagCache",
                            label="Caching strategy",
                            info="Which cache implementation to use, if any",
                        )
                        with gr.Row():
                            magcache_threshold = gr.Slider(
                                label="MagCache Threshold",
                                minimum=0.01,
                                maximum=1.0,
                                step=0.01,
                                value=0.1,
                                visible=True,
                                info="[⬇️ **Faster**] Error tolerance. Lower = more estimated steps",
                            )
                            magcache_max_consecutive_skips = gr.Slider(
                                label="MagCache Max Consecutive Skips",
                                minimum=1,
                                maximum=5,
                                step=1,
                                value=2,
                                visible=True,
                                info="[⬆️ **Faster**] Allow multiple estimated steps in a row",
                            )
                            magcache_retention_ratio = gr.Slider(
                                label="MagCache Retention Ratio",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.25,
                                visible=True,
                                info="[⬇️ **Faster**] Disallow estimation in critical early steps",
                            )
                        with gr.Row():
                            teacache_num_steps = gr.Slider(
                                label="TeaCache steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=25,
                                visible=False,
                                info="How many intermediate sections to keep in the cache",
                            )
                            teacache_rel_l1_thresh = gr.Slider(
                                label="TeaCache rel_l1_thresh",
                                minimum=0.01,
                                maximum=1.0,
                                step=0.01,
                                value=0.15,
                                visible=False,
                                info="[⬇️ **Faster**] Relative L1 Threshold",
                            )
                with gr.Row("Metadata"):
                    json_upload = gr.File(
                        label="Upload Metadata JSON (optional)",
                        file_types=[".json"],
                        type="filepath",
                        height=140,
                    )

        with gr.Column():
            preview_image = gr.Image(
                label="Next Latents",
                height=150,
                visible=not get_latents_display_top(),
                type="numpy",
                interactive=False,
                elem_classes="contain-image",
                image_mode="RGB",
            )
            result_video = gr.Video(
                label="Finished Frames",
                autoplay=True,
                show_share_button=False,
                height=256,
                loop=True,
            )
            progress_desc = gr.Markdown("", elem_classes="no-generating-animation")
            progress_bar = gr.HTML("", elem_classes="no-generating-animation")
            with gr.Row():
                current_job_id = gr.Textbox(
                    label="Current Job ID", value="", visible=True, interactive=True
                )
                start_button = gr.Button(
                    value="🚀 Add to Queue",
                    variant="primary",
                    elem_id="toolbar-add-to-queue-btn",
                )
                end_button = gr.Button(
                    value="❌ Cancel Current Job", interactive=True, visible=False
                )
                video_input_required_message = gr.Markdown(
                    "<p style='color: red; text-align: center;'>Input video required</p>",
                    visible=False,
                )

    ui_components = {
        "prompt": prompt,
        "n_prompt": n_prompt,
        "blend_sections": blend_sections,
        "steps": steps,
        "total_second_length": total_second_length,
        "resolutionW": resolutionW,
        "resolutionH": resolutionH,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "gs": gs,
        "cfg": cfg,
        "rs": rs,
        "latent_window_size": latent_window_size,
        "cache_type": cache_type,
        "teacache_num_steps": teacache_num_steps,
        "teacache_rel_l1_thresh": teacache_rel_l1_thresh,
        "magcache_threshold": magcache_threshold,
        "magcache_max_consecutive_skips": magcache_max_consecutive_skips,
        "magcache_retention_ratio": magcache_retention_ratio,
        "latent_type": latent_type,
        "end_frame_strength_original": end_frame_strength_original,
        "combine_with_source": combine_with_source,
        "num_cleaned_frames": num_cleaned_frames,
        "lora_selector": lora_selector,
        **lora_sliders,
    }

    return {
        "top_preview_row": top_preview_row,
        "top_preview_image": top_preview_image,
        "model_type": model_type,
        "preset_accordion": preset_accordion,
        "preset_dropdown": preset_dropdown,
        "delete_preset_button": delete_preset_button,
        "preset_name_textbox": preset_name_textbox,
        "save_preset_button": save_preset_button,
        "confirm_delete_row": confirm_delete_row,
        "confirm_delete_yes_btn": confirm_delete_yes_btn,
        "confirm_delete_no_btn": confirm_delete_no_btn,
        "basic_parameters_accordion": basic_parameters_accordion,
        "total_second_length": total_second_length,
        "resolutionW": resolutionW,
        "resolutionH": resolutionH,
        "resolution_text": resolution_text,
        "xy_plot_components": xy_plot_components,
        "standard_generation_group": standard_generation_group,
        "image_input_group": image_input_group,
        "input_image": input_image,
        "video_input_group": video_input_group,
        "input_video": input_video,
        "combine_with_source": combine_with_source,
        "num_cleaned_frames": num_cleaned_frames,
        "end_frame_group_original": end_frame_group_original,
        "end_frame_image_original": end_frame_image_original,
        "end_frame_slider_group": end_frame_slider_group,
        "end_frame_strength_original": end_frame_strength_original,
        "prompt": prompt,
        "enhance_prompt_btn": enhance_prompt_btn,
        "caption_btn": caption_btn,
        "n_prompt": n_prompt,
        "blend_sections": blend_sections,
        "batch_input_images": batch_input_images,
        "batch_input_gallery": batch_input_gallery,
        "add_batch_to_queue_btn": add_batch_to_queue_btn,
        "steps": steps,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "lora_selector": lora_selector,
        "refresh_loras_button": refresh_loras_button,
        "lora_names_states": lora_names_states,
        "lora_sliders": lora_sliders,
        "latent_type": latent_type,
        "latent_window_size": latent_window_size,
        "gs": gs,
        "cfg": cfg,
        "rs": rs,
        "cache_type": cache_type,
        "magcache_threshold": magcache_threshold,
        "magcache_max_consecutive_skips": magcache_max_consecutive_skips,
        "magcache_retention_ratio": magcache_retention_ratio,
        "teacache_num_steps": teacache_num_steps,
        "teacache_rel_l1_thresh": teacache_rel_l1_thresh,
        "json_upload": json_upload,
        "preview_image": preview_image,
        "result_video": result_video,
        "progress_desc": progress_desc,
        "progress_bar": progress_bar,
        "current_job_id": current_job_id,
        "start_button": start_button,
        "end_button": end_button,
        "video_input_required_message": video_input_required_message,
        "ui_components": ui_components,
    }


def connect_generate_events(g, s, q, f):
    # g: generate_ui_components, s: settings_components, q: queue_components, f: functions

    def on_input_image_change(img):
        return (
            gr.update(
                info="Nearest valid bucket size will be used. Height will be adjusted automatically."
            ),
            gr.update(visible=img is None),
        )

    g["input_image"].change(
        fn=on_input_image_change,
        inputs=[g["input_image"]],
        outputs=[g["resolutionW"], g["resolutionH"]],
    )

    def on_resolution_change(img, resW, resH):
        H, W = (resH, resW) if img is None else (img.shape[0], img.shape[1])
        aspect_ratio_res = (resW + resH) / 2 if img is None else resW
        out_bucket_resH, out_bucket_resW = find_nearest_bucket(
            H, W, resolution=aspect_ratio_res
        )
        return gr.update(
            value=f"<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: {out_bucket_resW} x {out_bucket_resH}</div>"
        )

    g["resolutionW"].change(
        fn=on_resolution_change,
        inputs=[g["input_image"], g["resolutionW"], g["resolutionH"]],
        outputs=[g["resolution_text"]],
        show_progress="hidden",
    )
    g["resolutionH"].change(
        fn=on_resolution_change,
        inputs=[g["input_image"], g["resolutionW"], g["resolutionH"]],
        outputs=[g["resolution_text"]],
        show_progress="hidden",
    )

    def update_cache_type(cache_type_val: str):
        is_mag = cache_type_val == "MagCache"
        is_tea = cache_type_val == "TeaCache"
        return [gr.update(visible=is_mag)] * 3 + [gr.update(visible=is_tea)] * 2

    g["cache_type"].change(
        fn=update_cache_type,
        inputs=g["cache_type"],
        outputs=[
            g["magcache_threshold"],
            g["magcache_max_consecutive_skips"],
            g["magcache_retention_ratio"],
            g["teacache_num_steps"],
            g["teacache_rel_l1_thresh"],
        ],
    )

    def process_with_queue_update(model_type_arg, *args):
        queue_status_data, queue_stats_text = f["update_stats"]()
        (
            input_image_arg,
            input_video_arg,
            end_frame_image_original_arg,
            end_frame_strength_original_arg,
            prompt_text_arg,
            n_prompt_arg,
            seed_arg,
            randomize_seed_arg,
            total_second_length_arg,
            latent_window_size_arg,
            steps_arg,
            cfg_arg,
            gs_arg,
            rs_arg,
            cache_type_arg,
            teacache_num_steps_arg,
            teacache_rel_l1_thresh_arg,
            magcache_threshold_arg,
            magcache_max_consecutive_skips_arg,
            magcache_retention_ratio_arg,
            blend_sections_arg,
            latent_type_arg,
            clean_up_videos_arg,
            selected_loras_arg,
            resolutionW_arg,
            resolutionH_arg,
            combine_with_source_arg,
            num_cleaned_frames_arg,
            lora_names_states_arg,
            *lora_slider_values_tuple,
        ) = args
        backend_model_type = (
            "Video" if model_type_arg == "Video with Endframe" else model_type_arg
        )
        is_ui_video_model = f["is_video_model"](model_type_arg)
        input_data = input_video_arg if is_ui_video_model else input_image_arg
        actual_end_frame_image_for_backend, actual_end_frame_strength_for_backend = (
            (end_frame_image_original_arg, end_frame_strength_original_arg)
            if model_type_arg
            in ["Original with Endframe", "F1 with Endframe", "Video with Endframe"]
            else (None, 1.0)
        )
        input_image_path = (
            input_video_arg
            if is_ui_video_model and input_video_arg is not None
            else None
        )
        result = f["process_fn"](
            backend_model_type,
            input_data,
            actual_end_frame_image_for_backend,
            actual_end_frame_strength_for_backend,
            prompt_text_arg,
            n_prompt_arg,
            seed_arg,
            total_second_length_arg,
            latent_window_size_arg,
            steps_arg,
            cfg_arg,
            gs_arg,
            rs_arg,
            cache_type_arg == "TeaCache",
            teacache_num_steps_arg,
            teacache_rel_l1_thresh_arg,
            cache_type_arg == "MagCache",
            magcache_threshold_arg,
            magcache_max_consecutive_skips_arg,
            magcache_retention_ratio_arg,
            blend_sections_arg,
            latent_type_arg,
            clean_up_videos_arg,
            selected_loras_arg,
            resolutionW_arg,
            resolutionH_arg,
            input_image_path,
            combine_with_source_arg,
            num_cleaned_frames_arg,
            lora_names_states_arg,
            *lora_slider_values_tuple,
        )
        new_seed_value = random.randint(0, 21474) if randomize_seed_arg else None
        if new_seed_value:
            logging.info(f"Generated new seed for next job: {new_seed_value}")
        start_button_update_after_add = gr.update(value="🚀 Add to Queue")
        if result and result[1]:
            job_id = result[1]
            queue_status_data, queue_stats_text = f["update_stats"]()
            base_return = [
                result[0],
                job_id,
                result[2],
                result[3],
                result[4],
                start_button_update_after_add,
                result[6],
                queue_status_data,
                queue_stats_text,
            ]
            return (
                base_return + [new_seed_value, gr.update()]
                if new_seed_value is not None
                else base_return + [gr.update(), gr.update()]
            )
        queue_status_data, queue_stats_text = f["update_stats"]()
        base_return = [
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            start_button_update_after_add,
            result[6],
            queue_status_data,
            queue_stats_text,
        ]
        return (
            base_return + [new_seed_value, gr.update()]
            if new_seed_value is not None
            else base_return + [gr.update(), gr.update()]
        )

    ips = [
        g["input_image"],
        g["input_video"],
        g["end_frame_image_original"],
        g["end_frame_strength_original"],
        g["prompt"],
        g["n_prompt"],
        g["seed"],
        g["randomize_seed"],
        g["total_second_length"],
        g["latent_window_size"],
        g["steps"],
        g["cfg"],
        g["gs"],
        g["rs"],
        g["cache_type"],
        g["teacache_num_steps"],
        g["teacache_rel_l1_thresh"],
        g["magcache_threshold"],
        g["magcache_max_consecutive_skips"],
        g["magcache_retention_ratio"],
        g["blend_sections"],
        g["latent_type"],
        s["clean_up_videos"],
        g["lora_selector"],
        g["resolutionW"],
        g["resolutionH"],
        g["combine_with_source"],
        g["num_cleaned_frames"],
        g["lora_names_states"],
    ] + list(g["lora_sliders"].values())

    def handle_start_button(selected_model, *args):
        return process_with_queue_update(selected_model, *args)

    def update_button_before_processing(*args):
        qs_data, qs_text = f["update_stats"]()
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="⏳ Adding...", interactive=False),
            gr.update(),
            qs_data,
            qs_text,
            gr.update(),
            gr.update(),
        )

    g["start_button"].click(
        fn=update_button_before_processing,
        inputs=[g["model_type"]] + ips,
        outputs=[
            g["result_video"],
            g["current_job_id"],
            g["preview_image"],
            g["top_preview_image"],
            g["progress_desc"],
            g["progress_bar"],
            g["start_button"],
            g["end_button"],
            q["queue_status"],
            q["queue_stats_display"],
            g["seed"],
            g["video_input_required_message"],
        ],
    ).then(
        fn=handle_start_button,
        inputs=[g["model_type"]] + ips,
        outputs=[
            g["result_video"],
            g["current_job_id"],
            g["preview_image"],
            g["progress_desc"],
            g["progress_bar"],
            g["start_button"],
            g["end_button"],
            q["queue_status"],
            q["queue_stats_display"],
            g["seed"],
            g["video_input_required_message"],
        ],
    ).then(
        fn=f["update_start_button_state"],
        inputs=[g["model_type"], g["input_video"]],
        outputs=[g["start_button"], g["video_input_required_message"]],
    )

    def handle_batch_add_to_queue(*args):
        batch_files = args[-1]
        if not batch_files:
            return
        logging.info(f"Starting batch processing for {len(batch_files)} images.")
        single_job_args = list(args[:-1])
        model_type_arg = single_job_args.pop(0)
        current_seed, randomize_seed_arg = single_job_args[6], single_job_args[7]
        for image_path in batch_files:
            try:
                numpy_image = np.array(Image.open(image_path).convert("RGB"))
                current_job_args = single_job_args[:]
                current_job_args[0], current_job_args[6] = numpy_image, current_seed
                process_with_queue_update(model_type_arg, *current_job_args)
                if randomize_seed_arg:
                    current_seed = random.randint(0, 21474)
            except Exception as e:
                logging.error(f"Error loading batch image {image_path}: {e}. Skipping.")
        logging.info("Batch processing complete.")

    g["batch_input_images"].change(
        fn=lambda files: gr.update(value=files, visible=bool(files)),
        inputs=[g["batch_input_images"]],
        outputs=[g["batch_input_gallery"]],
    )
    batch_ips = [g["model_type"]] + ips + [g["batch_input_images"]]
    g["add_batch_to_queue_btn"].click(
        fn=handle_batch_add_to_queue, inputs=batch_ips, outputs=None
    ).then(
        fn=f["update_stats"],
        inputs=None,
        outputs=[q["queue_status"], q["queue_stats_display"]],
    ).then(
        fn=f["check_for_current_job"],
        inputs=None,
        outputs=[
            g["current_job_id"],
            g["result_video"],
            g["preview_image"],
            g["top_preview_image"],
            g["progress_desc"],
            g["progress_bar"],
        ],
    ).then(
        fn=f["create_latents_layout_update"],
        inputs=None,
        outputs=[g["top_preview_row"], g["preview_image"]],
    )

    c = g["xy_plot_components"]
    xy_plot_process_btn = c["process_btn"]
    fn_xy_process_with_deps = functools.partial(
        f["xy_plot_process"], f["job_queue"], f["settings"]
    )
    xy_plot_input_components = [
        c["model_type"],
        c["input_image"],
        c["end_frame_image_original"],
        c["end_frame_strength_original"],
        c["latent_type"],
        c["prompt"],
        c["blend_sections"],
        c["steps"],
        c["total_second_length"],
        g["resolutionW"],
        g["resolutionH"],
        c["seed"],
        c["randomize_seed"],
        c["use_teacache"],
        c["teacache_num_steps"],
        c["teacache_rel_l1_thresh"],
        c["use_magcache"],
        c["magcache_threshold"],
        c["magcache_max_consecutive_skips"],
        c["magcache_retention_ratio"],
        c["latent_window_size"],
        c["cfg"],
        c["gs"],
        c["rs"],
        c["gpu_memory_preservation"],
        c["mp4_crf"],
        c["axis_x_switch"],
        c["axis_x_value_text"],
        c["axis_x_value_dropdown"],
        c["axis_y_switch"],
        c["axis_y_value_text"],
        c["axis_y_value_dropdown"],
        c["axis_z_switch"],
        c["axis_z_value_text"],
        c["axis_z_value_dropdown"],
        c["lora_selector"],
    ] + list(c["lora_sliders"].values())
    xy_plot_process_btn.click(
        fn=fn_xy_process_with_deps,
        inputs=xy_plot_input_components,
        outputs=[c["status"], c["output"]],
    ).then(
        fn=f["update_stats"],
        inputs=None,
        outputs=[q["queue_status"], q["queue_stats_display"]],
    ).then(
        fn=f["check_for_current_job"],
        inputs=None,
        outputs=[
            g["current_job_id"],
            g["result_video"],
            g["preview_image"],
            g["top_preview_image"],
            g["progress_desc"],
            g["progress_bar"],
        ],
    ).then(
        fn=f["create_latents_layout_update"],
        inputs=None,
        outputs=[g["top_preview_row"], g["preview_image"]],
    )

    def on_model_type_change(selected_model):
        is_xy = selected_model == "XY Plot"
        is_vid = f["is_video_model"](selected_model)
        shows_end = selected_model in ["Original with Endframe", "Video with Endframe"]
        return (
            gr.update(visible=not is_xy),
            gr.update(visible=is_xy),
            gr.update(visible=not is_xy and not is_vid),
            gr.update(visible=not is_xy and is_vid),
            gr.update(visible=not is_xy and shows_end),
            gr.update(visible=not is_xy and shows_end),
            gr.update(visible=not is_xy),
            gr.update(visible=is_xy),
        )

    g["model_type"].change(
        fn=on_model_type_change,
        inputs=g["model_type"],
        outputs=[
            g["standard_generation_group"],
            c["group"],
            g["image_input_group"],
            g["video_input_group"],
            g["end_frame_group_original"],
            g["end_frame_slider_group"],
            g["start_button"],
            xy_plot_process_btn,
        ],
    ).then(
        fn=f["update_start_button_state"],
        inputs=[g["model_type"], g["input_video"]],
        outputs=[g["start_button"], g["video_input_required_message"]],
    )

    g["input_video"].change(
        fn=f["update_start_button_state"],
        inputs=[g["model_type"], g["input_video"]],
        outputs=[g["start_button"], g["video_input_required_message"]],
    )
    g["input_video"].clear(
        fn=f["update_start_button_state"],
        inputs=[g["model_type"], g["input_video"]],
        outputs=[g["start_button"], g["video_input_required_message"]],
    )

    def update_lora_sliders(selected_loras, all_loras):
        actual_selected = [
            lora for lora in selected_loras if lora != f["DUMMY_LORA_NAME"]
        ]
        updates = [gr.update(value=actual_selected)]
        for lora_name in all_loras:
            updates.append(
                gr.update(
                    visible=(
                        lora_name in actual_selected
                        and lora_name != f["DUMMY_LORA_NAME"]
                    )
                )
            )
        return updates

    g["lora_selector"].change(
        fn=update_lora_sliders,
        inputs=[g["lora_selector"], g["lora_names_states"]],
        outputs=[g["lora_selector"]] + list(g["lora_sliders"].values()),
    )

    def apply_preset(preset_name, model_type_val):
        if not preset_name:
            return [gr.update()] * len(g["ui_components"])
        with open(PRESET_FILE, "r") as f:
            data = json.load(f)
        preset = data.get(model_type_val, {}).get(preset_name, {})
        updates = {key: gr.update(value=preset.get(key)) for key in g["ui_components"]}
        if "lora_values" in preset:
            for name, val in preset["lora_values"].items():
                if name in updates:
                    updates[name] = gr.update(value=val)
        return list(updates.values())

    def save_preset(preset_name, model_type_val, *args):
        if not preset_name:
            return gr.update()
        os.makedirs(os.path.dirname(PRESET_FILE), exist_ok=True)
        data = {}
        if os.path.exists(PRESET_FILE):
            with open(PRESET_FILE, "r") as f:
                data = json.load(f)
        if model_type_val not in data:
            data[model_type_val] = {}
        args_dict = dict(zip(g["ui_components"].keys(), args))
        preset_data = {k: v for k, v in args_dict.items() if k not in g["lora_sliders"]}
        preset_data["lora_values"] = {
            lora: args_dict[lora]
            for lora in args_dict.get("lora_selector", [])
            if lora in args_dict
        }
        data[model_type_val][preset_name] = preset_data
        with open(PRESET_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return gr.update(choices=load_presets(model_type_val), value=preset_name)

    def delete_preset(preset_name, model_type_val):
        if not preset_name:
            return gr.update(), gr.update(visible=True), gr.update(visible=False)
        with open(PRESET_FILE, "r") as f:
            data = json.load(f)
        if model_type_val in data and preset_name in data[model_type_val]:
            del data[model_type_val][preset_name]
        with open(PRESET_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return (
            gr.update(choices=load_presets(model_type_val), value=None),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def refresh_settings_tab_startup_presets(gen_model_type, settings_model_type):
        if gen_model_type == settings_model_type and settings_model_type != "None":
            return gr.update(choices=load_presets(settings_model_type))
        return gr.update()

    g["model_type"].change(
        lambda mt: (
            gr.update(choices=load_presets(mt)),
            gr.update(label=f"{mt} Presets"),
        ),
        inputs=[g["model_type"]],
        outputs=[g["preset_dropdown"], g["preset_accordion"]],
    )
    g["preset_dropdown"].select(
        fn=apply_preset,
        inputs=[g["preset_dropdown"], g["model_type"]],
        outputs=list(g["ui_components"].values()),
    ).then(
        lambda name: name,
        inputs=[g["preset_dropdown"]],
        outputs=[g["preset_name_textbox"]],
    )
    g["save_preset_button"].click(
        fn=save_preset,
        inputs=[g["preset_name_textbox"], g["model_type"]]
        + list(g["ui_components"].values()),
        outputs=[g["preset_dropdown"]],
    ).then(
        fn=refresh_settings_tab_startup_presets,
        inputs=[g["model_type"], s["startup_model_type_dropdown"]],
        outputs=[s["startup_preset_name_dropdown"]],
    )
    g["delete_preset_button"].click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[g["save_preset_button"], g["confirm_delete_row"]],
    )
    g["confirm_delete_no_btn"].click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[g["save_preset_button"], g["confirm_delete_row"]],
    )
    g["confirm_delete_yes_btn"].click(
        fn=delete_preset,
        inputs=[g["preset_dropdown"], g["model_type"]],
        outputs=[
            g["preset_dropdown"],
            g["save_preset_button"],
            g["confirm_delete_row"],
        ],
    )

    def load_metadata_from_json(json_path):
        num_outputs = 20 + len(g["lora_sliders"])
        if not json_path:
            return [gr.update()] * num_outputs
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
            lora_weights = metadata.get("loras", {})
            cache_type_val = metadata.get(
                "cache_type",
                "MagCache"
                if metadata.get("use_magcache")
                else "TeaCache"
                if metadata.get("use_teacache")
                else "None",
            )
            updates = [
                gr.update(value=metadata.get(k))
                for k in [
                    "prompt",
                    "negative_prompt",
                    "seed",
                    "steps",
                    "total_second_length",
                    "end_frame_strength",
                    "model_type",
                ]
            ]
            updates.append(gr.update(value=list(lora_weights.keys())))
            updates.extend(
                [
                    gr.update(value=metadata.get(k))
                    for k in [
                        "latent_window_size",
                        "resolutionW",
                        "resolutionH",
                        "blend_sections",
                    ]
                ]
            )
            updates.extend(
                [
                    gr.update(value=cache_type_val),
                    gr.update(value=metadata.get("magcache_threshold")),
                    gr.update(value=metadata.get("magcache_max_consecutive_skips")),
                    gr.update(value=metadata.get("magcache_retention_ratio")),
                    gr.update(value=metadata.get("teacache_num_steps")),
                    gr.update(value=metadata.get("teacache_rel_l1_thresh")),
                    gr.update(value=metadata.get("latent_type")),
                    gr.update(value=metadata.get("combine_with_source")),
                ]
            )
            for lora in g["lora_names_states"].value:
                updates.append(
                    gr.update(
                        value=lora_weights.get(lora), visible=lora in lora_weights
                    )
                )
            return updates
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            return [gr.update()] * num_outputs

    g["json_upload"].change(
        fn=load_metadata_from_json,
        inputs=[g["json_upload"]],
        outputs=[
            g["prompt"],
            g["n_prompt"],
            g["seed"],
            g["steps"],
            g["total_second_length"],
            g["end_frame_strength_original"],
            g["model_type"],
            g["lora_selector"],
            g["latent_window_size"],
            g["resolutionW"],
            g["resolutionH"],
            g["blend_sections"],
            g["cache_type"],
            g["magcache_threshold"],
            g["magcache_max_consecutive_skips"],
            g["magcache_retention_ratio"],
            g["teacache_num_steps"],
            g["teacache_rel_l1_thresh"],
            g["latent_type"],
            g["combine_with_source"],
        ]
        + list(g["lora_sliders"].values()),
    )

    g["enhance_prompt_btn"].click(
        fn=lambda p: gr.update(value=enhance_prompt(p)),
        inputs=[g["prompt"]],
        outputs=[g["prompt"]],
    )
    g["caption_btn"].click(
        fn=lambda img, p: gr.update(value=caption_image(img)) if img is not None else p,
        inputs=[g["input_image"], g["prompt"]],
        outputs=[g["prompt"]],
    )

    f["block"].load(
        fn=f["check_for_current_job_and_monitor"],
        inputs=[],
        outputs=[
            g["current_job_id"],
            g["result_video"],
            g["preview_image"],
            g["top_preview_image"],
            g["progress_desc"],
            g["progress_bar"],
            q["queue_status"],
            q["queue_stats_display"],
        ],
    ).then(
        fn=f["apply_startup_settings"],
        inputs=None,
        outputs=[g["model_type"], g["preset_dropdown"], g["preset_name_textbox"]]
        + list(g["ui_components"].values())
        + [s["latents_display_top"]],
    ).then(
        fn=f["update_start_button_state"],
        inputs=[g["model_type"], g["input_video"]],
        outputs=[g["start_button"], g["video_input_required_message"]],
    ).then(
        fn=f["create_latents_layout_update"],
        inputs=None,
        outputs=[g["top_preview_row"], g["preview_image"]],
    )
