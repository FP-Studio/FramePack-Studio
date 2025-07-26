import gradio as gr
from diffusers_helper.lora_utils_kohya_ss.enums import LoraLoader
from modules.ui.generate import load_presets


def create_settings_ui(settings, get_latents_display_top, model_type_choices):
    with gr.Row():
        with gr.Column():
            save_metadata = gr.Checkbox(
                label="Save Metadata",
                info="Save to JSON file",
                value=settings.get("save_metadata", 6),
            )
            gpu_memory_preservation = gr.Slider(
                label="Memory Buffer for Stability (VRAM GB)",
                minimum=1,
                maximum=128,
                step=0.1,
                value=settings.get("gpu_memory_preservation", 6),
                info="Increase reserve if you see computer freezes, stagnant generation, or super slow sampling steps (try 1G at a time). Otherwise smaller buffer is faster. (5.5 - 8.5 is a common range)",
            )
            mp4_crf = gr.Slider(
                label="MP4 Compression",
                minimum=0,
                maximum=100,
                step=1,
                value=settings.get("mp4_crf", 16),
                info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs.",
            )
            clean_up_videos = gr.Checkbox(
                label="Clean up video files",
                value=settings.get("clean_up_videos", True),
                info="If checked, only the final video will be kept after generation.",
            )
            auto_cleanup_on_startup = gr.Checkbox(
                label="Automatically clean up temp folders on startup",
                value=settings.get("auto_cleanup_on_startup", False),
                info="If checked, temporary files (inc. post-processing) will be cleaned up when the application starts.",
            )
            latents_display_top = gr.Checkbox(
                label="Display Next Latents across top of interface",
                value=get_latents_display_top(),
                info="If checked, the Next Latents preview will be displayed across the top of the interface instead of in the right column.",
            )
            gr.Markdown("")
            initial_startup_model_val = settings.get("startup_model_type", "None")
            initial_startup_presets_choices_val = (
                load_presets(initial_startup_model_val)
                if initial_startup_model_val != "None"
                else []
            )
            saved_preset = settings.get("startup_preset_name")
            initial_startup_preset_value_val = (
                saved_preset
                if saved_preset in initial_startup_presets_choices_val
                else None
            )
            startup_model_type_dropdown = gr.Dropdown(
                label="Startup Model Type",
                choices=["None"]
                + [
                    choice[0] for choice in model_type_choices if choice[0] != "XY Plot"
                ],
                value=initial_startup_model_val,
                info="Select a model type to load on startup. 'None' to disable.",
            )
            startup_preset_name_dropdown = gr.Dropdown(
                label="Startup Preset",
                choices=initial_startup_presets_choices_val,
                value=initial_startup_preset_value_val,
                info="Select a preset for the startup model. Updates when Startup Model Type changes.",
                interactive=True,
            )
            with gr.Accordion("System Prompt", open=False):
                with gr.Row(equal_height=True):
                    override_system_prompt = gr.Checkbox(
                        label="Override System Prompt",
                        value=settings.get("override_system_prompt", False),
                        info="If checked, the system prompt template below will be used instead of the default one.",
                        scale=1,
                    )
                    reset_system_prompt_btn = gr.Button("🔄 Reset", scale=0)
                system_prompt_template = gr.Textbox(
                    label="System Prompt Template",
                    value=settings.get(
                        "system_prompt_template",
                        '{"template": "<|start_header_id|>system<|end_header_id|>\\n\\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{}<|eot_id|>", "crop_start": 95}',
                    ),
                    lines=10,
                    info="System prompt template used for video generation. Must be a valid JSON or Python dictionary string with 'template' and 'crop_start' keys.",
                )
            output_dir = gr.Textbox(
                label="Output Directory",
                value=settings.get("output_dir"),
                placeholder="Path to save generated videos",
            )
            metadata_dir = gr.Textbox(
                label="Metadata Directory",
                value=settings.get("metadata_dir"),
                placeholder="Path to save metadata files",
            )
            lora_dir = gr.Textbox(
                label="LoRA Directory",
                value=settings.get("lora_dir"),
                placeholder="Path to LoRA models",
            )
            gradio_temp_dir = gr.Textbox(
                label="Gradio Temporary Directory",
                value=settings.get("gradio_temp_dir"),
            )
            auto_save = gr.Checkbox(
                label="Auto-save settings",
                value=settings.get("auto_save_settings", True),
            )
            gradio_themes = [
                "default",
                "base",
                "soft",
                "glass",
                "mono",
                "origin",
                "citrus",
                "monochrome",
                "ocean",
                "NoCrypt/miku",
                "earneleh/paris",
                "gstaff/xkcd",
            ]
            theme_dropdown = gr.Dropdown(
                label="Theme",
                choices=gradio_themes,
                value=settings.get("gradio_theme", "default"),
                info="Select the Gradio UI theme. Requires restart.",
            )
            with gr.Accordion("Experimental Settings", open=False):
                gr.Markdown(
                    "These settings are for advanced users. Changing them may affect the performance or functionality of the application."
                )
                lora_loader = gr.Dropdown(
                    label="LoRA Loader",
                    choices=[loader.value for loader in LoraLoader],
                    value=settings.lora_loader.value,
                    info="Select the LoRA loader to use. 'diffusers' for Diffusers format, 'lora_ready' for Kohya-ss format.",
                    interactive=True,
                )

                reuse_model_instance = gr.Checkbox(
                    label="Reuse Model Instance",
                    value=settings.get("reuse_model_instance", False),
                    info="If checked, the model instance will be reused across generations to save reload time when no LoRA changes are detected and the same model is used. If unchecked, a new model instance will be created for each generation.",
                )
            save_btn = gr.Button("💾 Save Settings")
            cleanup_btn = gr.Button("🗑️ Clean Up Temporary Files")
            status = gr.HTML("")
            cleanup_output = gr.Textbox(label="Cleanup Status", interactive=False)

    return {
        "save_metadata": save_metadata,
        "gpu_memory_preservation": gpu_memory_preservation,
        "mp4_crf": mp4_crf,
        "clean_up_videos": clean_up_videos,
        "auto_cleanup_on_startup": auto_cleanup_on_startup,
        "latents_display_top": latents_display_top,
        "startup_model_type_dropdown": startup_model_type_dropdown,
        "startup_preset_name_dropdown": startup_preset_name_dropdown,
        "override_system_prompt": override_system_prompt,
        "reset_system_prompt_btn": reset_system_prompt_btn,
        "system_prompt_template": system_prompt_template,
        "output_dir": output_dir,
        "metadata_dir": metadata_dir,
        "lora_dir": lora_dir,
        "gradio_temp_dir": gradio_temp_dir,
        "auto_save": auto_save,
        "theme_dropdown": theme_dropdown,
        "save_btn": save_btn,
        "cleanup_btn": cleanup_btn,
        "status": status,
        "cleanup_output": cleanup_output,
        "lora_loader": lora_loader,
        "reuse_model_instance": reuse_model_instance,
    }


# we can avoid passing around settings object into here, as the Settings class is now a singleton. Remove this comment if done.
def connect_settings_events(s, g, settings, create_latents_layout_update, tb_processor):
    def save_settings_func(*args):
        keys = list(s.keys())
        values = dict(zip(keys, args))
        try:
            settings.save_settings(**values)
            return "<p style='color:green;'>Settings saved successfully! Restart required for theme change.</p>"
        except Exception as e:
            return f"<p style='color:red;'>Error saving settings: {str(e)}</p>"

    def handle_individual_setting_change(key, value):
        if settings.get("auto_save_settings"):
            settings.set(key, value)
            return f"<p style='color:blue;'>'{key}' auto-saved.</p>"
        return f"<p style='color:gray;'>'{key}' changed (auto-save is off).</p>"

    s["save_btn"].click(
        fn=save_settings_func, inputs=list(s.values()), outputs=[s["status"]]
    ).then(
        fn=create_latents_layout_update,
        outputs=[g["top_preview_row"], g["preview_image"]],
    )
    s["reset_system_prompt_btn"].click(
        lambda: (settings.default_settings["system_prompt_template"], False),
        outputs=[s["system_prompt_template"], s["override_system_prompt"]],
    ).then(
        lambda: handle_individual_setting_change(
            "system_prompt_template",
            settings.default_settings["system_prompt_template"],
        ),
        outputs=[s["status"]],
    ).then(
        lambda: handle_individual_setting_change("override_system_prompt", False),
        outputs=[s["status"]],
    )
    s["cleanup_btn"].click(
        fn=tb_processor.tb_clear_temporary_files, outputs=[s["cleanup_output"]]
    )

    for key, comp in s.items():
        if isinstance(comp, (gr.Checkbox, gr.Slider, gr.Dropdown, gr.Textbox)):
            event = comp.change if not isinstance(comp, gr.Textbox) else comp.blur
            event(
                fn=lambda v, k=key: handle_individual_setting_change(k, v),
                inputs=[comp],
                outputs=[s["status"]],
            )

    s["latents_display_top"].change(
        fn=lambda v: create_latents_layout_update(),
        inputs=[s["latents_display_top"]],
        outputs=[g["top_preview_row"], g["preview_image"]],
    )
    s["startup_model_type_dropdown"].change(
        fn=lambda v: gr.update(choices=load_presets(v), value=None),
        inputs=[s["startup_model_type_dropdown"]],
        outputs=[s["startup_preset_name_dropdown"]],
    )
