import gradio as gr
import logging

from modules.version import APP_VERSION_DISPLAY


from diffusers_helper.gradio.progress_bar import make_progress_bar_css
from modules import DUMMY_LORA_NAME  # Import the constant

from modules.toolbox_app import tb_processor
from modules.toolbox_app import (
    tb_create_video_toolbox_ui,
    tb_get_formatted_toolbar_stats,
)
from modules.xy_plot_ui import xy_plot_process
from modules.ui.generate import (
    create_generate_ui,
    connect_generate_events,
    load_presets,
)
from modules.ui.queue import (
    create_queue_ui,
    connect_queue_events,
    update_queue_status_with_thumbnails,
)
from modules.ui.outputs import create_outputs_ui, connect_outputs_events
from modules.ui.settings import create_settings_ui, connect_settings_events

logger = logging.getLogger(__name__)
logger.info("Interface module loaded.")


def create_interface(
    process_fn,
    monitor_fn,
    end_process_fn,
    update_queue_status_fn,
    load_lora_file_fn,
    job_queue,
    settings,
    default_prompt: str = "[1s: The person waves hello] [3s: The person jumps up and down] [5s: The person does a dance]",
    lora_names: list = [],
    lora_values: list = [],
    enumerate_lora_dir_fn=None,
):
    """
    Create the Gradio interface for the video generation application
    """

    # --- Helper Functions ---
    def is_video_model(model_type_value):
        return model_type_value in ["Video", "Video with Endframe", "Video F1"]

    def get_latents_display_top():
        return settings.get("latents_display_top", False)

    def create_latents_layout_update():
        display_top = get_latents_display_top()
        return gr.update(visible=display_top), gr.update(
            visible=not display_top, value=None if display_top else gr.update()
        )

    def update_stats(*args):
        queue_status_data = update_queue_status_with_thumbnails()
        jobs = job_queue.get_all_jobs()
        pending_count = sum(1 for j in jobs if str(j.status) == "JobStatus.PENDING")
        running_count = sum(1 for j in jobs if str(j.status) in ["JobStatus.RUNNING", "JobStatus.CANCELLING"])
        completed_count = sum(1 for j in jobs if str(j.status) == "JobStatus.COMPLETED")
        queue_stats_text = f"<p style='margin:0;color:white;' class='toolbar-text'>Queue: {pending_count} | Running: {running_count} | Completed: {completed_count}</p>"
        return queue_status_data, queue_stats_text

    # --- UI Creation ---
    with open("modules/ui/styles.css", "r") as f:
        css = make_progress_bar_css() + f.read()

    current_theme = settings.get("gradio_theme", "default")
    block = gr.Blocks(css=css, title="FramePack Studio", theme=current_theme).queue()

    with block:
        # --- Toolbar ---
        with gr.Row(elem_id="fixed-toolbar"):
            gr.HTML(
                f"""<div style="display: flex; align-items: center;"><h1 class='toolbar-title'>FP Studio</h1><p class='toolbar-version'>{APP_VERSION_DISPLAY}</p><p class='toolbar-patreon'><a href='https://patreon.com/Colinu' target='_blank'>Support on Patreon</a></p></div>"""
            )
            queue_stats_display = gr.Markdown(
                "<p style='margin:0;color:white;' class='toolbar-text'>Queue: 0 | Running: 0 | Completed: 0</p>"
            )
            toolbar_ram_display_component = gr.Textbox(
                value="RAM: N/A",
                interactive=False,
                lines=1,
                max_lines=1,
                show_label=False,
                container=False,
                elem_id="toolbar-ram-stat",
                elem_classes="toolbar-stat-textbox",
            )
            toolbar_vram_display_component = gr.Textbox(
                value="VRAM: N/A",
                interactive=False,
                lines=1,
                max_lines=1,
                show_label=False,
                container=False,
                elem_id="toolbar-vram-stat",
                elem_classes="toolbar-stat-textbox",
            )
            toolbar_gpu_display_component = gr.Textbox(
                value="GPU: N/A",
                interactive=False,
                lines=1,
                max_lines=1,
                show_label=False,
                container=False,
                elem_id="toolbar-gpu-stat",
                elem_classes="toolbar-stat-textbox",
            )

        # --- Tabs ---
        with gr.Tabs(elem_id="main_tabs") as main_tabs_component:
            with gr.Tab("Generate", id="generate_tab"):
                g = create_generate_ui(
                    lora_names,
                    default_prompt,
                    DUMMY_LORA_NAME,
                    get_latents_display_top,
                    settings,
                )

            with gr.Tab("Queue", id="queue_tab"):
                q = create_queue_ui()
                q["queue_stats_display"] = queue_stats_display

            with gr.Tab("Outputs", id="outputs_tab"):
                o = create_outputs_ui(settings)

            with gr.Tab("Post-processing", id="toolbox_tab"):
                toolbox_ui_layout, tb_target_video_input = tb_create_video_toolbox_ui()

            with gr.Tab("Settings", id="settings_tab"):
                s = create_settings_ui(
                    settings, get_latents_display_top, g["model_type"].choices
                )

        # --- Event Handlers ---
        def check_for_current_job():
            with job_queue.lock:
                current_job = job_queue.current_job
                if current_job:
                    job_id = current_job.id
                    result = current_job.result
                    preview = (
                        current_job.progress_data.get("preview")
                        if current_job.progress_data
                        else None
                    )
                    desc = (
                        current_job.progress_data.get("desc", "")
                        if current_job.progress_data
                        else ""
                    )
                    html = (
                        current_job.progress_data.get("html", "")
                        if current_job.progress_data
                        else ""
                    )
                    logging.info(
                        f"Auto-check found current job {job_id}, triggering monitor_job"
                    )
                    return job_id, result, preview, preview, desc, html
            return None, None, None, None, "", ""

        def check_for_current_job_and_monitor():
            job_id, result, preview, top_preview, desc, html = check_for_current_job()
            queue_status_data, queue_stats_text = update_stats()
            return (
                job_id,
                result,
                preview,
                top_preview,
                desc,
                html,
                queue_status_data,
                queue_stats_text,
            )

        def end_process_with_update():
            end_process_fn()
            queue_status_data, queue_stats_text = update_stats()
            return (
                queue_status_data,
                queue_stats_text,
                gr.update(value="Cancelling...", interactive=False),
                gr.update(value=None),
            )

        def update_start_button_state(*args):
            selected_model, input_video_value = args[-2], args[-1]
            video_provided = input_video_value is not None
            if is_video_model(selected_model) and not video_provided:
                return gr.update(
                    value="❌ Missing Video", interactive=False
                ), gr.update(visible=True)
            return gr.update(value="🚀 Add to Queue", interactive=True), gr.update(
                visible=False
            )

        def apply_startup_settings():
            startup_model_val = settings.get("startup_model_type", "None")
            startup_preset_val = settings.get("startup_preset_name", None)
            model_type_update = gr.update()
            preset_dropdown_update = gr.update()
            preset_name_textbox_update = gr.update()
            ui_components_updates_list = [
                gr.update() for _ in g.get("ui_components", [])
            ]
            if startup_model_val and startup_model_val != "None":
                model_type_update = gr.update(value=startup_model_val)
                presets_for_startup_model = load_presets(startup_model_val)
                preset_dropdown_update = gr.update(choices=presets_for_startup_model)
                if (
                    startup_preset_val
                    and startup_preset_val in presets_for_startup_model
                ):
                    preset_dropdown_update = gr.update(
                        choices=presets_for_startup_model, value=startup_preset_val
                    )
                    preset_name_textbox_update = gr.update(value=startup_preset_val)
                    # This logic needs to be handled carefully, as apply_preset is in generate.py
                    # We will call it from there via the `functions` dict
            latents_display_top_update = gr.update(value=get_latents_display_top())
            return tuple(
                [model_type_update, preset_dropdown_update, preset_name_textbox_update]
                + ui_components_updates_list
                + [latents_display_top_update]
            )

        # --- Connect Events ---
        functions = {
            "is_video_model": is_video_model,
            "get_latents_display_top": get_latents_display_top,
            "create_latents_layout_update": create_latents_layout_update,
            "update_stats": update_stats,
            "check_for_current_job": check_for_current_job,
            "check_for_current_job_and_monitor": check_for_current_job_and_monitor,
            "end_process_with_update": end_process_with_update,
            "update_start_button_state": update_start_button_state,
            "apply_startup_settings": apply_startup_settings,
            "process_fn": process_fn,
            "monitor_fn": monitor_fn,
            "end_process_fn": end_process_fn,
            "xy_plot_process": xy_plot_process,
            "job_queue": job_queue,
            "settings": settings,
            "DUMMY_LORA_NAME": DUMMY_LORA_NAME,
            "block": block,
        }
        connect_generate_events(g, s, q, functions)
        connect_queue_events(q, g, functions, job_queue)
        get_gallery_items_fn = connect_outputs_events(
            o, tb_target_video_input, main_tabs_component
        )
        connect_settings_events(
            s, g, settings, create_latents_layout_update, tb_processor
        )

        def refresh_loras():
            if enumerate_lora_dir_fn:
                new_lora_names = enumerate_lora_dir_fn()
                return gr.update(choices=new_lora_names)
            return gr.update()

        g["refresh_loras_button"].click(fn=refresh_loras, outputs=[g["lora_selector"]])

        # General Connections
        def initial_gallery_load():
            items = get_gallery_items_fn()
            return items, gr.update(value=[item[0] for item in items])

        block.load(
            fn=initial_gallery_load, outputs=[o["gallery_items_state"], o["thumbs"]]
        )
        g["current_job_id"].change(
            fn=monitor_fn,
            inputs=[g["current_job_id"]],
            outputs=[
                g["result_video"],
                g["preview_image"],
                g["top_preview_image"],
                g["progress_desc"],
                g["progress_bar"],
                g["start_button"],
                g["end_button"],
            ],
        ).then(
            fn=update_stats, outputs=[q["queue_status"], q["queue_stats_display"]]
        ).then(
            fn=update_start_button_state,
            inputs=[g["model_type"], g["input_video"]],
            outputs=[g["start_button"], g["video_input_required_message"]],
        ).then(
            fn=create_latents_layout_update,
            outputs=[g["top_preview_row"], g["preview_image"]],
        )
        g["end_button"].click(
            fn=end_process_with_update,
            outputs=[
                q["queue_status"],
                q["queue_stats_display"],
                g["end_button"],
                g["current_job_id"],
            ],
        ).then(
            fn=check_for_current_job_and_monitor,
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
            fn=create_latents_layout_update,
            outputs=[g["top_preview_row"], g["preview_image"]],
        )

        main_toolbar_system_stats_timer = gr.Timer(2, active=True)
        main_toolbar_system_stats_timer.tick(
            fn=tb_get_formatted_toolbar_stats,
            inputs=None,
            outputs=[
                toolbar_ram_display_component,
                toolbar_vram_display_component,
                toolbar_gpu_display_component,
            ],
        )

        # Footer
        with gr.Row(elem_id="footer"):
            gr.HTML(
                f"""<div style="text-align: center; padding: 20px; color: #666;"><div style="margin-top: 10px;"><span class="footer-version" style="margin: 0 10px; color: #666;">{APP_VERSION_DISPLAY}</span><a href="https://patreon.com/Colinu" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;" class="footer-patreon"><i class="fab fa-patreon"></i>Support on Patreon</a><a href="https://discord.gg/MtuM7gFJ3V" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;"><i class="fab fa-discord"></i> Discord</a><a href="https://github.com/colinurbs/FramePack-Studio" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;"><i class="fab fa-github"></i> GitHub</a></div></div>"""
            )

    return block
