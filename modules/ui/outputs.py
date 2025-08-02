import gradio as gr
import os
import json
import logging

logger = logging.getLogger(__name__)


def create_outputs_ui(settings):
    outputDirectory_video = settings.get(
        "output_dir", settings.default_settings["output_dir"]
    )
    outputDirectory_metadata = settings.get(
        "metadata_dir", settings.default_settings["metadata_dir"]
    )

    os.makedirs(outputDirectory_video, exist_ok=True)
    os.makedirs(outputDirectory_metadata, exist_ok=True)

    gallery_items_state = gr.State([])
    selected_original_video_path_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            thumbs = gr.Gallery(
                columns=[4], allow_preview=False, object_fit="cover", height="auto"
            )
            refresh_gallery_button = gr.Button("🔄 Update Gallery")
        with gr.Column(scale=5):
            video_out = gr.Video(sources=[], autoplay=True, loop=True, visible=False)
        with gr.Column(scale=1):
            info_out = gr.Textbox(label="Generation info", visible=False)
            send_to_toolbox_btn = gr.Button("➡️ Send to Post-processing", visible=False)

    return {
        "gallery_items_state": gallery_items_state,
        "selected_original_video_path_state": selected_original_video_path_state,
        "thumbs": thumbs,
        "refresh_gallery_button": refresh_gallery_button,
        "video_out": video_out,
        "info_out": info_out,
        "send_to_toolbox_btn": send_to_toolbox_btn,
        "outputDirectory_video": outputDirectory_video,
        "outputDirectory_metadata": outputDirectory_metadata,
    }


def connect_outputs_events(o, tb_target_video_input, main_tabs_component):
    def get_gallery_items():
        if not os.path.exists(o["outputDirectory_metadata"]):
            logging.error(
                f"Error: Metadata directory not found at {o['outputDirectory_metadata']}"
            )
            return []

        files_with_mtime = []

        all_video_files = os.listdir(o["outputDirectory_video"])

        for f in os.listdir(o["outputDirectory_metadata"]):
            if f.endswith(".png"):
                prefix = os.path.splitext(f)[0]

                matching_videos = []
                for video_file in all_video_files:
                    if video_file.startswith(prefix) and video_file.endswith(".mp4"):
                        matching_videos.append(
                            os.path.join(o["outputDirectory_video"], video_file)
                        )

                if matching_videos:
                    latest_video = max(matching_videos, key=os.path.getmtime)
                    files_with_mtime.append(
                        (
                            os.path.join(o["outputDirectory_metadata"], f),
                            prefix,
                            os.path.getmtime(latest_video),
                        )
                    )

        files_with_mtime.sort(key=lambda x: x[2], reverse=True)

        return [(thumb, prefix) for thumb, prefix, _ in files_with_mtime]

    def refresh_gallery():
        new_items = get_gallery_items()
        return new_items, gr.update(value=[item[0] for item in new_items])

    def get_latest_video_version(prefix):
        max_number = -1
        selected_file = None
        for f in os.listdir(o["outputDirectory_video"]):
            if f.startswith(prefix + "_") and f.endswith(".mp4"):
                if "combined" in f:
                    continue
                try:
                    num_str = f.replace(prefix + "_", "").replace(".mp4", "")
                    if num_str.isdigit():
                        num = int(num_str)
                        if num > max_number:
                            max_number = num
                            selected_file = f
                except (ValueError, TypeError):
                    continue
        return selected_file

    def load_video_and_info_from_prefix(prefix):
        video_file = get_latest_video_version(prefix)
        if not video_file:
            video_file = f"{prefix}.mp4"

        video_path = os.path.join(o["outputDirectory_video"], video_file)
        json_path = os.path.join(o["outputDirectory_metadata"], f"{prefix}.json")

        if not os.path.exists(video_path) or not os.path.exists(json_path):
            return None, "Video or JSON not found.", gr.update(visible=False), None

        with open(json_path, "r", encoding="utf-8") as f:
            info_content = json.load(f)

        return (
            video_path,
            json.dumps(info_content, indent=2, ensure_ascii=False),
            gr.update(visible=True),
            video_path,
        )

    def on_select(gallery_items, evt: gr.SelectData):
        if evt.index is None or not gallery_items or evt.index >= len(gallery_items):
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                None,
            )

        prefix = gallery_items[evt.index][1]
        original_video_path, info_string, button_visibility, new_selected_path = (
            load_video_and_info_from_prefix(prefix)
        )

        video_out_update = gr.update(
            value=original_video_path, visible=bool(original_video_path)
        )
        info_out_update = gr.update(
            value=info_string, visible=bool(original_video_path)
        )

        return video_out_update, info_out_update, button_visibility, new_selected_path

    def send_to_toolbox(selected_video_path):
        return gr.update(value=selected_video_path), gr.update(selected="toolbox_tab")

    o["refresh_gallery_button"].click(
        fn=refresh_gallery, inputs=[], outputs=[o["gallery_items_state"], o["thumbs"]]
    )
    o["thumbs"].select(
        fn=on_select,
        inputs=[o["gallery_items_state"]],
        outputs=[
            o["video_out"],
            o["info_out"],
            o["send_to_toolbox_btn"],
            o["selected_original_video_path_state"],
        ],
    )
    o["send_to_toolbox_btn"].click(
        fn=send_to_toolbox,
        inputs=[o["selected_original_video_path_state"]],
        outputs=[tb_target_video_input, main_tabs_component],
    )

    return get_gallery_items
