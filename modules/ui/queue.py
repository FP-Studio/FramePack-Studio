import gradio as gr
import time
import logging
from modules.video_queue import JobStatus
from modules.ui.template_loader import (
    render_thumbnail_html,
    get_queue_documentation,
    render_queue_row,
    render_queue,
)

logger = logging.getLogger(__name__)


def format_queue_status(jobs):
    """Format queue status with custom HTML templates."""

    rows = []
    for job in jobs:
        elapsed_time = ""
        if job.started_at:
            end_time = job.completed_at or time.time()
            elapsed_seconds = end_time - job.started_at
            elapsed_time = f"{elapsed_seconds:.2f}s"

        # Get job data
        generation_type = getattr(job, "generation_type", "Original")
        thumbnail = getattr(job, "thumbnail", None)
        thumbnail_html = render_thumbnail_html(thumbnail)

        # Get job settings from job.params (where JSON data is actually stored)
        width = job.params.get("resolutionW", 512) if hasattr(job, "params") else 512
        height = job.params.get("resolutionH", 512) if hasattr(job, "params") else 512
        size = f"{width}×{height}"

        # Get video length from total_second_length (already in seconds)
        total_seconds = (
            job.params.get("total_second_length", 6) if hasattr(job, "params") else 6
        )
        length = f"{total_seconds}s" if total_seconds else "N/A"

        # Get seed from params
        seed = job.params.get("seed", "Random") if hasattr(job, "params") else "Random"

        # Get prompt from params
        prompt = job.params.get("prompt_text", "") if hasattr(job, "params") else ""

        # Get all job data for changed settings detection from job.params
        params = job.params if hasattr(job, "params") else {}
        job_data = {
            "prompt": prompt,
            "negative_prompt": params.get("n_prompt", ""),
            "seed": seed,
            "steps": params.get("steps", 25),
            "cfg": params.get("cfg", 1.0),
            "gs": params.get("gs", 10),
            "rs": params.get("rs", 0),
            "latent_type": params.get("latent_type", "Noise"),
            "latent_window_size": params.get("latent_window_size", 9),
            "resolutionW": params.get("resolutionW", 512),
            "resolutionH": params.get("resolutionH", 512),
            "model_type": params.get("model_type", generation_type),
            "generation_type": params.get("model_type", generation_type),
            "total_second_length": params.get("total_second_length", 6),
            "blend_sections": params.get("blend_sections", 4),
            "num_cleaned_frames": params.get("num_cleaned_frames", 0),
            "end_frame_strength": params.get("end_frame_strength", 1.0),
            "end_frame_image_path": params.get("end_frame_image_path", None),
            "end_frame_used": params.get("end_frame_used", "False"),
            "input_video": params.get("input_video", None),
            "video_path": params.get("video_path", None),
            "x_param": params.get("x_param", None),
            "y_param": params.get("y_param", None),
            "x_values": params.get("x_values", None),
            "y_values": params.get("y_values", None),
            "combine_with_source": params.get("combine_with_source", True),
            "use_teacache": params.get("use_teacache", False),
            "teacache_num_steps": params.get("teacache_num_steps", 25),
            "teacache_rel_l1_thresh": params.get("teacache_rel_l1_thresh", 0.15),
            "use_magcache": params.get("use_magcache", True),
            "magcache_threshold": params.get("magcache_threshold", 0.1),
            "magcache_max_consecutive_skips": params.get(
                "magcache_max_consecutive_skips", 2
            ),
            "magcache_retention_ratio": params.get("magcache_retention_ratio", 0.25),
            "loras": params.get("loras", {}),
        }

        # Render each row using the template
        row_html = render_queue_row(
            job_id=job.id[:6] + "...",
            job_id_full=job.id,
            generation_type=generation_type,
            status=job.status.value,
            prompt_text=prompt,
            size=size,
            length=length,
            seed=str(seed),
            started=time.strftime("%H:%M:%S", time.localtime(job.started_at))
            if job.started_at
            else "--",
            completed=time.strftime("%H:%M:%S", time.localtime(job.completed_at))
            if job.completed_at
            else "--",
            elapsed=elapsed_time or "--",
            thumbnail=thumbnail_html,
            job_data=job_data,
        )
        rows.append(row_html)

    # Render the complete queue
    return render_queue(rows)


def update_queue_status_with_thumbnails():
    try:
        from __main__ import job_queue

        jobs = job_queue.get_all_jobs()
        for job in jobs:
            if job.status == JobStatus.PENDING:
                job.queue_position = job_queue.get_queue_position(job.id)
        if job_queue.current_job:
            # Only set to RUNNING if not already in a cancellation state
            if job_queue.current_job.status not in [
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
            ]:
                job_queue.current_job.status = JobStatus.RUNNING

        # Sort jobs according to the requested priority:
        # 1. Running and Cancelling items first (same job transitioning states)
        # 2. Queued (pending) items in ascending order by created_at
        # 3. Editing items (future feature)
        # 4. Completed, Failed, and Cancelled items in descending order by completed_at
        def sort_key(job):
            if job.status in [JobStatus.RUNNING, JobStatus.CANCELLING]:
                return (
                    0,
                    job.created_at,
                )  # Running/Cancelling items first (same priority - never coexist)
            elif job.status == JobStatus.PENDING:
                # Sort by order number if available, otherwise fall back to created_at
                return (
                    1,
                    job.order_number
                    if job.order_number is not None
                    else job.created_at,
                )
            elif job.status == JobStatus.EDITING:
                return (
                    2,
                    job.created_at,
                )  # Editing items third, ordered by creation time
            else:  # COMPLETED, FAILED, CANCELLED
                # Use completed_at if available, otherwise use created_at, with descending order
                timestamp = job.completed_at if job.completed_at else job.created_at
                return (
                    3,
                    -timestamp,
                )  # Completed/Failed/Cancelled items last, descending order

        jobs.sort(key=sort_key)

        return format_queue_status(jobs)
    except ImportError:
        logging.error(
            "Error: Could not import job_queue. Queue status update might fail."
        )
        return []
    except Exception as e:
        logging.error(f"Error updating queue status: {e}")
        return []


def create_queue_ui():
    with gr.Row():
        with gr.Column():
            with gr.Row() as queue_controls_row:
                refresh_button = gr.Button("🔄 Refresh Queue")
                load_queue_button = gr.Button("▶️ Resume Queue")
                queue_export_button = gr.Button("📦 Export Queue")
                clear_complete_button = gr.Button(
                    "🧹 Clear Completed Jobs", variant="secondary"
                )
                clear_queue_button = gr.Button("❌ Cancel Queued Jobs", variant="stop")
            with gr.Row():
                import_queue_file = gr.File(
                    label="Import Queue",
                    file_types=[".json", ".zip"],
                    type="filepath",
                    visible=True,
                    elem_classes="short-import-box",
                )
            with gr.Row(visible=False) as confirm_cancel_row:
                gr.Markdown("### Are you sure you want to cancel all pending jobs?")
                confirm_cancel_yes_btn = gr.Button("❌ Yes, Cancel All", variant="stop")
                confirm_cancel_no_btn = gr.Button("↩️ No, Go Back")
            with gr.Row():
                # Create custom queue HTML component
                queue_status = gr.HTML(
                    render_queue([]),  # Start with empty queue
                    label="Job Queue",
                )
            with gr.Accordion("Queue Documentation", open=False):
                gr.Markdown(get_queue_documentation())
    return {
        "queue_status": queue_status,
        "refresh_button": refresh_button,
        "load_queue_button": load_queue_button,
        "queue_export_button": queue_export_button,
        "clear_complete_button": clear_complete_button,
        "clear_queue_button": clear_queue_button,
        "import_queue_file": import_queue_file,
        "queue_controls_row": queue_controls_row,
        "confirm_cancel_row": confirm_cancel_row,
        "confirm_cancel_yes_btn": confirm_cancel_yes_btn,
        "confirm_cancel_no_btn": confirm_cancel_no_btn,
    }


def connect_queue_events(q, g, f, job_queue):
    def clear_all_jobs():
        job_queue.clear_queue()
        return f["update_stats"]()

    def clear_completed_jobs():
        job_queue.clear_completed_jobs()
        return f["update_stats"]()

    def load_queue_from_json():
        job_queue.load_queue_from_json()
        return f["update_stats"]()

    def import_queue_from_file(file_path):
        if file_path:
            job_queue.load_queue_from_json(file_path)
        return f["update_stats"]()

    def export_queue_to_zip():
        job_queue.export_queue_to_zip()
        return f["update_stats"]()

    q["refresh_button"].click(
        fn=f["update_stats"],
        inputs=[],
        outputs=[q["queue_status"], q["queue_stats_display"]],
    )
    q["clear_queue_button"].click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[q["queue_controls_row"], q["confirm_cancel_row"]],
    )
    q["confirm_cancel_no_btn"].click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[q["queue_controls_row"], q["confirm_cancel_row"]],
    )
    q["confirm_cancel_yes_btn"].click(
        fn=lambda: clear_all_jobs()
        + (gr.update(visible=True), gr.update(visible=False)),
        outputs=[
            q["queue_status"],
            q["queue_stats_display"],
            q["queue_controls_row"],
            q["confirm_cancel_row"],
        ],
    )
    q["clear_complete_button"].click(
        fn=clear_completed_jobs,
        inputs=[],
        outputs=[q["queue_status"], q["queue_stats_display"]],
    )
    q["queue_export_button"].click(
        fn=export_queue_to_zip,
        inputs=[],
        outputs=[q["queue_status"], q["queue_stats_display"]],
    )
    q["load_queue_button"].click(
        fn=load_queue_from_json,
        inputs=[],
        outputs=[q["queue_status"], q["queue_stats_display"]],
    ).then(
        fn=f["check_for_current_job"],
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
        outputs=[g["top_preview_row"], g["preview_image"]],
    )
    q["import_queue_file"].change(
        fn=import_queue_from_file,
        inputs=[q["import_queue_file"]],
        outputs=[q["queue_status"], q["queue_stats_display"]],
    ).then(
        fn=f["check_for_current_job"],
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
        outputs=[g["top_preview_row"], g["preview_image"]],
    )
