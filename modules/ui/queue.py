import gradio as gr
import time
import logging
from modules.video_queue import JobStatus


def format_queue_status(jobs):
    rows = []
    for job in jobs:
        created = (
            time.strftime("%H:%M:%S", time.localtime(job.created_at))
            if job.created_at
            else ""
        )
        started = (
            time.strftime("%H:%M:%S", time.localtime(job.started_at))
            if job.started_at
            else ""
        )
        completed = (
            time.strftime("%H:%M:%S", time.localtime(job.completed_at))
            if job.completed_at
            else ""
        )
        elapsed_time = ""
        if job.started_at:
            end_time = job.completed_at or time.time()
            elapsed_seconds = end_time - job.started_at
            status_suffix = "" if job.completed_at else " (running)"
            elapsed_time = f"{elapsed_seconds:.2f}s{status_suffix}"
        generation_type = getattr(job, "generation_type", "Original")
        thumbnail = getattr(job, "thumbnail", None)
        thumbnail_html = (
            f'<img src="{thumbnail}" width="64" height="64" style="object-fit: contain;">'
            if thumbnail
            else ""
        )
        rows.append(
            [
                job.id[:6] + "...",
                generation_type,
                job.status.value,
                created,
                started,
                completed,
                elapsed_time,
                thumbnail_html,
            ]
        )
    return rows


def update_queue_status_with_thumbnails():
    try:
        from __main__ import job_queue

        jobs = job_queue.get_all_jobs()
        for job in jobs:
            if job.status == JobStatus.PENDING:
                job.queue_position = job_queue.get_queue_position(job.id)
        if job_queue.current_job:
            job_queue.current_job.status = JobStatus.RUNNING
        return format_queue_status(jobs)
    except ImportError:
        logging.error("Error: Could not import job_queue. Queue status update might fail.")
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
                queue_status = gr.DataFrame(
                    headers=[
                        "Job ID",
                        "Type",
                        "Status",
                        "Created",
                        "Started",
                        "Completed",
                        "Elapsed",
                        "Preview",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str", "str", "html"],
                    label="Job Queue",
                )
            with gr.Accordion("Queue Documentation", open=False):
                gr.Markdown("""
                ## Queue Tab Guide
                
                This tab is for managing your generation jobs.
                
                - **Refresh Queue**: Update the job list.
                - **Cancel Queue**: Stop all pending jobs.
                - **Clear Complete**: Remove finished, failed, or cancelled jobs from the list.
                - **Load Queue**: Load jobs from the default `queue.json`.
                - **Export Queue**: Save the current job list and its images to a zip file.
                - **Import Queue**: Load a queue from a `.json` or `.zip` file.
                """)
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
