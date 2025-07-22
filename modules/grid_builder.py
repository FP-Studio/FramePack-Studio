import os
import cv2
import numpy as np
import math
from modules.video_queue import JobStatus


def assemble_grid_video(grid_job, child_jobs, settings):
    """
    Assembles a grid video from the results of child jobs.
    """
    print(f"Starting grid assembly for job {grid_job.id}")

    output_dir = settings.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    video_paths = [
        child.result
        for child in child_jobs
        if child.status == JobStatus.COMPLETED
        and child.result
        and os.path.exists(child.result)
    ]

    if not video_paths:
        print(f"No valid video paths found for grid job {grid_job.id}")
        return None

    print(f"Found {len(video_paths)} videos for grid assembly.")

    # Determine grid size (e.g., 2x2, 3x3)
    num_videos = len(video_paths)
    grid_size = math.ceil(math.sqrt(num_videos))

    # Get video properties from the first video
    try:
        cap = cv2.VideoCapture(video_paths[0])
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_paths[0]}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    except Exception as e:
        print(f"Error getting video properties from {video_paths[0]}: {e}")
        return None

    output_filename = os.path.join(output_dir, f"grid_{grid_job.id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_filename, fourcc, fps, (width * grid_size, height * grid_size)
    )

    caps = [cv2.VideoCapture(p) for p in video_paths]

    while True:
        frames = []
        all_frames_read = True
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                # If one video ends, stop processing
                all_frames_read = False
                break

        if not all_frames_read or not frames:
            break

        # Create a blank canvas for the grid
        grid_frame = np.zeros(
            (height * grid_size, width * grid_size, 3), dtype=np.uint8
        )

        # Place frames into the grid
        for i, frame in enumerate(frames):
            row = i // grid_size
            col = i % grid_size
            grid_frame[
                row * height : (row + 1) * height, col * width : (col + 1) * width
            ] = frame

        video_writer.write(grid_frame)

    for cap in caps:
        cap.release()
    video_writer.release()

    print(f"Grid video saved to {output_filename}")
    return output_filename
