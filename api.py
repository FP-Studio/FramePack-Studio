import json
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from modules.studio_manager import StudioManager
import numpy as np
from PIL import Image
from diffusers_helper.utils import generate_timestamp

app = FastAPI()
studio_manager = StudioManager()
job_queue = studio_manager.job_queue
settings = studio_manager.settings


@app.post("/upload_metadata/")
async def upload_metadata(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        metadata = json.loads(contents)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")


@app.post("/add_to_queue/")
async def add_to_queue(params: dict):
    try:
        job_params = params.copy()

        # Handle input files - copy to input_files_dir to prevent them from being deleted by temp cleanup
        input_files_dir = settings.get("input_files_dir")
        os.makedirs(input_files_dir, exist_ok=True)

        # Process input image (if it's a file path)
        input_image_path = None
        if (
            "input_image" in job_params
            and isinstance(job_params["input_image"], str)
            and os.path.exists(job_params["input_image"])
        ):
            # It's a file path, copy it to input_files_dir
            filename = os.path.basename(job_params["input_image"])
            input_image_path = os.path.join(
                input_files_dir, f"{generate_timestamp()}_{filename}"
            )
            try:
                shutil.copy2(job_params["input_image"], input_image_path)
                print(f"Copied input image to {input_image_path}")
                # For Video model, we'll use the path
                if job_params.get("model_type") == "Video":
                    job_params["input_image"] = input_image_path
                else:
                    job_params["input_image"] = np.array(Image.open(input_image_path))

            except Exception as e:
                print(f"Error copying input image: {e}")

        job_params["input_image_path"] = input_image_path

        # Add other necessary parameters that might be missing from the metadata file
        job_params.setdefault("output_dir", settings.get("output_dir"))
        job_params.setdefault("metadata_dir", settings.get("metadata_dir"))
        job_params.setdefault("input_files_dir", input_files_dir)
        job_params.setdefault("lora_loaded_names", [])
        job_params.setdefault("lora_values", [])

        job_id = job_queue.add_job(job_params)
        job = job_queue.get_job(job_id)
        if job:
            job.generation_type = job_params.get("model_type")

        return {"message": "Job added to queue", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding job to queue: {e}")
