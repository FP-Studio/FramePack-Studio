import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, StoppingCriteria


# --- Job State ---
class JobState:
    def __init__(self):
        self.running = False
        self.interrupted = False

    def reset(self):
        self.running = False
        self.interrupted = False


captioner_job_state = JobState()


class StopCaptionerCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return captioner_job_state.interrupted


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = None
processor = None


def _load_captioning_model():
    """Load the Florence-2"""
    global model, processor
    if model is None or processor is None:
        print("Loading Florence-2 model for image captioning...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
        print("Florence-2 model loaded successfully.")


def stop_captioning():
    captioner_job_state.interrupted = True
    # Unload model when stopped to free memory
    unload_captioning_model()
    import gradio as gr
    return (
        gr.update(interactive=True), # re-enable enhance button
        gr.update(visible=False),    # hide stop caption button
        gr.update(interactive=True)  # re-enable caption button
    )


def is_captioning():
    return captioner_job_state.running


def unload_captioning_model():
    """Unload the Florence-2"""
    global model, processor
    if model is not None:
        del model
        model = None
    if processor is not None:
        del processor
        processor = None
    torch.cuda.empty_cache()
    print("Florence-2 model unloaded successfully.")


prompt = "<MORE_DETAILED_CAPTION>"


# The image parameter now directly accepts a PIL Image object
def caption_image(image: np.array):
    """
    Args:
        image_np (np.ndarray): The input image as a NumPy array (e.g., HxWx3, RGB).
                                Gradio passes this when type="numpy" is set.
    """
    captioner_job_state.reset()
    captioner_job_state.running = True

    try:
        _load_captioning_model()

        if image is None:
            return ""

        image_pil = Image.fromarray(image)

        inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(
            device, torch_dtype
        )

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            stopping_criteria=[StopCaptionerCriteria()],
        )

        if captioner_job_state.interrupted:
            # Unload model if interrupted
            unload_captioning_model()
            return ""

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Unload model after successful completion to free memory
        unload_captioning_model()
        return generated_text
    finally:
        captioner_job_state.running = False
