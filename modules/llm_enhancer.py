import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria


# --- Job State ---
class JobState:
    def __init__(self):
        self.running = False
        self.interrupted = False

    def reset(self):
        self.running = False
        self.interrupted = False


enhancer_job_state = JobState()


class StopEnhancerCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return enhancer_job_state.interrupted


# --- Configuration ---
# Using a smaller, faster model for this feature.
# This can be moved to a settings file later.
MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = (
    "You are a tool to enhance video prompts. Make them more detailed and cinematic while keeping them concise.\n"
    "Requirements:\n"
    "1. Keep enhanced prompts under 50 words per section\n"
    "2. Add visual details, camera angles, and motion\n"
    "3. Preserve the original meaning exactly\n"
    "4. Use clear, descriptive language\n"
    "5. End sentences properly - never cut off mid-sentence\n"
    "6. Focus on key visual elements and movement\n\n"
    "Examples:\n"
    "Input: 'person waves hello'\n"
    "Output: 'A person with a warm smile waves their hand in greeting, fingers gently moving in a friendly gesture. Medium shot with soft lighting.'\n\n"
    "Input: 'cat plays guitar'\n"
    "Output: 'A playful cat sits on a stool, strumming guitar strings with its paws. Close-up shot capturing the cat's focused expression and paw movements.'"
)
PROMPT_TEMPLATE = (
    "Enhance this prompt to be more cinematic and detailed, but keep it under 50 words. "
    "Preserve the original meaning exactly and end with complete sentences.\n\n"
    'Prompt: "{text_to_enhance}"\n\nEnhanced:'
)

# --- Model Loading (cached) ---
model = None
tokenizer = None


def _load_enhancing_model():
    """Loads the model and tokenizer, caching them globally."""
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"LLM Enhancer: Loading model '{MODEL_NAME}' to {DEVICE}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
        print("LLM Enhancer: Model loaded successfully.")


def _load_enhancing_model_with_progress():
    """Loads the model and tokenizer with progress updates."""
    global model, tokenizer
    if model is None or tokenizer is None:
        yield f"Loading model '{MODEL_NAME}' to {DEVICE}..."
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        yield "Loading model weights..."
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
        yield "Model loaded successfully"
    else:
        yield "Model already loaded"


def _run_inference(text_to_enhance: str) -> str:
    """Runs the LLM inference to enhance a single piece of text."""

    if enhancer_job_state.interrupted:
        return ""

    formatted_prompt = PROMPT_TEMPLATE.format(text_to_enhance=text_to_enhance)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # When tokenizing, the tokenizer can return the attention_mask directly.
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(DEVICE)

    # Set pad_token_id to eos_token_id if it's not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=80,  # Further reduced to prevent cutoffs and keep responses concise
        do_sample=True,
        temperature=0.3,  # Reduced from 0.5 for more focused output
        top_p=0.9,  # Reduced from 0.95 for faster sampling
        top_k=20,  # Reduced from 30 for faster sampling
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=[StopEnhancerCriteria()],
        use_cache=True,  # Enable KV-cache for faster generation
    )

    if enhancer_job_state.interrupted:
        return ""

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Clean up the response
    response = response.strip().replace('"', "")
    return response


def _run_inference_with_progress(text_to_enhance: str):
    """Runs the LLM inference with progress updates."""
    if enhancer_job_state.interrupted:
        return ""

    yield "Preparing prompt..."
    formatted_prompt = PROMPT_TEMPLATE.format(text_to_enhance=text_to_enhance)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    yield "Tokenizing input..."
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(DEVICE)

    # Set pad_token_id to eos_token_id if it's not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    yield "Generating enhanced text..."
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=80,  # Optimized for concise, complete responses
        do_sample=True,
        temperature=0.3,  # More focused output
        top_p=0.9,  # Faster sampling
        top_k=20,  # Faster sampling
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=[StopEnhancerCriteria()],
        use_cache=True,  # Enable KV-cache
    )

    if enhancer_job_state.interrupted:
        return ""

    yield "Processing response..."
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Clean up the response
    response = response.strip().replace('"', "")
    yield response


def stop_enhancing():
    enhancer_job_state.interrupted = True
    # Unload model when stopped to free memory
    unload_enhancing_model()
    import gradio as gr

    return (
        gr.update(interactive=True),  # re-enable enhance button
        gr.update(visible=False),  # hide stop enhance button
        gr.update(interactive=True),  # re-enable caption button
    )


def is_enhancing():
    return enhancer_job_state.running


def unload_enhancing_model():
    global model, tokenizer
    if model is not None:
        del model
        model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    torch.cuda.empty_cache()


def enhance_prompt(prompt_text: str) -> str:
    """
    Enhances a prompt, handling both plain text and timestamped formats.

    Args:
        prompt_text: The user's input prompt.

    Returns:
        The enhanced prompt string.
    """
    enhancer_job_state.reset()
    enhancer_job_state.running = True

    try:
        _load_enhancing_model()

        if not prompt_text:
            return ""

        # Regex to find timestamp sections like [0s: text] or [1.1s-2.2s: text]
        timestamp_pattern = r"(\[\d+(?:\.\d+)?s(?:-\d+(?:\.\d+)?s)?\s*:\s*)(.*?)(?=\])"

        matches = list(re.finditer(timestamp_pattern, prompt_text))

        if not matches:
            # No timestamps found, enhance the whole prompt
            print("LLM Enhancer: Enhancing a simple prompt.")
            result = _run_inference(prompt_text)
        else:
            # Timestamps found, enhance each section's text
            print(
                f"LLM Enhancer: Enhancing {len(matches)} sections in a timestamped prompt."
            )
            enhanced_parts = []
            last_end = 0

            for match in matches:
                if enhancer_job_state.interrupted:
                    return ""
                # Add the part of the string before the current match (e.g., whitespace)
                enhanced_parts.append(prompt_text[last_end : match.start()])

                timestamp_prefix = match.group(1)
                text_to_enhance = match.group(2).strip()

                if text_to_enhance:
                    enhanced_text = _run_inference(text_to_enhance)
                    enhanced_parts.append(f"{timestamp_prefix}{enhanced_text}")
                else:
                    # Keep empty sections as they are
                    enhanced_parts.append(f"{timestamp_prefix}")

                last_end = match.end()

            # Add the closing bracket for the last match and any trailing text
            enhanced_parts.append(prompt_text[last_end:])

            result = "".join(enhanced_parts)

        # Unload model after successful completion to free memory
        unload_enhancing_model()
        return result
    finally:
        enhancer_job_state.running = False


def enhance_prompt_with_progress(prompt_text: str):
    """
    Enhances a prompt with progress updates, handling both plain text and timestamped formats.

    Args:
        prompt_text: The user's input prompt.

    Yields:
        Status messages during processing, final result as last yield.
    """
    enhancer_job_state.reset()
    enhancer_job_state.running = True

    try:
        # Load model with progress updates
        for status in _load_enhancing_model_with_progress():
            if enhancer_job_state.interrupted:
                # Unload model if interrupted during loading
                unload_enhancing_model()
                return
            yield status

        if not prompt_text:
            # Unload model if no prompt provided
            unload_enhancing_model()
            yield ""
            return

        # Regex to find timestamp sections like [0s: text] or [1.1s-2.2s: text]
        timestamp_pattern = r"(\[\d+(?:\.\d+)?s(?:-\d+(?:\.\d+)?s)?\s*:\s*)(.*?)(?=\])"

        matches = list(re.finditer(timestamp_pattern, prompt_text))

        final_result = None
        if not matches:
            # No timestamps found, enhance the whole prompt
            yield "Enhancing simple prompt..."
            for result in _run_inference_with_progress(prompt_text):
                if enhancer_job_state.interrupted:
                    # Unload model if interrupted during inference
                    unload_enhancing_model()
                    return
                if isinstance(result, str) and not result.startswith(
                    ("Preparing", "Tokenizing", "Generating", "Processing")
                ):
                    final_result = result
                else:
                    yield result
        else:
            # Timestamps found, enhance each section's text
            yield f"Enhancing {len(matches)} sections in timestamped prompt..."
            enhanced_parts = []
            last_end = 0

            for i, match in enumerate(matches, 1):
                if enhancer_job_state.interrupted:
                    # Unload model if interrupted during processing
                    unload_enhancing_model()
                    return

                yield f"Processing section {i}/{len(matches)}..."

                # Add the part of the string before the current match (e.g., whitespace)
                enhanced_parts.append(prompt_text[last_end : match.start()])

                timestamp_prefix = match.group(1)
                text_to_enhance = match.group(2).strip()

                if text_to_enhance:
                    enhanced_text = None
                    for result in _run_inference_with_progress(text_to_enhance):
                        if enhancer_job_state.interrupted:
                            # Unload model if interrupted during section processing
                            unload_enhancing_model()
                            return
                        if isinstance(result, str) and not result.startswith(
                            ("Preparing", "Tokenizing", "Generating", "Processing")
                        ):
                            enhanced_text = result
                        else:
                            yield f"Section {i}/{len(matches)}: {result}"

                    if enhanced_text:
                        enhanced_parts.append(f"{timestamp_prefix}{enhanced_text}")
                else:
                    # Keep empty sections as they are
                    enhanced_parts.append(f"{timestamp_prefix}")

                last_end = match.end()

            # Add the closing bracket for the last match and any trailing text
            enhanced_parts.append(prompt_text[last_end:])

            final_result = "".join(enhanced_parts)

        # Unload model after successful completion to free memory
        yield "Cleaning up..."
        unload_enhancing_model()
        yield final_result
    finally:
        enhancer_job_state.running = False
