"""SageMaker inference handler for OmniVoice TTS.

Loads OmniVoice model on GPU and serves TTS requests via HTTP.
Input:  JSON {text, language, num_step (optional), instruct (optional)}
Output: JSON {audio_base64, sample_rate, duration}
The reference voice is baked in at load time from the bundled WAV (see model_fn).
"""

import json
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000

model = None
voice_clone_prompt = None


def model_fn(model_dir):
    """Load model at startup."""
    global model, voice_clone_prompt
    from omnivoice import OmniVoice

    logger.info("Loading OmniVoice model on CUDA with float16...")
    model = OmniVoice.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")

    # Load voice clone prompt if reference audio exists
    ref_audio_path = os.path.join(model_dir, "voice_ref_th.wav")
    if os.path.exists(ref_audio_path):
        logger.info("Loading voice clone prompt from reference audio...")
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text="สวัสดีค่า นี่คือเสียงพูดภาษาไทย",
        )
        logger.info("Voice clone prompt ready")

    logger.info("Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """Parse input request."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Generate speech audio from text."""
    text = input_data["text"]
    language = input_data.get("language", "Thai")
    num_step = input_data.get("num_step", 16)
    instruct = input_data.get("instruct", None)

    kwargs = {
        "text": text,
        "language": language,
        "num_step": num_step,
    }

    if voice_clone_prompt:
        kwargs["voice_clone_prompt"] = voice_clone_prompt
    elif instruct:
        kwargs["instruct"] = instruct

    with torch.no_grad():
        audios = model.generate(**kwargs)

    audio_np = audios[0]
    pcm_bytes = (audio_np * 32767).astype(np.int16).tobytes()

    return {
        "audio": pcm_bytes,
        "sample_rate": SAMPLE_RATE,
        "duration": len(audio_np) / SAMPLE_RATE,
    }


def output_fn(prediction, response_content_type):
    """Return audio bytes."""
    if response_content_type == "application/octet-stream":
        return prediction["audio"]
    elif response_content_type == "application/json":
        import base64

        return json.dumps(
            {
                "audio_base64": base64.b64encode(prediction["audio"]).decode(),
                "sample_rate": prediction["sample_rate"],
                "duration": prediction["duration"],
            }
        )
    return prediction["audio"]
