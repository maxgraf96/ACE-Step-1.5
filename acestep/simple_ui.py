"""
ACE-Step Simple UI - Streamlined interface for executives
A clean, simplified Gradio interface for music generation
"""

import os
import sys
import json
import random
import argparse
import tempfile
import time
import urllib.parse
import urllib.request
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    _current_file = os.path.abspath(__file__)
    _project_root = os.path.dirname(_current_file)
    _env_path = os.path.join(_project_root, ".env")
    _env_example_path = os.path.join(_project_root, ".env.example")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
    elif os.path.exists(_env_example_path):
        load_dotenv(_env_example_path)
except ImportError:
    pass

# Clear proxy settings
for proxy_var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(proxy_var, None)

import gradio as gr
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, understand_music, create_sample, format_sample, GenerationParams, GenerationConfig
from acestep.constants import TASK_TYPES_TURBO, TASK_TYPES_BASE, VALID_LANGUAGES
from acestep.gpu_config import get_gpu_config, set_global_gpu_config


DEFAULT_API_BASE_URL = os.getenv("ACESTEP_SIMPLE_UI_API_BASE_URL", "").strip()
DEFAULT_API_KEY = os.getenv("ACESTEP_API_KEY", "").strip() or None


# Custom CSS for a clean, professional look
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --font-main: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --color-primary: #6366F1;
    --color-primary-hover: #4F46E5;
    --color-bg: #111827;
    --color-surface: #1F2937;
    --color-border: #374151;
    --color-text: #F9FAFB;
    --color-text-secondary: #D1D5DB;
}

/* Force Gradio to dark mode regardless of system/browser preference */
.gradio-container,
.gradio-container.light,
.gradio-container.dark {
    color-scheme: dark !important;
    --body-background-fill: #111827 !important;
    --body-background-fill-subdued: #0F172A !important;
    --background-fill-primary: #1F2937 !important;
    --background-fill-secondary: #111827 !important;
    --block-background-fill: #1F2937 !important;
    --block-border-color: #374151 !important;
    --border-color-primary: #374151 !important;
    --input-background-fill: #111827 !important;
    --input-border-color: #374151 !important;
    --input-placeholder-color: #9CA3AF !important;
    --body-text-color: #F9FAFB !important;
    --body-text-color-subdued: #D1D5DB !important;
    --block-label-text-color: #F9FAFB !important;
}

* {
    font-family: var(--font-main) !important;
}

body {
    background-color: var(--color-bg) !important;
}

.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--color-border);
}

.header h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #F9FAFB !important;
    margin-bottom: 0.5rem;
    letter-spacing: -0.025em;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.header p {
    font-size: 1rem;
    color: #E5E7EB !important;
    font-weight: 400;
}

.section-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #E5E7EB !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

.gr-button-primary {
    background-color: var(--color-primary) !important;
    border-color: var(--color-primary) !important;
    font-weight: 500 !important;
    border-radius: 0.5rem !important;
    transition: all 0.2s ease !important;
}

.gr-button-primary:hover {
    background-color: var(--color-primary-hover) !important;
    border-color: var(--color-primary-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
}

.gr-button-secondary {
    border-radius: 0.5rem !important;
    font-weight: 500 !important;
    border-color: var(--color-border) !important;
}

.gr-box {
    border-radius: 0.75rem !important;
    border-color: var(--color-border) !important;
    background-color: var(--color-surface) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

.gr-input, .gr-textarea, .gr-dropdown {
    border-radius: 0.5rem !important;
    border-color: var(--color-border) !important;
    font-size: 0.9375rem !important;
}

.gr-input:focus, .gr-textarea:focus {
    border-color: var(--color-primary) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
}

.gr-accordion {
    border-radius: 0.75rem !important;
    border: 1px solid var(--color-border) !important;
    background-color: var(--color-surface) !important;
}

.gr-audio {
    border-radius: 0.5rem !important;
}

.gr-audio label,
.gr-audio .label-wrap,
.gr-audio .label-wrap span,
.gr-audio .label-wrap p {
    color: var(--color-text) !important;
    opacity: 1 !important;
}

.gr-audio button,
.gr-audio [role="button"] {
    color: var(--color-text) !important;
    border-color: var(--color-border) !important;
    background-color: var(--color-surface) !important;
}

.gr-audio .wrap,
.gr-audio .container,
.gr-audio .empty {
    background-color: var(--color-surface) !important;
    border-color: var(--color-border) !important;
}

.status-box {
    font-family: 'SF Mono', Monaco, monospace !important;
    font-size: 0.8125rem !important;
    border-radius: 0.5rem !important;
}

/* Compact rows */
.compact-row {
    gap: 1rem !important;
}

/* Model selection section */
.model-section {
    background: linear-gradient(135deg, #1F2937 0%, #374151 100%);
    padding: 1.25rem;
    border-radius: 0.75rem;
    margin-bottom: 1.5rem;
    border: 1px solid #4B5563;
}

/* Generation button */
.generate-btn {
    font-size: 1.125rem !important;
    padding: 0.875rem 2rem !important;
    margin-top: 1rem !important;
}

/* Audio output cards */
.audio-output {
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 0.75rem;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Hide some Gradio default styling */
.gr-form {
    gap: 0.75rem !important;
}

.gr-checkbox {
    align-items: center !important;
}

label {
    font-weight: 500 !important;
    color: var(--color-text) !important;
}

/* Tips and hints */
.hint-text {
    font-size: 0.8125rem;
    color: var(--color-text-secondary);
    font-style: italic;
}

/* Prominent checkbox for instrumental */
.prominent-checkbox label {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #F9FAFB !important;
    padding: 0.75rem 1rem !important;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    border-radius: 0.5rem !important;
    border: 2px solid #6366F1 !important;
}

.prominent-checkbox input[type="checkbox"] {
    width: 1.5rem !important;
    height: 1.5rem !important;
}
"""


def _normalize_api_base_url(api_base_url: Optional[str]) -> Optional[str]:
    """Normalize API base URL for consistent endpoint composition."""
    if not api_base_url:
        return None
    normalized = api_base_url.strip().rstrip("/")
    return normalized or None


def _build_auth_headers(api_key: Optional[str]) -> dict:
    """Build optional authorization headers for API requests."""
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _api_post_json(api_base_url: str, endpoint: str, payload: dict, api_key: Optional[str] = None, timeout: int = 180) -> dict:
    """POST JSON payload to ACE-Step API and return parsed wrapped response."""
    url = f"{api_base_url}{endpoint}"
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **_build_auth_headers(api_key)},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _api_get_bytes(api_base_url: str, endpoint: str, query: dict, api_key: Optional[str] = None, timeout: int = 180) -> bytes:
    """GET raw bytes from ACE-Step API endpoint with query parameters."""
    qs = urllib.parse.urlencode(query)
    url = f"{api_base_url}{endpoint}?{qs}"
    req = urllib.request.Request(
        url=url,
        headers=_build_auth_headers(api_key),
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _http_get_bytes(url: str, api_key: Optional[str] = None, timeout: int = 180) -> bytes:
    """GET raw bytes from a fully-qualified or resolved URL."""
    req = urllib.request.Request(
        url=url,
        headers=_build_auth_headers(api_key),
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _unwrap_api_response(response: dict) -> dict:
    """Unwrap ACE-Step standard response envelope and raise for API-level errors."""
    if not isinstance(response, dict):
        raise RuntimeError("Invalid API response format")
    code = response.get("code", 500)
    if code != 200:
        raise RuntimeError(response.get("error") or f"API call failed with code {code}")
    return response.get("data")


def _download_api_audio(api_base_url: str, api_key: Optional[str], source_path: str) -> str:
    """Resolve API-produced audio path to a local file path for Gradio playback."""
    if not source_path:
        raise RuntimeError("Empty audio path from API")

    source_path = str(source_path).strip()

    if source_path and os.path.exists(source_path):
        return source_path

    parsed_ref = urllib.parse.urlparse(source_path)
    raw_audio: bytes
    resolved_audio_path = source_path

    # Case 1: absolute URL returned by API (already points to /v1/audio)
    if parsed_ref.scheme in {"http", "https"}:
        raw_audio = _http_get_bytes(source_path, api_key=api_key)
        if parsed_ref.path.endswith("/v1/audio"):
            query_path = urllib.parse.parse_qs(parsed_ref.query).get("path", [resolved_audio_path])[0]
            resolved_audio_path = urllib.parse.unquote(query_path)

    # Case 2: relative API path like /v1/audio?path=%2Ftmp%2F...
    elif source_path.startswith("/v1/audio"):
        query_path = urllib.parse.parse_qs(parsed_ref.query).get("path", [""])[0]
        if query_path:
            resolved_audio_path = urllib.parse.unquote(query_path)
        audio_url = f"{api_base_url}{source_path}"
        raw_audio = _http_get_bytes(audio_url, api_key=api_key)

    # Case 3: plain filesystem path from API result
    else:
        raw_audio = _api_get_bytes(
            api_base_url=api_base_url,
            endpoint="/v1/audio",
            query={"path": source_path},
            api_key=api_key,
        )

    ext = os.path.splitext(resolved_audio_path)[1] or ".mp3"
    out_dir = os.path.join(tempfile.gettempdir(), "ace_step_ui_downloads")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"api_{int(time.time() * 1000)}_{random.randint(1000, 9999)}{ext}")
    with open(out_path, "wb") as f:
        f.write(raw_audio)
    return out_path


def load_random_example(task_type: str, api_base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Load a random example from API or local examples directory."""
    normalized_api_url = _normalize_api_base_url(api_base_url)
    if normalized_api_url:
        try:
            response = _api_post_json(
                api_base_url=normalized_api_url,
                endpoint="/create_random_sample",
                payload={"sample_type": "custom_mode"},
                api_key=api_key,
            )
            data = _unwrap_api_response(response) or {}

            bpm = data.get("bpm")
            duration = data.get("duration")
            keyscale = data.get("keyscale", data.get("key_scale", ""))
            timesignature = data.get("timesignature", data.get("time_signature", ""))

            gr.Info("Loaded random example from API")
            return (
                data.get("caption", data.get("prompt", "")),
                data.get("lyrics", ""),
                bpm,
                duration if duration is not None else -1,
                keyscale,
                timesignature,
            )
        except Exception as e:
            gr.Warning(f"API example load failed, falling back to local examples: {e}")

    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        examples_dir = os.path.join(project_root, "..", "examples", task_type)

        if not os.path.exists(examples_dir):
            return "", "", None, None, "", ""

        json_files = [f for f in os.listdir(examples_dir) if f.endswith(".json")]
        if not json_files:
            return "", "", None, None, "", ""

        selected_file = random.choice(json_files)

        with open(os.path.join(examples_dir, selected_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        caption = data.get("caption", data.get("prompt", ""))
        lyrics = data.get("lyrics", "")
        bpm = data.get("bpm")
        duration = data.get("duration")
        keyscale = data.get("keyscale", "")
        timesignature = data.get("timesignature", "")

        if isinstance(bpm, str):
            try:
                bpm = int(bpm) if bpm and bpm != "N/A" else None
            except:
                bpm = None

        if isinstance(duration, str):
            try:
                duration = float(duration) if duration and duration != "N/A" else -1
            except:
                duration = -1

        gr.Info(f"Loaded example: {os.path.basename(selected_file)}")
        return caption, lyrics, bpm, duration, keyscale, timesignature

    except Exception as e:
        gr.Warning(f"Failed to load example: {e}")
        return "", "", None, None, "", ""


def generate_caption_with_llm(llm_handler, task_type: str, api_base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Generate caption/lyrics using local LLM or API-backed random sample."""
    normalized_api_url = _normalize_api_base_url(api_base_url)
    if normalized_api_url:
        return load_random_example(task_type=task_type, api_base_url=normalized_api_url, api_key=api_key)

    if not llm_handler.llm_initialized:
        # Fall back to random example
        return load_random_example(task_type, api_base_url=api_base_url, api_key=api_key)

    try:
        result = understand_music(
            llm_handler=llm_handler,
            audio_codes="NO USER INPUT",
            temperature=0.85,
            use_constrained_decoding=True,
        )

        if result.success:
            gr.Info("Generated new caption using AI")
            return (
                result.caption,
                result.lyrics,
                result.bpm,
                result.duration if result.duration and result.duration > 0 else -1,
                result.keyscale,
                result.timesignature,
            )
        else:
            gr.Warning("AI caption generation failed, using example instead")
            return load_random_example(task_type, api_base_url=api_base_url, api_key=api_key)

    except Exception as e:
        gr.Warning(f"AI generation error: {e}")
        return load_random_example(task_type, api_base_url=api_base_url, api_key=api_key)


def format_lyrics_with_llm(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    duration,
    keyscale,
    timesig,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Format lyrics and caption using local LLM or remote API."""
    normalized_api_url = _normalize_api_base_url(api_base_url)
    if normalized_api_url:
        try:
            param_obj = {}
            if bpm is not None and str(bpm).strip():
                param_obj["bpm"] = int(bpm)
            if duration is not None and str(duration).strip() and float(duration) > 0:
                param_obj["duration"] = float(duration)
            if keyscale and keyscale.strip():
                param_obj["key_scale"] = keyscale.strip()
            if timesig and timesig.strip():
                param_obj["time_signature"] = timesig.strip()

            payload = {
                "prompt": caption,
                "lyrics": lyrics,
                "temperature": 0.85,
                "param_obj": param_obj,
            }
            response = _api_post_json(
                api_base_url=normalized_api_url,
                endpoint="/format_input",
                payload=payload,
                api_key=api_key,
            )
            data = _unwrap_api_response(response) or {}

            gr.Info("Formatted with API")
            return (
                data.get("caption", caption),
                data.get("lyrics", lyrics),
                data.get("bpm", bpm),
                data.get("duration", duration),
                data.get("key_scale", data.get("keyscale", keyscale)),
                data.get("time_signature", data.get("timesignature", timesig)),
            )
        except Exception as e:
            gr.Warning(f"API format error: {e}")
            return caption, lyrics, bpm, duration, keyscale, timesig

    if not llm_handler.llm_initialized:
        gr.Warning("LLM not initialized - cannot format")
        return caption, lyrics, bpm, duration, keyscale, timesig

    try:
        # Build user metadata
        user_metadata = {}
        if bpm is not None and bpm > 0:
            user_metadata["bpm"] = int(bpm)
        if duration is not None and float(duration) > 0:
            user_metadata["duration"] = int(float(duration))
        if keyscale and keyscale.strip():
            user_metadata["keyscale"] = keyscale.strip()
        if timesig and timesig.strip():
            user_metadata["timesignature"] = timesig.strip()

        result = format_sample(
            llm_handler=llm_handler,
            caption=caption,
            lyrics=lyrics,
            user_metadata=user_metadata if user_metadata else None,
            temperature=0.85,
            use_constrained_decoding=True,
        )

        if result.success:
            gr.Info("Formatted with AI")
            return (
                result.caption,
                result.lyrics,
                result.bpm,
                result.duration if result.duration and result.duration > 0 else -1,
                result.keyscale,
                result.timesignature,
            )
        else:
            gr.Warning("Formatting failed")
            return caption, lyrics, bpm, duration, keyscale, timesig

    except Exception as e:
        gr.Warning(f"Format error: {e}")
        return caption, lyrics, bpm, duration, keyscale, timesig


def handle_instrumental_toggle(is_instrumental: bool, current_lyrics: str):
    """Handle instrumental checkbox toggle"""
    if is_instrumental:
        return "[Instrumental]"
    elif current_lyrics == "[Instrumental]":
        return ""
    return current_lyrics


def run_generation(
    dit_handler,
    llm_handler,
    task_type: str,
    caption: str,
    lyrics: str,
    instrumental: bool,
    vocal_language: str,
    negative_prompt: str,
    batch_size: int,
    reference_audio,
    src_audio,
    repainting_start: float,
    repainting_end: float,
    inference_steps: int,
    guidance_scale: float,
    seed: str,
    duration: float,
    bpm,
    keyscale: str,
    timesignature: str,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    progress=gr.Progress(),
):
    """Run music generation locally or through remote API."""
    normalized_api_url = _normalize_api_base_url(api_base_url)
    if normalized_api_url:
        try:
            seed_val = int(seed) if seed.strip() != "-1" else random.randint(0, 2**32 - 1)
            payload = {
                "prompt": caption,
                "lyrics": lyrics,
                "thinking": False,
                "task_type": task_type,
                "vocal_language": vocal_language,
                "inference_steps": int(inference_steps),
                "guidance_scale": float(guidance_scale),
                "seed": seed_val,
                "use_random_seed": False,
                "batch_size": int(batch_size),
                "audio_duration": float(duration) if duration is not None and float(duration) > 0 else None,
                "bpm": int(bpm) if bpm is not None and str(bpm).strip() != "" else None,
                "key_scale": keyscale or "",
                "time_signature": timesignature or "",
                "lm_negative_prompt": negative_prompt.strip() if negative_prompt else "NO USER INPUT",
                "reference_audio_path": reference_audio if reference_audio else None,
                "src_audio_path": src_audio if src_audio else None,
                "repainting_start": repainting_start if task_type in ["repaint", "lego"] else 0.0,
                "repainting_end": repainting_end if task_type in ["repaint", "lego"] else None,
            }

            progress(0.1, desc="Submitting generation task to API...")
            submit_resp = _api_post_json(normalized_api_url, "/release_task", payload, api_key=api_key)
            submit_data = _unwrap_api_response(submit_resp) or {}
            task_id = submit_data.get("task_id")
            if not task_id:
                raise RuntimeError("API did not return task_id")

            poll_payload = {"task_id_list": json.dumps([task_id])}
            started_at = time.time()
            timeout_seconds = 3600

            while True:
                query_resp = _api_post_json(normalized_api_url, "/query_result", poll_payload, api_key=api_key)
                query_data = _unwrap_api_response(query_resp) or []
                if not query_data:
                    time.sleep(1.0)
                    continue

                first = query_data[0] or {}
                status = int(first.get("status", 0))
                progress_text = first.get("progress_text") or "Processing..."

                if status == 0:
                    progress(0.2, desc=progress_text)
                elif status == 2:
                    result_raw = first.get("result") or "[]"
                    err_msg = "Generation failed"
                    try:
                        parsed = json.loads(result_raw)
                        if parsed and isinstance(parsed, list):
                            err_msg = parsed[0].get("error") or err_msg
                    except Exception:
                        pass
                    raise RuntimeError(err_msg)
                elif status == 1:
                    result_raw = first.get("result") or "[]"
                    parsed = json.loads(result_raw)
                    outputs = []
                    for item in parsed:
                        file_path = item.get("file")
                        if file_path:
                            try:
                                outputs.append(_download_api_audio(normalized_api_url, api_key, file_path))
                            except Exception as download_error:
                                gr.Warning(f"Failed to fetch output audio: {download_error}")
                    progress(1.0, desc="Complete!")
                    return outputs

                if time.time() - started_at > timeout_seconds:
                    raise RuntimeError("Timed out waiting for generation result")
                time.sleep(1.0)
        except Exception as e:
            gr.Error(f"API generation failed: {str(e)}")
            return []

    if dit_handler.model is None:
        gr.Error("Model not initialized. Please initialize the model first.")
        return []

    try:
        # Parse seed
        seed_val = int(seed) if seed.strip() != "-1" else random.randint(0, 2**32 - 1)
        seeds = [seed_val + i for i in range(batch_size)]

        # Parse BPM
        bpm_val = int(bpm) if bpm is not None and bpm != "" else None

        # Determine if using turbo model
        is_turbo = dit_handler.is_turbo_model() if hasattr(dit_handler, "is_turbo_model") else True

        # Create generation params
        neg_prompt = negative_prompt.strip() if negative_prompt else ""
        params = GenerationParams(
            task_type=task_type,
            caption=caption,
            lyrics=lyrics,
            instrumental=instrumental,
            vocal_language=vocal_language,
            lm_negative_prompt=neg_prompt if neg_prompt else "NO USER INPUT",
            reference_audio=reference_audio if reference_audio else None,
            src_audio=src_audio if src_audio else None,
            repainting_start=repainting_start if task_type in ["repaint", "lego"] else 0.0,
            repainting_end=repainting_end if task_type in ["repaint", "lego"] else -1,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed_val,
            duration=float(duration) if duration is not None and duration > 0 else -1,
            bpm=bpm_val,
            keyscale=keyscale or "",
            timesignature=timesignature or "",
            shift=3.0 if is_turbo else 1.0,
        )

        config = GenerationConfig(
            batch_size=batch_size,
            seeds=seeds,
            use_random_seed=False,
            audio_format="mp3",
        )

        progress(0.1, desc="Starting generation...")

        # Create output directory for generated files
        output_dir = os.path.join(tempfile.gettempdir(), "ace_step_outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Run generation
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=output_dir,
        )

        if not result.success:
            gr.Error(f"Generation failed: {result.error}")
            return []

        progress(1.0, desc="Complete!")

        # Return audio outputs
        outputs = []
        for audio_dict in result.audios:
            audio_path = audio_dict.get("path")
            if audio_path and os.path.exists(audio_path):
                outputs.append(audio_path)

        return outputs

    except Exception as e:
        gr.Error(f"Error during generation: {str(e)}")
        import traceback

        traceback.print_exc()
        return []


def create_simple_ui(dit_handler: AceStepHandler, llm_handler: LLMHandler, init_params=None):
    """Create the simplified Gradio UI."""

    api_base_url = _normalize_api_base_url(init_params.get("api_base_url") if init_params else None)
    api_key = init_params.get("api_key") if init_params else None
    use_remote_api = bool(api_base_url)

    # Get GPU config
    gpu_config = init_params.get("gpu_config") if init_params else get_gpu_config()
    lm_initialized = init_params.get("init_llm", False) if init_params else False
    max_duration = gpu_config.max_duration_with_lm if lm_initialized else gpu_config.max_duration_without_lm

    # Determine if model is pre-initialized
    model_initialized = True if use_remote_api else dit_handler.model is not None
    available_models = dit_handler.get_available_acestep_v15_models() if not use_remote_api else []
    default_model = "acestep-v15-base" if "acestep-v15-base" in available_models else (available_models[0] if available_models else None)

    with gr.Blocks(title="ACE-Step Music Generator", css=CUSTOM_CSS) as demo:
        gr.HTML("""
        <div class="header">
            <h1>🎵 ACE-Step Music Generator</h1>
        </div>
        """)

        # Model Configuration Section
        with gr.Accordion("⚙️ Model Configuration", open=not model_initialized, visible=not use_remote_api):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=available_models, value=init_params.get("config_path", default_model) if init_params else default_model, label="Model", info="Select the generation model"
                )
                device_dropdown = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value=init_params.get("device", "auto") if init_params else "auto",
                    label="Device",
                )

            with gr.Row():
                lm_model_dropdown = gr.Dropdown(
                    choices=llm_handler.get_available_5hz_lm_models(),
                    value=init_params.get("lm_model_path", "acestep-5Hz-lm-1.7B") if init_params else "acestep-5Hz-lm-1.7B",
                    label="Language Model (for AI features)",
                    info="Required for AI caption generation",
                )
                init_llm_checkbox = gr.Checkbox(
                    label="Initialize Language Model",
                    info="Only needed if you want to auto-generate captions - uses extra VRAM",
                    value=True,
                    interactive=True,
                )

            with gr.Row():
                offload_checkbox = gr.Checkbox(
                    label="Offload to CPU (save VRAM)",
                    value=init_params.get("offload_to_cpu", False) if init_params else False,
                )
                flash_attn_checkbox = gr.Checkbox(
                    label="Use Flash Attention",
                    value=True,
                    interactive=dit_handler.is_flash_attention_available(),
                )

            init_btn = gr.Button("Initialize Model", variant="primary", size="lg")
            init_status = gr.Textbox(
                label="Status", value=init_params.get("init_status", "Model not initialized") if init_params else "Model not initialized", interactive=False, elem_classes=["status-box"]
            )

        # Main Generation Section
        gr.HTML("<div class='section-title'>🎼 Music Generation</div>")

        with gr.Group():
            # Task Type
            task_type = gr.Dropdown(choices=TASK_TYPES_TURBO, value="text2music", label="Generation Mode", info="text2music: Create from text | repaint: Edit audio | cover: Style transfer")

            # Audio uploads (for repaint/cover)
            with gr.Accordion("📁 Audio Uploads (for Repaint/Cover modes)", open=False) as audio_accordion:
                with gr.Row():
                    reference_audio = gr.Audio(
                        label="Reference Audio (for Cover)",
                        type="filepath",
                    )
                    source_audio = gr.Audio(
                        label="Source Audio (for Repaint)",
                        type="filepath",
                    )

                # Repainting controls
                with gr.Row(visible=False) as repaint_controls:
                    repainting_start = gr.Number(
                        label="Start Time (seconds)",
                        value=0.0,
                        step=0.5,
                    )
                    repainting_end = gr.Number(
                        label="End Time (seconds, -1 = end)",
                        value=-1,
                        step=0.5,
                    )

        # Caption Section
        gr.HTML("<div class='section-title'>📝 Music Description</div>")

        with gr.Group():
            # Prominent auto-generate button
            format_btn = gr.Button(
                "✨ Auto-generate caption and lyrics",
                variant="primary",
                size="lg",
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    caption_text = gr.TextArea(label="Caption", placeholder="Describe the music you want (genre, mood, instruments, tempo, etc.)", lines=3, info="Main description of the music")
                    lyrics_text = gr.TextArea(label="Lyrics", placeholder="Enter lyrics here, or check 'Instrumental' below", lines=6, info="Leave empty or check Instrumental for no vocals")
                with gr.Column(scale=1):
                    with gr.Group():
                        instrumental_cb = gr.Checkbox(
                            label="🎵 Instrumental",
                            value=False,
                            elem_classes=["prominent-checkbox"],
                        )
                        vocal_lang = gr.Dropdown(
                            choices=VALID_LANGUAGES,
                            value="unknown",
                            allow_custom_value=True,
                            label="Language",
                        )
                        sample_btn = gr.Button("🎲 Random example", variant="secondary", size="sm")

        # Parameters Section
        gr.HTML("<div class='section-title'>⚡ Generation Settings</div>")

        with gr.Group():
            with gr.Row():
                batch_slider = gr.Slider(minimum=1, maximum=3, value=1, step=1, label="Batch Size", info="Generate multiple variations at once")
                steps_slider = gr.Slider(minimum=4, maximum=200, value=50, step=1, label="Inference Steps", info="50 = default, 4-8 for turbo, up to 200 for higher quality")

            with gr.Row():
                duration_num = gr.Number(label="Duration (seconds)", value=-1, minimum=-1, maximum=float(max_duration), step=1, info=f"-1 = auto (recommended), max {max_duration}s")
                bpm_num = gr.Number(label="BPM", value=120, minimum=30, maximum=300, step=1, info="Leave empty for auto")

            with gr.Row():
                guidance_scale_slider = gr.Slider(
                    minimum=1.0, maximum=15.0, value=7.0, step=0.1, label="CFG Scale (Guidance)", info="7.0 = default, higher = follow prompt more strictly (base model only)"
                )

            with gr.Row():
                keyscale_text = gr.Textbox(label="Key/Scale", placeholder="e.g., C Major, Am", value="", info="Leave empty for auto")
                timesig_text = gr.Textbox(label="Time Signature", placeholder="e.g., 4", value="", info="4 = 4/4 time")

            with gr.Row():
                seed_text = gr.Textbox(label="Seed", value="-1", info="-1 for random, same seed = reproducible results")
                negative_prompt = gr.TextArea(
                    label="Negative Prompt (optional)", value="", placeholder="Things to avoid in generation (leave empty for default)", lines=1, info="What to exclude from the music"
                )

        # Generate Button
        generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg", elem_classes=["generate-btn"], interactive=model_initialized)

        # Output Section
        gr.HTML("<div class='section-title'>🎧 Generated Music</div>")

        # Output 1
        with gr.Group():
            output_audio_1 = gr.Audio(label="Output 1", type="filepath", interactive=False)
            download_link_1 = gr.File(label="Download 1", visible=False)

        # Output 2
        with gr.Group(visible=True):
            output_audio_2 = gr.Audio(label="Output 2", type="filepath", interactive=False)
            download_link_2 = gr.File(label="Download 2", visible=False)

        # Output 3
        with gr.Group(visible=True):
            output_audio_3 = gr.Audio(label="Output 3", type="filepath", interactive=False)
            download_link_3 = gr.File(label="Download 3", visible=False)

        # Status output
        gen_status = gr.Textbox(label="Generation Status", interactive=False, visible=True)

        # Event Handlers

        # Initialize model
        def init_model(config_path, device, lm_path, init_llm, offload, flash_attn):
            status_parts = []

            # Initialize DiT
            status, success = dit_handler.initialize_service(
                project_root=os.path.dirname(os.path.abspath(__file__)),
                config_path=config_path,
                device=device,
                use_flash_attention=flash_attn,
                offload_to_cpu=offload,
                offload_dit_to_cpu=offload,
            )
            status_parts.append(status)

            # Initialize LM if requested
            if init_llm and lm_path:
                checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
                lm_status, lm_success = llm_handler.initialize(
                    checkpoint_dir=checkpoint_dir,
                    lm_model_path=lm_path,
                    device=device,
                    offload_to_cpu=offload,
                    dtype=dit_handler.dtype,
                )
                status_parts.append(lm_status)

            return "\n".join(status_parts), gr.update(interactive=True)

        if not use_remote_api:
            init_btn.click(fn=init_model, inputs=[model_dropdown, device_dropdown, lm_model_dropdown, init_llm_checkbox, offload_checkbox, flash_attn_checkbox], outputs=[init_status, generate_btn])

        # Task type change - show/hide repaint controls
        def on_task_change(task):
            is_repaint = task in ["repaint", "lego"]
            return gr.update(visible=is_repaint)

        task_type.change(fn=on_task_change, inputs=[task_type], outputs=[repaint_controls])

        # Caption generation
        sample_btn.click(
            fn=lambda task: generate_caption_with_llm(llm_handler, task, api_base_url=api_base_url, api_key=api_key),
            inputs=[task_type],
            outputs=[caption_text, lyrics_text, bpm_num, duration_num, keyscale_text, timesig_text],
        )

        # Format button
        format_btn.click(
            fn=lambda cap, lyr, bpm, dur, key, ts: format_lyrics_with_llm(
                llm_handler,
                cap,
                lyr,
                bpm,
                dur,
                key,
                ts,
                api_base_url=api_base_url,
                api_key=api_key,
            ),
            inputs=[caption_text, lyrics_text, bpm_num, duration_num, keyscale_text, timesig_text],
            outputs=[caption_text, lyrics_text, bpm_num, duration_num, keyscale_text, timesig_text],
        )

        # Instrumental toggle
        instrumental_cb.change(fn=handle_instrumental_toggle, inputs=[instrumental_cb, lyrics_text], outputs=[lyrics_text])

        # Generate button
        def do_generate(*args):
            outputs = run_generation(dit_handler, llm_handler, *args, api_base_url=api_base_url, api_key=api_key)
            # Prepare audio outputs (3 audio players + 3 download links)
            audio_results = []
            download_results = []
            for i in range(3):
                if i < len(outputs) and outputs[i] and os.path.exists(outputs[i]):
                    audio_results.append(gr.update(value=outputs[i]))
                    download_results.append(gr.update(value=outputs[i], visible=True))
                else:
                    audio_results.append(gr.update(value=None))
                    download_results.append(gr.update(value=None, visible=False))
            return audio_results + download_results

        # Parameter order must match run_generation() function signature
        generate_btn.click(
            fn=do_generate,
            inputs=[
                task_type,
                caption_text,
                lyrics_text,
                instrumental_cb,
                vocal_lang,
                negative_prompt,
                batch_slider,
                reference_audio,
                source_audio,
                repainting_start,
                repainting_end,
                steps_slider,
                guidance_scale_slider,
                seed_text,
                duration_num,
                bpm_num,
                keyscale_text,
                timesig_text,
            ],
            outputs=[output_audio_1, output_audio_2, output_audio_3, download_link_1, download_link_2, download_link_3],
        )

        # Model type change - update task choices
        def on_model_change(model_name):
            if model_name and "turbo" in model_name.lower():
                return gr.update(choices=TASK_TYPES_TURBO)
            else:
                return gr.update(choices=TASK_TYPES_BASE)

        if not use_remote_api:
            model_dropdown.change(fn=on_model_change, inputs=[model_dropdown], outputs=[task_type])

    return demo


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ACE-Step Simple UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--init", action="store_true", help="Initialize on startup")
    parser.add_argument("--model", type=str, default="acestep-v15-base", help="Model to use")
    parser.add_argument("--lm-model", type=str, default=None, help="LM model to use")
    parser.add_argument("--root-path", type=str, default=None, help="Root path for serving behind reverse proxy (e.g., /acestep_demo)")
    parser.add_argument("--api-base-url", type=str, default=DEFAULT_API_BASE_URL, help="Use existing acestep-api server instead of local models (e.g., http://127.0.0.1:8001)")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key for acestep-api if enabled")

    args = parser.parse_args()

    api_base_url = _normalize_api_base_url(args.api_base_url)

    # Get GPU config
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)

    print(f"\n{'=' * 60}")
    print("ACE-Step Simple UI")
    print(f"{'=' * 60}")
    print(f"GPU Memory: {gpu_config.gpu_memory_gb:.2f} GB")
    print(f"Max Duration: {gpu_config.max_duration_with_lm}s")
    print(f"{'=' * 60}\n")

    # Create handlers
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # Prepare init params
    init_params = {
        "gpu_config": gpu_config,
        "init_llm": True,
        "offload_to_cpu": False,
        "use_flash_attention": dit_handler.is_flash_attention_available(),
        "api_base_url": api_base_url,
        "api_key": args.api_key,
    }

    # Auto-initialize if requested and not using remote API
    if args.init and not api_base_url:
        print("Initializing model...")
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

        status, success = dit_handler.initialize_service(
            project_root=checkpoint_dir,
            config_path=args.model,
            device="auto",
            use_flash_attention=init_params["use_flash_attention"],
            offload_to_cpu=init_params["offload_to_cpu"],
            offload_dit_to_cpu=init_params["offload_to_cpu"],
        )

        init_params["init_status"] = status
        init_params["config_path"] = args.model
        init_params["device"] = "auto"

        if success and args.lm_model:
            lm_status, lm_success = llm_handler.initialize(
                checkpoint_dir=os.path.join(checkpoint_dir, "checkpoints"),
                lm_model_path=args.lm_model,
                device="auto",
                offload_to_cpu=init_params["offload_to_cpu"],
                dtype=dit_handler.dtype,
            )
            init_params["init_status"] = status + "\n" + lm_status
            init_params["init_llm"] = lm_success
            init_params["lm_model_path"] = args.lm_model

        print(init_params["init_status"])

    if api_base_url:
        init_params["init_status"] = f"Using remote API at {api_base_url}"
        print(f"Using remote API mode: {api_base_url}")
        print("Local model initialization is disabled in this mode.")

    # Create and launch UI
    demo = create_simple_ui(dit_handler, llm_handler, init_params)

    demo.queue(max_size=10, default_concurrency_limit=1)

    print(f"\nLaunching UI at http://{args.server_name}:{args.port}")
    if args.root_path:
        print(f"Root path configured: {args.root_path}")
        print(f"Access URL will be: http://{args.server_name}:{args.port}{args.root_path}/")
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
        root_path=args.root_path,
    )


if __name__ == "__main__":
    main()
