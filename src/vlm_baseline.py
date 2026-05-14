"""
Vision-Language Model (VLM) Baseline for FracAtlas Bone Fracture Classification
===============================================================================

This script evaluates a zero-shot multimodal LLM / vision-language model on the
same held-out FracAtlas test split used by the CNN experiments.

Research-only purpose:
    This script is for benchmarking general-purpose VLMs against trained CNN
    transfer-learning models. It is NOT a clinical diagnostic tool and must not
    be used for medical advice or patient care.

Expected input CSV columns:
    image_id      -> image filename, for example IMG0001234.jpg
    fractured     -> binary label, 1 = fracture, 0 = no fracture

Output CSV columns:
    filename, true_label, prediction, fracture_probability, short_reason,
    provider, model, prompt_mode, raw_response, error, image_path

Example:
    python src/vlm_baseline.py \
        --provider openai \
        --model gpt-4o-mini \
        --test_csv data/FracAtlas/test.csv \
        --image_dir data/FracAtlas/images \
        --prompt_mode simple \
        --limit 20 \
        --output_csv outputs/vlm/openai_simple_debug.csv

Author: Bone Fracture ML Detection project
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# -----------------------------------------------------------------------------
# Prompt design
# -----------------------------------------------------------------------------

BASE_INSTRUCTION = """You are participating in a research-only benchmark on public musculoskeletal X-ray images.
This is NOT for clinical diagnosis, medical advice, or patient care.

Task:
Classify the image as binary fracture detection:
- 1 = fracture-positive
- 0 = no-fracture

Return ONLY valid JSON with exactly these keys:
{
  "prediction": 0 or 1,
  "fracture_probability": number between 0 and 1,
  "short_reason": "one short sentence"
}

Important:
- fracture_probability should mean the probability/confidence that the image is fracture-positive.
- Do not return markdown.
- Do not return extra text outside JSON.
"""

PROMPT_MODES: Dict[str, str] = {
    "simple": BASE_INSTRUCTION + "\nUse your best visual judgment from the X-ray image.",
    "conservative": BASE_INSTRUCTION
    + "\nBe conservative. Choose fracture-positive only when there is clear visual evidence. If uncertain, choose no-fracture.",
    "sensitive": BASE_INSTRUCTION
    + "\nPrioritize sensitivity. If there is plausible visual evidence of fracture, choose fracture-positive, even if this may increase false positives.",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

@dataclass
class VLMResult:
    prediction: Optional[int]
    fracture_probability: Optional[float]
    short_reason: str
    raw_response: str
    error: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    """Build filename -> path index for recursive image lookup."""
    index: Dict[str, Path] = {}
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            # If duplicate basenames exist, keep the lexicographically first path
            # for deterministic behavior.
            key = p.name
            if key not in index or str(p) < str(index[key]):
                index[key] = p
    return index


def resolve_image_path(image_dir: Path, filename: str, image_index: Dict[str, Path]) -> Path:
    """Resolve an image filename either directly or recursively."""
    filename = str(filename)
    direct = image_dir / filename
    if direct.exists():
        return direct

    # Some CSVs may store relative paths.
    rel = image_dir / filename.replace("\\", "/")
    if rel.exists():
        return rel

    base = Path(filename).name
    if base in image_index:
        return image_index[base]

    raise FileNotFoundError(f"Could not find image '{filename}' under {image_dir}")


def encode_image_for_api(
    image_path: Path,
    max_side: int = 1024,
    send_format: str = "jpeg",
    jpeg_quality: int = 90,
) -> Tuple[bytes, str, str]:
    """
    Read and optionally resize an image for VLM API submission.

    Returns:
        image_bytes, mime_type, data_url
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    fmt = send_format.lower()
    if fmt in {"jpg", "jpeg"}:
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        mime_type = "image/jpeg"
    elif fmt == "png":
        img.save(buffer, format="PNG", optimize=True)
        mime_type = "image/png"
    else:
        raise ValueError("send_format must be 'jpeg' or 'png'")

    image_bytes = buffer.getvalue()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    return image_bytes, mime_type, data_url


def extract_json_object(text: str) -> Dict:
    """Robustly extract one JSON object from a VLM response."""
    if text is None:
        raise ValueError("Empty response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def normalize_vlm_response(raw_text: str) -> VLMResult:
    """Parse and validate the JSON output returned by the VLM."""
    try:
        obj = extract_json_object(raw_text)
        pred = int(obj.get("prediction"))
        if pred not in (0, 1):
            raise ValueError(f"prediction must be 0 or 1, got {pred}")

        prob = float(obj.get("fracture_probability"))
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"fracture_probability must be in [0, 1], got {prob}")

        reason = str(obj.get("short_reason", "")).strip()
        return VLMResult(
            prediction=pred,
            fracture_probability=prob,
            short_reason=reason,
            raw_response=raw_text,
            error="",
        )
    except Exception as exc:
        return VLMResult(
            prediction=None,
            fracture_probability=None,
            short_reason="",
            raw_response=raw_text or "",
            error=f"parse_error: {exc}",
        )


def default_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
        "anthropic": "claude-sonnet-4-20250514",
        "mock": "mock-vlm",
    }
    return defaults[provider]


# -----------------------------------------------------------------------------
# Provider adapters
# -----------------------------------------------------------------------------

def call_openai(
    prompt: str,
    data_url: str,
    model: str,
    detail: str = "low",
    temperature: float = 0.0,
) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=250,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
    )
    return response.output_text


def call_gemini(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=250,
        ),
    )
    return response.text or ""


def call_anthropic(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = client.messages.create(
        model=model,
        max_tokens=250,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


def call_mock(filename: str, true_label: int) -> str:
    """
    Deterministic fake provider for testing the pipeline without API calls.

    This intentionally uses true_label to generate plausible fake outputs; never
    report mock results as real benchmark results.
    """
    h = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    r = int(h[:8], 16) / 0xFFFFFFFF
    # Roughly 75% correct, deterministic by filename.
    correct = r < 0.75
    pred = true_label if correct else 1 - true_label
    prob = 0.75 + (r % 0.2) if pred == 1 else 0.05 + (r % 0.2)
    prob = max(0.0, min(1.0, prob))
    return json.dumps(
        {
            "prediction": int(pred),
            "fracture_probability": round(float(prob), 4),
            "short_reason": "Mock output for pipeline testing only.",
        }
    )


def call_provider(
    provider: str,
    prompt: str,
    image_path: Path,
    filename: str,
    true_label: int,
    model: str,
    max_side: int,
    send_format: str,
    jpeg_quality: int,
    openai_detail: str,
    temperature: float,
) -> str:
    if provider == "mock":
        return call_mock(filename, true_label)

    image_bytes, mime_type, data_url = encode_image_for_api(
        image_path=image_path,
        max_side=max_side,
        send_format=send_format,
        jpeg_quality=jpeg_quality,
    )

    if provider == "openai":
        return call_openai(prompt, data_url, model, detail=openai_detail, temperature=temperature)
    if provider == "gemini":
        return call_gemini(prompt, image_bytes, mime_type, model, temperature=temperature)
    if provider == "anthropic":
        return call_anthropic(prompt, image_bytes, mime_type, model, temperature=temperature)

    raise ValueError(f"Unknown provider: {provider}")


# -----------------------------------------------------------------------------
# Main benchmark loop
# -----------------------------------------------------------------------------

def load_completed_keys(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()
    try:
        df = pd.read_csv(output_csv)
        if "filename" not in df.columns:
            return set()
        return set(df["filename"].astype(str).tolist())
    except Exception:
        return set()


def append_row(output_csv: Path, fieldnames: List[str], row: Dict) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    provider = args.provider
    model = args.model or default_model_for_provider(provider)
    prompt = PROMPT_MODES[args.prompt_mode]

    test_csv = Path(args.test_csv)
    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)

    df = pd.read_csv(test_csv)
    if args.filename_column not in df.columns:
        raise ValueError(f"Missing filename column: {args.filename_column}. Available: {list(df.columns)}")
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}. Available: {list(df.columns)}")

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit)

    print("=" * 72)
    print("  VLM BASELINE — FRACTURE CLASSIFICATION")
    print("=" * 72)
    print(f"  Provider      : {provider}")
    print(f"  Model         : {model}")
    print(f"  Prompt mode   : {args.prompt_mode}")
    print(f"  Test CSV      : {test_csv}")
    print(f"  Image dir     : {image_dir}")
    print(f"  Samples       : {len(df)}")
    print(f"  Output CSV    : {output_csv}")
    print("  Research-only : not clinical diagnosis")
    print("=" * 72)

    print("Building recursive image index...")
    image_index = build_image_index(image_dir)
    print(f"Indexed {len(image_index)} image files.")

    completed = load_completed_keys(output_csv) if args.resume else set()
    if completed:
        print(f"Resume mode: {len(completed)} filenames already present in output CSV.")

    fieldnames = [
        "timestamp_utc",
        "provider",
        "model",
        "prompt_mode",
        "filename",
        "image_path",
        "true_label",
        "prediction",
        "fracture_probability",
        "correct",
        "short_reason",
        "raw_response",
        "error",
    ]

    n_done = 0
    n_errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="VLM inference"):
        filename = str(row[args.filename_column])
        true_label = int(row[args.label_column])

        if args.resume and filename in completed:
            continue

        try:
            image_path = resolve_image_path(image_dir, filename, image_index)
        except Exception as exc:
            result = VLMResult(None, None, "", "", f"image_error: {exc}")
            image_path = Path("")
        else:
            result = None
            for attempt in range(args.max_retries + 1):
                try:
                    raw_text = call_provider(
                        provider=provider,
                        prompt=prompt,
                        image_path=image_path,
                        filename=filename,
                        true_label=true_label,
                        model=model,
                        max_side=args.max_side,
                        send_format=args.send_format,
                        jpeg_quality=args.jpeg_quality,
                        openai_detail=args.openai_detail,
                        temperature=args.temperature,
                    )
                    result = normalize_vlm_response(raw_text)
                    break
                except Exception as exc:
                    if attempt < args.max_retries:
                        wait = args.retry_sleep * (attempt + 1)
                        print(f"\nRetry {attempt + 1}/{args.max_retries} for {filename}: {exc}. Sleeping {wait:.1f}s")
                        time.sleep(wait)
                    else:
                        result = VLMResult(None, None, "", "", f"api_error: {exc}")

        if result is None:
            result = VLMResult(None, None, "", "", "unknown_error")

        pred = result.prediction
        correct = "" if pred is None else int(pred == true_label)
        if result.error:
            n_errors += 1

        append_row(
            output_csv,
            fieldnames,
            {
                "timestamp_utc": now_iso(),
                "provider": provider,
                "model": model,
                "prompt_mode": args.prompt_mode,
                "filename": filename,
                "image_path": str(image_path),
                "true_label": true_label,
                "prediction": "" if pred is None else pred,
                "fracture_probability": "" if result.fracture_probability is None else round(result.fracture_probability, 6),
                "correct": correct,
                "short_reason": result.short_reason,
                "raw_response": result.raw_response,
                "error": result.error,
            },
        )
        n_done += 1

        if args.sleep_sec > 0 and provider != "mock":
            time.sleep(args.sleep_sec)

    print("\nDone.")
    print(f"Rows written this run: {n_done}")
    print(f"Errors this run      : {n_errors}")
    print(f"Output CSV           : {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a zero-shot VLM baseline on FracAtlas test images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", choices=["openai", "gemini", "anthropic", "mock"], required=True)
    parser.add_argument("--model", default=None, help="Provider model name. Defaults depend on provider.")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--filename_column", default="image_id")
    parser.add_argument("--label_column", default="fractured")
    parser.add_argument("--prompt_mode", choices=sorted(PROMPT_MODES.keys()), default="simple")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for debugging. Omit for full test set.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip filenames already present in output CSV.")

    parser.add_argument("--max_side", type=int, default=1024, help="Resize longest image edge before API call. Use 0 to disable.")
    parser.add_argument("--send_format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--openai_detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--sleep_sec", type=float, default=0.5, help="Delay between API calls to reduce rate-limit errors.")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
