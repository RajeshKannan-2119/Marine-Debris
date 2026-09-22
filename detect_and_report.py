"""
Run the trained YOLO model on an image, then ask Gemini (free API tier) to turn
the detections into a plain-language pollution report.

BEFORE RUNNING:
1. Get a free API key from https://aistudio.google.com/apikey (2 minutes, no
   billing needed for the free tier).
2. Set it as an environment variable so it's never hardcoded in your code:
       Windows (PowerShell):  $env:GEMINI_API_KEY="your-key-here"
       Mac/Linux:              export GEMINI_API_KEY="your-key-here"
   (Do this every time you open a new terminal, or add it to your shell profile.)
3. pip install -r requirements.txt

USAGE:
    python detect_and_report.py --weights runs/marine_debris/weights/best.pt --image path/to/photo.jpg
"""

import argparse
import os
import time
from collections import Counter

from google import genai
from google.genai import errors as genai_errors
from ultralytics import YOLO

GEMINI_MODEL = "gemini-3.6-flash"  # fast + free-tier friendly


def run_detection(weights_path, image_path, conf_threshold=0.35):
    model = YOLO(weights_path)
    results = model.predict(source=image_path, conf=conf_threshold, save=True)
    result = results[0]

    class_names = result.names
    detected_classes = [class_names[int(box.cls)] for box in result.boxes]
    counts = Counter(detected_classes)

    print(f"Annotated image saved under: {result.save_dir}")
    return counts


def build_prompt(counts):
    if not counts:
        return (
            "An underwater image was analyzed and no debris was detected. "
            "Write two short sentences noting the area appears clean, and one "
            "sentence suggesting continued periodic monitoring."
        )

    detected_list = ", ".join(f"{count}x {name}" for name, count in counts.items())
    return f"""You are an environmental monitoring assistant. An AI vision model
scanned an underwater/coastal image and detected the following objects: {detected_list}.

Write a short report with three parts:
1. Pollution Analysis - one paragraph summarizing what was found.
2. Environmental Impact - one paragraph on the likely impact of these specific
   items on marine life or the ecosystem.
3. Cleanup Recommendation - two to three concrete, actionable bullet points.

Keep the whole report under 200 words. Plain language, no headers other than
the three bolded labels above."""


def generate_report(counts, max_retries=3):
    """Calls Gemini with simple retry/backoff in case the free tier's
    per-minute rate limit is hit - waits and tries again rather than crashing."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. See the instructions "
            "at the top of this file."
        )

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(counts)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except genai_errors.ClientError as e:
            # Rate limit (429) or transient server error - back off and retry.
            if attempt < max_retries - 1:
                wait_seconds = 15 * (attempt + 1)
                print(f"Gemini call failed ({e}). Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
            else:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/marine_debris/weights/best.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    counts = run_detection(args.weights, args.image, args.conf)
    print("\nDetections:", dict(counts) if counts else "none")

    print("\nGenerating report with Gemini...")
    report = generate_report(counts)

    print("\n--- AI-Generated Environmental Report ---\n")
    print(report)


if __name__ == "__main__":
    main()
