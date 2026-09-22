"""
Fine-tune a pretrained YOLOv8 model on a marine-debris dataset.

BEFORE RUNNING:
1. Download a YOLO-format marine debris dataset (e.g. TrashCan / TACO / AquaTrash
   exported from Roboflow Universe) and unzip it somewhere like ./dataset/
2. That folder should contain a data.yaml file plus train/ and valid/ subfolders
   with images/ and labels/. Roboflow exports already look like this - if yours
   doesn't, open data.yaml and fix the train/val paths to match your folders.
3. pip install -r requirements.txt

USAGE:
    python train_yolo.py --data dataset/data.yaml --epochs 50
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                         help="Path to the dataset's data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Base checkpoint. yolov8n.pt (fastest) or yolov8s.pt "
                              "(a bit more accurate) both fit comfortably in 6GB VRAM.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                         help="Drop to 8 if you hit a CUDA out-of-memory error.")
    args = parser.parse_args()

    # Loads the pretrained checkpoint (auto-downloads the first time you run this).
    model = YOLO(args.model)

    # device=0 uses your RTX 4050. Ultralytics auto-detects CUDA if it's available.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project="runs",
        name="marine_debris",
    )

    # Runs validation and prints mAP/precision/recall - use these numbers in your report.
    metrics = model.val()
    print("\n--- Validation metrics (use these in your report/slides) ---")
    print(metrics)

    print("\nBest weights saved to: runs/marine_debris/weights/best.pt")
    print("Use that path as --weights in detect_and_report.py")


if __name__ == "__main__":
    main()
