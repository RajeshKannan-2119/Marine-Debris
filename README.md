# AI-Driven Marine Debris Detection — Local Detection + Gemini Reports

Detection runs locally on your RTX 4050. Report generation uses the free
Gemini API — no cost for this project's usage level.

## 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## 2. Get a free Gemini API key

1. Go to https://aistudio.google.com/apikey and sign in with your Google account.
2. Click "Create API key" — takes about 2 minutes, no billing setup needed for
   the free tier.
3. Set it as an environment variable (don't paste it directly into your code):

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your-key-here"

# Mac/Linux
export GEMINI_API_KEY="your-key-here"
```

You'll need to set this again each time you open a new terminal, unless you
add it to your shell profile (`.bashrc`, `.zshrc`, or a PowerShell profile).

Note: your free student Gemini Advanced subscription (the chat app) is a
separate product from this API key — the API has its own free tier, which is
what this project uses.

## 3. Get a dataset

Go to Roboflow Universe (universe.roboflow.com) and search "TrashCan" or
"marine debris". Pick a project, export it in **YOLOv8** format, and unzip it
into a `dataset/` folder here. It should contain `data.yaml`, `train/`, `valid/`.

TACO (tacodataset.org) and AquaTrash are good supplementary sets if you want
more images — merge them into your Roboflow project before exporting.

## 4. Train the detector

```bash
python train_yolo.py --data dataset/data.yaml --epochs 50
```

- Uses `yolov8n.pt` by default (fits easily in 6GB VRAM). Add `--batch 8` if
  you hit a CUDA out-of-memory error.
- Takes roughly 1–3 hours depending on dataset size.
- Note the printed mAP/precision/recall when it finishes — use these in your
  report and slides.
- Trained weights land at `runs/marine_debris/weights/best.pt`.

## 5. Try single-image detection + report from the command line

```bash
python detect_and_report.py --weights runs/marine_debris/weights/best.pt --image path/to/test.jpg
```

## 6. Run the full web app

```bash
streamlit run app.py
```

Opens in your browser. Upload an image, click Analyze, see the detected
debris and the Gemini-generated pollution report.

## Notes for your report/slides

- Detection model: YOLOv8 (nano/small), fine-tuned via transfer learning on
  [name your dataset here], trained locally on an RTX 4050.
- Insight generation: Gemini 2.5 Flash (Google's Gemini API, free tier),
  prompted with the structured detection output.
- Includes retry/backoff so a rate-limited API call waits and retries
  automatically instead of failing.
