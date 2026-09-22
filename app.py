"""
Streamlit web platform: upload an image, see debris detections, read an
AI-generated environmental report. Fully local, no API keys.

USAGE:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from detect_and_report import build_prompt, generate_report
from collections import Counter

st.set_page_config(page_title="AI-Driven Marine Debris Detection", layout="centered")

st.title("🌊 AI-Driven Marine Debris Detection")
st.caption("Upload an underwater or coastal image to detect debris and generate an environmental report.")

WEIGHTS_PATH = "runs/detect/runs/marine_debris-2/weights/best.pt"


@st.cache_resource
def load_model():
    return YOLO(WEIGHTS_PATH)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Detecting debris..."):
            model = load_model()
            results = model.predict(source=image, conf=0.25, save=False)
            result = results[0]
            class_names = result.names
            detected = [class_names[int(box.cls)] for box in result.boxes]
            counts = Counter(detected)

            annotated = result.plot()  # numpy array, BGR
            st.image(annotated[:, :, ::-1], caption="Detected debris", use_container_width=True)

        if counts:
            st.subheader("Detections")
            st.json({k: v for k, v in counts.items()})
        else:
            st.info("No debris detected above the confidence threshold.")

        with st.spinner("Generating environmental report with Gemini..."):
            report = generate_report(counts)

        st.subheader("AI-Generated Environmental Report")
        st.markdown(report)
else:
    st.info("Upload an image to get started.")
