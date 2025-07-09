import sys
sys.path.insert(0, "sahi")
import streamlit as st
import platform
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import pandas as pd
from roboflow import Roboflow
from sahi.model import RemoteInferenceModel
from sahi.predict import get_sliced_prediction

st.set_page_config(page_title="SAHI + Roboflow Detector", layout="centered")
st.title("🧠 Roboflow Object Detection with SAHI")
st.sidebar.text(f"Python version: {platform.python_version()}")

st.sidebar.header("Settings")
version = st.sidebar.selectbox("Model Version", options=list(range(12, 0, -1)), index=0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

MODEL_NAME = "flower-counter"
ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(MODEL_NAME)
model = project.version(version).model

sahi_model = RemoteInferenceModel(
    model_type="yolov8",
    endpoint_url=model.url,
    confidence_threshold=confidence_threshold,
    device="cpu"
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running sliced inference..."):
        try:
            prediction_result = get_sliced_prediction(
                image,
                detection_model=sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            font = ImageFont.load_default()
            label_counts = {}

            for obj in prediction_result.object_prediction_list:
                box = obj.bbox.to_xyxy()
                cls = obj.category.name
                score = obj.score.value
                label = f"{cls} ({score:.2f})"

                draw.rectangle(box, outline="red", width=4)
                draw.text((box[0], box[1] - 10), label, fill="red", font=font)
                label_counts[cls] = label_counts.get(cls, 0) + 1

            st.image(annotated, caption="Detections (SAHI)", use_container_width=True)

            if label_counts:
                st.subheader("📊 Detection Summary")
                df_summary = pd.DataFrame(label_counts.items(), columns=["Class", "Count"])
                st.table(df_summary)

            out_buf = BytesIO()
            annotated.save(out_buf, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=out_buf.getvalue(),
                file_name="detections.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Detection failed: {e}")
