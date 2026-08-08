import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import os
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fruit Ripeness Detector", page_icon="🍎", layout="centered")

# ---------------- MODEL PATH----------------

MODEL_PATH = "fruit_model.keras"

# ---------------- DOWNLOAD MODEL FROM GOOGLE DRIVE ----------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1Wlie4NcWNAW48px094QSzRJ92tdCFk0v"

        session = requests.Session()
        response = session.get(url, stream=True)
        with open (MODEL_PATH, "wb" ) as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)


download_model()

# CALL FUNCTION (VERY IMPORTANT)
download_model()

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH, compile=False, safe_mode=False)

classes = ['overripe', 'ripe', 'unripe']

# ---------------- BACKGROUND IMAGE ----------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("bg.jpg")   # 👈 bg.jpg same folder में होना चाहिए

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
}}

/* overlay */
.stApp::before {{
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.6);
    z-index: -1;
}}

.title {{
    text-align: center;
    font-size: 40px;
    color: #2c3e50;
    font-weight: bold;
}}

.subtitle {{
    text-align: center;
    font-size: 18px;
    color: #555;
}}

.result-box {{
    padding: 15px;
    border-radius: 10px;
    background-color: #ecf9f1;
    text-align: center;
    font-size: 20px;
    margin-top: 20px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🍎 AI Fruit Ripeness Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or capture a fruit image to detect ripeness</div>', unsafe_allow_html=True)

st.write("")

# ---------------- INPUT OPTION ----------------
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

if option == "Upload Image":
    file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])
else:
    file = st.camera_input("📸 Capture Image")

# ---------------- PREDICTION ----------------
if file is not None:
    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### 🔍 Prediction")

        if st.button("🚀 Predict"):
            img = image.resize((100, 100))
            img = np.array(img) / 255.0
            img = img.reshape(1, 100, 100, 3)

            prediction = model.predict(img)
            index = np.argmax(prediction)
            confidence = round(np.max(prediction) * 100, 2)

            if classes[index] == "ripe":
                st.success("🍏 Ripe (Ready to eat!)")
            elif classes[index] == "unripe":
                st.warning("🟡 Unripe (Not ready yet)")
            else:
                st.error("🔴 Overripe (Too ripe)")

            st.markdown(f"""
                <div class="result-box">
                    📊 <b>Confidence:</b> {confidence}%
                </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("---")
st.caption("Made with ❤️ using Streamlit")
