import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fruit Ripeness Detector", page_icon="🍎", layout="centered")

# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 👉 Make sure bg.jpg same folder me ho
img = get_base64("bg.jpg")

# ---------------- APPLY BACKGROUND IMAGE ----------------
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* Light overlay for better text visibility */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.6);
    z-index: -1;
}}

/* Title styling */
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

# ---------------- LOAD MODEL ----------------
model = load_model("models/fruit_model.h5", compile=False)

classes = ['overripe', 'ripe', 'unripe']

# ---------------- TITLE ----------------
st.markdown('<div class="title">🍎 AI Fruit Ripeness Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or capture a fruit image to detect ripeness</div>', unsafe_allow_html=True)

st.write("")

# ---------------- INPUT OPTION ----------------
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

# ---------------- IMAGE INPUT ----------------
if option == "Upload Image":
    file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])
else:
    file = st.camera_input("📸 Capture Image")

# ---------------- PROCESS ----------------
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

            # Result display
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
