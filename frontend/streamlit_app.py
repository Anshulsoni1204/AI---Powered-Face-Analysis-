import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

# 🔗 Backend API (change after deployment)
API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="AI Skin Analysis", layout="centered")

st.title("🧠 AI Skin Disease Detection")
st.write("Upload a face image to analyze skin conditions.")

# 📤 Upload image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # 📸 Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🔘 Analyze button
    if st.button("Analyze"):

        with st.spinner("Analyzing..."):

            try:
                # ✅ Correct file format for FastAPI
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()

                    # ✅ Handle backend error safely
                    if "error" in data:
                        st.error(f"❌ {data['error']}")

                    else:
                        # 📊 Show predictions
                        st.subheader("📊 Disease Predictions")

                        for region, values in data["predictions"].items():
                            st.markdown(f"### 🔹 {region.upper()}")

                            for condition, result in values.items():
                                st.write(
                                    f"**{condition}** → {result['label']} "
                                    f"({result['confidence']:.2f})"
                                )

                        # 🖼️ Show annotated image
                        if "annotated_image" in data:
                            image_bytes = base64.b64decode(data["annotated_image"])
                            processed_image = Image.open(BytesIO(image_bytes))

                            st.subheader("🖼️ Processed Image")
                            st.image(processed_image, use_column_width=True)

                else:
                    st.error(f"Backend error: {response.status_code}")
                    st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure FastAPI is running.")

            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")