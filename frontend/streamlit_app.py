import streamlit as st
import requests
from PIL import Image

API_URL = "https://YOUR_RENDER_BACKEND_URL/analyze"

st.title("AI Skin Disease Detection")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    files = {"file": uploaded_file.getvalue()}

    response = requests.post(API_URL, files={"file": uploaded_file})

    if response.status_code == 200:

        data = response.json()

        st.subheader("Disease Predictions")

        st.json(data["predictions"])

    else:
        st.error("Backend error")