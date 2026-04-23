import os
import urllib.request

# folder where models will be saved
MODEL_DIR = "backend/models"

# your Google Drive links (paste here)
MODELS = {
    "acne_pigmentation_model.pth": "https://drive.google.com/uc?id=1pWfWlZexDMPQizbiNDwdnjPgu-jVmPyg",
    "redness_model.pth": "https://drive.google.com/uc?id=1Xiyd7EQWPZn_pXPyLPhAqSnFDpYRrf_Z",
    "red_eye_model.pth": "https://drive.google.com/uc?id=1ULjwj5rXD1D2QHUAYUdr62-HY_dxATL2"
}

# create folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# download models
for name, url in MODELS.items():

    path = os.path.join(MODEL_DIR, name)

    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"{name} downloaded")
    else:
        print(f"{name} already exists")