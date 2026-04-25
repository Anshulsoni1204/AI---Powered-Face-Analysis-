import download_models
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware 
from ai.model_loader import load_models
from ai.face_processor import process_face_image
from ai.inference import run_inference

app = FastAPI()
# ✅ ADD CORS HERE (right after app creation)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (good for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# load models once
models = load_models()


@app.get("/")
def home():
    return {"message": "Skin Analysis API running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    image_bytes = await file.read()
    if not image_bytes:
        return {"error": "Empty image received"}

    data, error = process_face_image(image_bytes)

    if error:
        return {"error": error}

    crops = data["crops"]

    predictions = run_inference(models, crops)

    return {
        "predictions": predictions,
        "annotated_image": data["annotated_image"],
        "face_regions": data["encoded_crops"]
    }
