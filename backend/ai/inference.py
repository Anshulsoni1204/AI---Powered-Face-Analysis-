import torch
import cv2
import numpy as np

ACNE_CLASSES = ["acne", "pigmentation"]
REDNESS_CLASSES = ["normal", "redness"]
EYE_CLASSES = ["normal", "red_eye"]


def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0)


def predict(model, tensor, classes):

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

        conf, pred = torch.max(probs, dim=1)

        return {
            "label": classes[pred.item()],
            "confidence": float(conf.item())
        }


def run_inference(models, crops):

    results = {}

    for region, img in crops.items():

        tensor = preprocess(img)

        results[region] = {
            "acne_pigmentation": predict(models["acne"], tensor, ACNE_CLASSES),
            "redness": predict(models["redness"], tensor, REDNESS_CLASSES),
            "red_eye": predict(models["eye"], tensor, EYE_CLASSES)
        }

    return results