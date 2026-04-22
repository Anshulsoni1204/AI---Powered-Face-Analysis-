import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_m


def build_model(num_classes=2):

    model = efficientnet_v2_m(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def safe_load(model, path):

    checkpoint = torch.load(path, map_location="cpu")

    model_dict = model.state_dict()

    # keep only matching layers
    filtered = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(filtered)

    model.load_state_dict(model_dict)

    model.eval()

    return model


def load_models():

    acne_model = safe_load(
        build_model(),
        "models/acne_pigmentation_model.pth"
    )

    redness_model = safe_load(
        build_model(),
        "models/redness_model.pth"
    )

    eye_model = safe_load(
        build_model(),
        "models/red_eye_model.pth"
    )

    return {
        "acne": acne_model,
        "redness": redness_model,
        "eye": eye_model
    }