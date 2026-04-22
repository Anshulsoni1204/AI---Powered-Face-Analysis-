import os
import torch
import torch.nn as nn

# create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

class SkinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(16,3)
        )

    def forward(self,x):
        return self.model(x)

model = SkinModel()

torch.save(model.state_dict(), "models/acne_pigmentation_model.pth")
torch.save(model.state_dict(), "models/redness_model.pth")
torch.save(model.state_dict(), "models/red_eyes_model.pth")

print("Dummy models created successfully!")