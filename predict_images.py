# predict_images.py
import os
from PIL import Image
import torch
from torchvision import transforms
from src.model import create_model

# -------------------------------
# 1. Config
# -------------------------------
model_path = "models/vgg16_best.pth"  # path to your downloaded model
images_folder = "test_images"  # folder with images to predict
class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
]  # your classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. Load model
# -------------------------------
model = create_model("vgg16").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded and ready for inference!")

# -------------------------------
# 3. Define preprocessing
# -------------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# -------------------------------
# 4. Predict on images in folder
# -------------------------------
for img_file in os.listdir(images_folder):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(images_folder, img_file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_class = torch.max(outputs, 1)

    print(f"{img_file} --> Predicted class: {class_names[pred_class.item()]}")
