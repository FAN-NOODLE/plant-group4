import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import sys

# ✅ 1. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 加载类别名
with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ✅ 3. 加载模型结构与权重
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best model/bestone_0.2698.pth", map_location=device))
model.to(device)
model.eval()

# ✅ 4. 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ✅ 5. 预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    predicted_class = class_names[predicted.item()]
    print(f"✅ Prediction: {predicted_class} (Confidence: {confidence.item():.2f})")
    return predicted_class, confidence.item()

# ✅ 6. 支持命令行运行
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)
