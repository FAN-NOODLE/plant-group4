import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ✅ Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ✅ Paths
data_dir = './PlantVillage'
checkpoint_path = './checkpoints/resnet18-f37072fd.pth'
os.makedirs("checkpoints", exist_ok=True)

# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ✅ Load dataset
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ Load model structure and manually load weights
model = models.resnet18()
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ✅ Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training settings
num_epochs = 30
save_every = 5
early_stop_patience = 7
best_val_loss = float('inf')
epochs_no_improve = 0

# ✅ Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"\n📘 Epoch {epoch + 1}/{num_epochs}")

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # ✅ Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)

    print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ✅ Auto-save model
    if (epoch + 1) % save_every == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        print(f"💾 Model saved as model_epoch_{epoch + 1}.pth")

    # ✅ Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'loss_0.4838.pth')
        print(f"🌟 Best model saved (Val Loss: {val_loss:.4f})")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement in val loss ({epochs_no_improve}/{early_stop_patience})")

    if epochs_no_improve >= early_stop_patience:
        print("⛔ Early stopping: no improvement in validation loss.")
        break

# ✅ Evaluate on test set
print("\n🔍 Evaluating on test set...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ✅ Print classification report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ✅ Confusion matrix visualization
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()



