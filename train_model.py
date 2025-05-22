import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # 2 classes: seedling and mature
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    classes = ['seedling', 'mature']  # Seedling and mature
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(class_idx)
    
    return np.array(images), np.array(labels)

def plot_training_history(history):
    # Create plot
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Create data directories
    os.makedirs('data/seedling', exist_ok=True)
    os.makedirs('data/mature', exist_ok=True)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data('data')
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_model()
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model.save('plant_growth_model.h5')
    print("Model training completed and saved!")
    print("Training history chart has been saved as 'training_history.png'")

if __name__ == "__main__":
    main() 