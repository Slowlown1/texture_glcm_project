import os
import cv2
import numpy as np
from feature_extraction import extract_glcm_features
from classifier import train_model, predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from classifier import train_model, evaluate_model
dataset_path = "../dataset"
X = []
y = []
for label in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, label)
    if not os.path.isdir(class_folder):
        continue
    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = extract_glcm_features(gray)
        X.append(features)
        y.append(label)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = train_model(X_train, y_train)
y_pred = predict(model, X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
model = train_model(X_train, y_train)
y_pred = evaluate_model(model, X_test, y_test)