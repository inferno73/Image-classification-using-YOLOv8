import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO('./runs/classify/train30/weights/last.pt')

# Predict with the model, local ABSOLUTE path
results = model(r"C:\Users\Korisnik\Documents\Coding\CourseCVEmaterial\weather-dataset\train\cloudy\cloudy5.jpg")  # predict on an image

names_dict = results[0].names
probs = results[0].probs.tolist()  #probabilities

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])