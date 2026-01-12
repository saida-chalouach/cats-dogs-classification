import sys
import numpy as np
from PIL import Image
from tensorflow import keras

MODEL_PATH = 'cat_dog_model.h5'
IMAGE_SIZE = 150

def predict(image_path):
    model = keras.models.load_model(MODEL_PATH)
    
    img = Image.open(image_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        print(f"Chien 🐶 - Confiance: {prediction*100:.2f}%")
    else:
        print(f"Chat 🐱 - Confiance: {(1-prediction)*100:.2f}%")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image>")
    else:
        predict(sys.argv[1])