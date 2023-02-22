from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

MODEL_PATH = r'C:\Data\dogs_neural.h5'
IMG_PATH = r'C:\Data\h.1.jpg'
class_names = ['Corgi', 'Husky', 'Shiba']

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(MODEL_PATH, compile=False)
model.summary()
model.get_layer(index=1).summary()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Open image
image = Image.open(IMG_PATH).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print(f"Class:{class_name}")
print(f"Confidence Score:{confidence_score}")
