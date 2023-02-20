from tensorflow import keras
import numpy as np

MODEL_PATH = ''
IMG_PATH = ''
class_names = ['Corgi', 'Huski', 'Shibu']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

model = keras.models.load_model(MODEL_PATH)
result = model.predict(IMG_PATH)
pred_label = np.argmax(result, axis= 1)

print (class_names_label[pred_label])