from tensorflow import keras
import numpy as np
from keras.utils import load_img, img_to_array
import tensorflow as tf
import cv2


#from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.resnet import preprocess_input, decode_predictions

MODEL_PATH = r'C:\bak\Data\new\dogs_neural.h5'
IMG_PATH = r'C:\bak\Data\new\c7.jpg'
IMAGE_SIZE = (150, 150)
class_names = ['Corgi', 'Husky', 'Shiba']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

images = []
image = cv2.imread(IMG_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, IMAGE_SIZE)
images.append(image)

images = np.array(images, dtype='float32')

# img = load_img(IMG_PATH, target_size=(150, 150))
# img_array = img_to_array(img)
# img_batch = np.expand_dims(img_array, axis=0)
# img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_batch)

model = keras.models.load_model(MODEL_PATH)
prediction = model.predict(images)

# decoded = tf.keras.applications.resnet50.decode_predictions(prediction, top=3)
#
# print(tf.keras.applications.resnet50.decode_predictions(prediction, top=3)[0])

pred_label = np.argmax(prediction, axis=1)
print(f"label {pred_label[0]}")
print(class_names[pred_label[0]])

# prediction = np.argmax(prediction)
#
# print("Номер класса:", prediction)
# print("Название класса:", class_names[prediction])