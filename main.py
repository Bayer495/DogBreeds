import zipfile
import os
import cv2
import keras.initializers.initializers
import numpy as np
from keras.applications import InceptionResNetV2
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras import optimizers

DIRECTORY = r"C:\bak\Data\New_folder"
CATEGORY = ["seg_train", "seg_test"]
IMAGE_SIZE = (224, 224)
class_names = ['Corgi', 'Husky', 'Shiba']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

print (class_names_label)


def load_image_data() :
    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        images = []
        labels = []

        print("Loading {}".format((category)))

        for folder in os.listdir(path):
            label = class_names_label[folder]

            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(os.path.join(path, folder), file)

                # open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')

        output.append((images, labels))

    return output

def plot_accuracy_loss(history):
    fig = plt.figure(figsize=(10,5))

    # plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label= "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label= "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'], 'bo--', label="loss")
    plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()

def init_network_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])


    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

(train_images, train_labels), (test_images, test_labels) = load_image_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

# init network
model = init_network_model()

# fit network
history = model.fit(train_images,
                    train_labels,
                    epochs=30,
                    validation_split=0.2)

# evaluate network efficiency
test_loss = model.evaluate(test_images, test_labels)

# vector of probabilities
predictions = model.predict(test_images)
# we take the highest probability
pred_labels = np.argmax(predictions, axis=1)
print(classification_report(test_labels, pred_labels))
plot_accuracy_loss(history)

model.save(os.path.join(DIRECTORY, "dogs_neural.h5"))
