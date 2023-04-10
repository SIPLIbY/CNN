import numpy as np
import random
from tensorflow import keras as k
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

def printing_image(single_test):
    plt.figure()
    plt.imshow(single_test[0], cmap="gray")
    plt.show()

(x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = k.utils.to_categorical(y_train, 10)
y_test_cat = k.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)



model = k.Sequential([
    k.layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)),
    k.layers.MaxPooling2D((2,2), strides=2),
    k.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    k.layers.MaxPooling2D((2,2), strides=2),
    k.layers.Flatten(),
    k.layers.Dense(128, activation='softmax'),
    k.layers.Dense(10, activation='softmax')
])


model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics=['accuracy'])

his = model.fit(x_train, y_train_cat, batch_size=64, epochs=10, validation_split=0.2)

model.evaluate(x_test, y_test_cat)


single_test = []
single_test.append(x_test[56])
np_single_test = np.array(single_test)
printing_image(np_single_test)


predicted = model.predict(np_single_test)
print('Predicted class is: ', np.argmax(predicted[0]))





