import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import datetime

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0
y = np.array(y)



model1 = keras.Sequential(
    [
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)

model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

log_dir = "logs/fit/" + "Circles-vs-Squares-vs-Triangles-cnn-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# model.summary()
model1.fit(X, y, epochs=50, batch_size = 32, validation_split=0.3, callbacks=[tensorboard_callback])
