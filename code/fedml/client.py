import tensorflow as tf
from tensorflow.keras import layers

import flwr as fl

from dataset import load_data

(x_train, y_train), (x_test, y_test) = load_data()

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(8,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)  # binary_crossentropy questionable


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=10)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
fl.client.start_client(
    server_address="127.0.0.1:8080", client=FlowerClient().to_client()
)
