import tensorflow as tf

class CNN:
    def __init__(self, batch_size=128, epochs=8, data_path="../data/spec_ann"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_path = data_path

        self.model = None

    def build_model(self):
        self.model = tf.keras.models.Sequential()

        # TODO fix input layer
        self.model.add(tf.keras.layers.Input(shape=(128, 128, 1), batch_size=self.batch_size))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))

        # TODO add hidden layers

        # TODO add neurons and activations, this is the output layer
        self.model.add(tf.keras.layers.Dense())
        self.model.add(tf.keras.layers.Reshape())
        self.model.add(tf.keras.layers.Activation())

        # TODO implement model compilation (optimizer, loss, etc.)
        self.model.compile()