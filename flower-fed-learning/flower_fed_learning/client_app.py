"""flower-fed-learning: A Flower / TensorFlow app."""
import numpy as np
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flower_fed_learning.task import load_data, load_model


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        x_train = np.asarray(self.x_train, dtype=np.float32)
        y_train = np.asarray(self.y_train)
        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # Ensure that x_test and y_test are proper NumPy arrays
        x_test = np.asarray(self.x_test, dtype=np.float32)
        y_test = np.asarray(self.y_test)

        loss = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        return loss[0], len(x_test), {}

def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
