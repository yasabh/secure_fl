"""secagg_har: A Flower with SecAgg+ app."""

import time

import torch
from secagg_har.task import Net, get_weights, load_data, set_weights, test, train

from logging import WARNING
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, log


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        # self.timeout = timeout

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        loss, accuracy = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
        )

        # if config.get("drop", False):
        #     log(WARNING, f"Client {config.get("index")} dropped for testing purposes.")
        #     time.sleep(self.timeout + 1)

        return get_weights(self.net), self.trainloader['data'].shape[0], {"accuracy": accuracy}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Read run_config to fetch hyperparameters relevant to this run
    trainloader, valloader = load_data(
        partition_id=context.node_config["partition-id"],
        num_partitions=context.node_config["num-partitions"],
        seed=context.run_config["seed"],
        bias=context.run_config["bias"],
        server_pc=context.run_config["server-pc"],
        p=context.run_config["p"],
        batch_size=context.run_config["batch-size"],
    )

    # Return Client instance
    return FlowerClient(
        trainloader=trainloader,
        valloader=valloader,
        local_epochs= context.run_config["local-epochs"],
        learning_rate=context.run_config["learning-rate"],
        # timeout=context.run_config["timeout"],
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)
