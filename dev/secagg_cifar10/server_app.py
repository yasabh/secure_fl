"""secagg_cifar10: A Flower with SecAgg+ app."""

import torch

from logging import DEBUG, INFO
from typing import List, Tuple

from secagg_cifar10.task import get_weights, set_weights, make_net, load_data, test
from secagg_cifar10.workflow_with_log import SecAggPlusWorkflowWithLogs

from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.common.logger import update_console_handler
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from attacks import get_byz
from mpc import MPC

# Define metric aggregation function for accuracy (modify if needed)
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}

def aggregate_train_acc(metrics_list):
    total = sum(n for n, _ in metrics_list)
    avg = sum(n * m["accuracy"] for n, m in metrics_list) / total
    return {"accuracy": avg}

def extract_gradients(results):
    gradients = []
    for _, fit_res in results:
        # Convert fit_res.parameters (Flower Parameters) into a list of torch tensors.
        # You might need to convert using parameters_to_ndarrays, then torch.tensor()
        ndarrays = parameters_to_ndarrays(fit_res.parameters)
        tensor_list = [torch.tensor(arr) for arr in ndarrays]
        gradients.append(tensor_list)
    return gradients

class LoggingFedAvg(FedAvg):
    
    def __init__(self, net, num_rounds: int, mpc_setup: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initial_parameters = kwargs.get("initial_parameters")
        self.latest_parameters = None
        self.net = net
        self.mpc_setup = mpc_setup
        self.accuracy_history = []
        self.loss_history = []
        self.net.to(self.device)
        # self.mpc_setup.compile()
        # self.mpc_setup.run(self.device)
    
    def aggregate_fit(self, rnd, results, failures):
        if self.latest_parameters is not None:
            set_weights(self.net, parameters_to_ndarrays(self.latest_parameters))
        elif self._initial_parameters is not None:
            set_weights(self.net, parameters_to_ndarrays(self._initial_parameters))
        else:
            raise ValueError("No initial parameters available for aggregation.")
        
        # Call your mpspdz_aggregation function to securely aggregate updates
        # This function should update the model 'net' in place.
        self.mpc_setup.aggregate(gradients=extract_gradients(results))
        
        # Extract the new aggregated parameters from the updated net
        self.latest_parameters = ndarrays_to_parameters(get_weights(net))
        return self.latest_parameters, {}

    # def aggregate_fit(self, rnd, results, failures):

    #     # # Log client fit metrics (if any)
    #     # print("Client fit metrics:")
    #     # for _, res in results:
    #     #     print(res.metrics)

    #     aggregated_result = super().aggregate_fit(rnd, results, failures)
    #     if aggregated_result is not None:
    #         parameters, _ = aggregated_result
    #         self.latest_parameters = parameters
    #     return aggregated_result

    def aggregate_evaluate(self, rnd: int, results, failures):

        # # Log evaluation metrics from clients
        # print("Client evaluation metrics:")
        # for _, res in results:
        #     print(res.metrics)

        aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_metrics:
            # Expect aggregated_metrics is a tuple (loss, metrics)
            loss, metrics = aggregated_metrics
            # We assume metrics contains both accuracy and loss
            if "accuracy" in metrics:
                self.accuracy_history.append((rnd, metrics["accuracy"]))
            if "loss" in metrics:
                self.loss_history.append((rnd, metrics["loss"]))
            log(INFO, f"results: Loss {loss:.4f}, Metrics {metrics}")
        else:
            log(INFO, f"results: None")

        if rnd == self.num_rounds:
            self.run_centralized_evaluation()

        return aggregated_metrics

    def run_centralized_evaluation(self):
        # self.mpc_setup.wait()

        log(INFO, f"")
        log(INFO, f"[GLOBAL EVALUATION]")
        if self.latest_parameters is None:
            log(INFO, "No aggregated parameters available.")
            return
        
        set_weights(self.net, parameters_to_ndarrays(self.latest_parameters))
        
        # Load global CIFAR-10 test data (modify parameters as needed)
        # Here, we use partition_id=0 and num_partitions=1 to load the entire test set
        _, test_loader = load_data(partition_id=0, num_partitions=1, batch_size=32)
        
        # Set device
        self.net.eval()
        
        # Compute test loss and accuracy using your provided test function
        global_loss, global_accuracy = test(self.net, test_loader, self.device)
        
        # Print the computed global loss and accuracy
        log(INFO, f"Test Loss: {global_loss:.4f}, Test Accuracy: {global_accuracy:.4f}")

# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Get initial parameters
    net = make_net()

    mpc_setup = MPC(
        net=net,
        protocol=context.run_config["protocol"],
        aggregation="fedavg",
        num_parties=context.run_config["num-parties"],
        port=context.run_config["port"],
        niter=context.run_config["num-server-rounds"],
        learning_rate=context.run_config["learning-rate"],
        chunk_size=context.run_config["chunk-size"],
        nworkers=context.run_config["num-clients"],
        byz=get_byz(context.run_config["byz"]),
        num_byz=context.run_config["num-byz"],
        threads=4,
        parallels=4,
        always_compile=False
    )

    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy using our LoggingFedAvg
    strategy = LoggingFedAvg(
        net=net,
        num_rounds=num_rounds,
        mpc_setup=mpc_setup,
        fraction_fit=1.0,
        min_fit_clients=2,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=aggregate_train_acc,
        initial_parameters=parameters,
    )

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Create fit workflow
    # For further information, please see:
    # https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html
    
    # update_console_handler(DEBUG, True, True)
    fit_workflow = SecAggPlusWorkflowWithLogs(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        max_weight=context.run_config["max-weight"],
        num_dropped=context.run_config.get("num-dropped", 0)
    )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute workflow
    workflow(grid, context)
