"""secagg_har: A Flower with SecAgg+ app."""

import torch

from logging import DEBUG, INFO, WARNING
from typing import List, Tuple

from secagg_har.task import get_weights, set_weights, make_net, load_data, test
from secagg_har.workflow_with_log import SecAggPlusWorkflowWithLogs

from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays, log
from flwr.common.logger import update_console_handler
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from attacks import get_byz
from mpc import MPC

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import data_loaders

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
    
    def __init__(self, net, num_rounds: int, mpc_enabled: bool, mpc_setup: dict, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initial_parameters = kwargs.get("initial_parameters")
        self.seed = seed
        self.latest_parameters = None
        self.net = net
        self.num_rounds = num_rounds
        self.mpc_enabled = mpc_enabled
        self.mpc_setup = mpc_setup
        self.accuracy_history = []
        self.loss_history = []
        self.net.to(self.device)

        if self.mpc_enabled:
            self.mpc_setup.compile()
            self.mpc_setup.run(self.device)
    
    def aggregate_fit(self, rnd, results, failures):
        if self.mpc_enabled:
            if self.latest_parameters is not None:
                set_weights(self.net, parameters_to_ndarrays(self.latest_parameters))
            elif self._initial_parameters is not None:
                set_weights(self.net, parameters_to_ndarrays(self._initial_parameters))
            else:
                raise ValueError("No initial parameters available for aggregation.")

            gradients = extract_gradients(results)

            num_expected_clients = self.mpc_setup.num_clients
            num_active_clients = len(results)
            num_missing_clients = num_expected_clients - num_active_clients

            # Pad missing gradients with dummy (zero) gradients
            if num_missing_clients > 0:
                log(INFO, f"Padding with {num_missing_clients} dummy gradients")
                # Assume all clients' gradients have the same structure as gradients[0]
                dummy_gradient = [torch.zeros_like(tensor) for tensor in gradients[0]]
                for _ in range(num_missing_clients):
                    gradients.append(dummy_gradient)

            # Call your mpspdz_aggregation function to securely aggregate updates
            # This function should update the model 'net' in place.
            with torch.no_grad():
                self.mpc_setup.aggregate(gradients=gradients)
            
            # Scale the aggregated update by the ratio of expected to active clients
            scaling = num_expected_clients / num_active_clients
            new_weights = [w * scaling for w in get_weights(self.net)]

            self.latest_parameters = ndarrays_to_parameters(new_weights)

            # Convert new_weights and initial parameters to tensors
            new_w = [w if isinstance(w, torch.Tensor) else torch.from_numpy(w).to(self.device)
                    for w in new_weights]
            init_w = [torch.from_numpy(p).to(self.device)
                    for p in parameters_to_ndarrays(self._initial_parameters)]
            if all(torch.equal(a, b) for a, b in zip(new_w, init_w)):
                log(WARNING, "MPC aggregation may have failed or returned unchanged parameters.")

            return self.latest_parameters, {}
        else:
            aggregated_result = super().aggregate_fit(rnd, results, failures)
            if aggregated_result is not None:
                parameters, _ = aggregated_result
                self.latest_parameters = parameters
            return aggregated_result

    def aggregate_evaluate(self, rnd: int, results, failures):
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
        log(INFO, f"")
        log(INFO, f"[GLOBAL EVALUATION]")

        if self.latest_parameters is None:
            log(INFO, "No aggregated parameters available.")
            return
        
        if self.mpc_enabled:
            self.mpc_setup.wait()

        # Fresh model
        net_geval = make_net()
        
        set_weights(net_geval, parameters_to_ndarrays(self.latest_parameters))

        net_geval.to(self.device)
        net_geval.eval()
        
        train_loader, test_loader = data_loaders.load_data(seed=self.seed)
        global_loss, global_accuracy = test(net_geval, test_loader, self.device)
        log(INFO, f"Test Loss: {global_loss:.4f}, Test Accuracy: {global_accuracy:.4f}")

# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_clients = len(grid.get_node_ids())
    
    seed = context.run_config["seed"]
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Get initial parameters
    net = make_net(seed=seed)

    mpc_setup = MPC(
        net=net,
        protocol=context.run_config["protocol"],
        aggregation="fedavg",
        num_parties=context.run_config["num-parties"],
        port=context.run_config["port"],
        niter=context.run_config["num-server-rounds"],
        learning_rate=context.run_config["learning-rate"],
        chunk_size=context.run_config["chunk-size"],
        num_clients=num_clients,
        byz=get_byz(context.run_config["byz"]),
        num_byz=context.run_config["num-byz"],
        threads=4,
        parallels=1,
        always_compile=False,
    )

    # Define strategy using our LoggingFedAvg
    strategy = LoggingFedAvg(
        net=net,
        num_rounds=num_rounds,
        mpc_enabled=context.run_config["mpc-enabled"],
        mpc_setup=mpc_setup,
        seed=seed,
        fraction_fit=1.0,
        min_fit_clients=2,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=aggregate_train_acc,
        initial_parameters=ndarrays_to_parameters(get_weights(net)),
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
        num_dropped=context.run_config.get("num-dropped", 0),
        # timeout=context.run_config["timeout"],
    )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute workflow
    workflow(grid, context)
