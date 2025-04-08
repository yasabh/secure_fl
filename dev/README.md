---
tags: [advanced, secure_aggregation, semi2k, mpc, privacy]
dataset: [HAR (Human Activity Recognition)]
framework: [Flower]
---

# Secure aggregation with Flower (SecAgg+ + Semi2k MPC using HAR Dataset)

This repository provides a detailed example of using **Flower's SecAgg+ protocol** integrated with the **Semi2k MPC** protocol to enhance privacy guarantees during federated learning using the **HAR (Human Activity Recognition)** dataset. It reflects the current state of an advanced FL system based on our research and implementation.

The following steps describe how to use Flower's built-in Secure Aggregation components. This example demonstrates how to apply `SecAgg+` to the same federated learning workload as in the [quickstart-pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example. The `ServerApp` uses the [`SecAggPlusWorkflow`](https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html#secaggplusworkflow) while `ClientApp` uses the [`secaggplus_mod`](https://flower.ai/docs/framework/ref-api/flwr.client.mod.secaggplus_mod.html#flwr.client.mod.secaggplus_mod). To introduce the various steps involved in `SecAgg+`, this example introduces as a sub-class of `SecAggPlusWorkflow` the `SecAggPlusWorkflowWithLogs`. It is enabled by default, but you can disable it (see later in this readme).

## Project Motivation and Scope
This project demonstrates how to improve data privacy in federated learning by integrating Flower's SecAgg+ protocol with Semi2k MPC while using the HAR dataset. The goal is to protect individual client updates and securely handle client dropouts during training. If some clients drop out, we pad their missing gradients with zeros and then scale the aggregate update to adjust for the missing data, ensuring the global model remains accurate.

## Practical Notes
- Dropouts and Scaling:
When fewer clients contribute than expected, the missing gradients are replaced with zeros. Since averaging with zeros dilutes the update, we scale the aggregate by the ratio of expected to active clients. This adjustment prevents training accuracy from stagnating due to dropouts.

- *Computation Cost*
Secure aggregation with MPC is computationally expensive. For running simulations effectively, it's recommended to use a dedicated VM or a GPU cluster.

- *Logging*
Log the number of masked/unmasked vectors and monitor the count of active versus dropped clients each round to diagnose any impact on model performance.

## Key Features

- *SecAgg+ Masking*
Clients mask their updates using additive secret sharing, where random masks are distributed among peers. These masks hide individual updates, and only when enough shares are combined (meeting the reconstruction threshold) can the missing mask be reconstructed.

- *Dropout Tolerance*
If a client drops out, its missing mask is reconstructed from other clients’ secret shares (provided the number of shares meets or exceeds the threshold). This allows secure, uninterrupted aggregation even when not all clients participate.

- *Secure MPC Aggregation (Semi2k)*
After unmasking, Semi2k MPC securely aggregates the model updates. This extra layer ensures that the computation of the global model is performed without exposing any intermediate values, reinforcing the privacy guarantees.

---

## Set up the project

### 1. Clone the project
```bash
git clone https://github.com/yasabh/secure_fl.git && cd secure_fl/dev
```

### 2. Install dependencies
```bash
pip install -e .
```

## Running the Project
You can run the system in Simulation Mode for development or locally distributed mode for testing across multiple processes.
```bash
flwr run .
```
You may override parameters:
```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.25 batch-size=128"
```
`pyproject.toml` run config example:
```toml
[tool.flwr.app.config]
num-clients = 30
num-server-rounds = 5
fraction-evaluate = 0.3
local-epochs = 1
learning-rate = 0.25
batch-size = 128
seed = 1
bias = 0.5
p = 0.1
server-pc = 100

# SecAgg+
num-shares = 3
reconstruction-threshold = 2
max-weight = 9000
num-dropped = 2
timeout = 120

# MPC settings
mpc-enabled = false
protocol = "semi2k"
byz = "label_flipping_attack"
num-byz = 0
port = 14000
chunk-size = 500
num-parties = 3

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 30
```

## Directory Structure
```plaintext
dev/
├── datasets/                      # Custom or preprocessed datasets (e.g., HAR)
├── mpspdz/                        # MP-SPDZ framework directory for MPC computations
├── secagg_cifar10/                # CIFAR-10 based version of the same framework (NON-MPC ready)
│   └── ...                        # Similar structure to secagg_har but for CIFAR-10
├── secagg_har/                    # HAR-based secure federated learning implementation
│   ├── __init__.py                # Marks this as a Python package
│   ├── client_app.py              # Flower ClientApp with SecAgg+ mod integration
│   ├── server_app.py              # Flower ServerApp using SecAggPlusWorkflow + MPC
│   ├── data_loaders.py            # HAR dataset loading, preprocessing, partitioning
│   ├── task.py                    # Model definition, train/test logic
│   ├── workflow_with_log.py       # Custom workflow logging and dropout simulation
├── attacks.py                     # Adversarial strategies
├── Dockerfile                     # Docker container setup for reproducibility
├── mpc_install.sh                 # Shell script to setup MP-SPDZ
├── mpc.py                         # Orchestrates MPC setup, client communication, aggregation
├── pyproject.toml                 # Flower config, dependencies, runtime settings
└── README.md                      # Complete guide and documentation (this file)
```

## References
- [`Flower Documentation`](https://flower.ai/docs/)
- [`MP-SPDZ MPC framework`](https://github.com/data61/MP-SPDZ)
- [`SecAgg+ Paper`](https://arxiv.org/pdf/2205.06117) (Bonawitz et al., 2022)