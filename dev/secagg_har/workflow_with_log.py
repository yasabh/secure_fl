"""secagg_har: A Flower with SecAgg+ app."""

from logging import INFO

from secagg_har.task import get_weights, make_net

import random
from flwr.common import Context, log
from flwr.common.secure_aggregation.quantization import quantize
from flwr.server import Grid, LegacyContext
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
    WorkflowState,
)


class SecAggPlusWorkflowWithLogs(SecAggPlusWorkflow):
    """The SecAggPlusWorkflow augmented for this example.

    This class includes additional logging and modifies one of the FitIns to instruct
    the target client to simulate a dropout.
    """

    def __init__(self, num_dropped: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_ids = []
        self.num_dropped = num_dropped

    def __call__(self, grid: Grid, context: Context) -> None:
        super().__call__(grid, context)

    def setup_stage(self, grid: Grid, context: LegacyContext, state: WorkflowState) -> bool:
        return super().setup_stage(grid, context, state)

    def collect_masked_vectors_stage(
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        self.simulate_dropped_clients(state)
        ret = super().collect_masked_vectors_stage(grid, context, state)
        log(INFO, "collect_masked_vectors_stage: Completed, Active nodes: %s", len(state.active_node_ids))

        for node_id in state.sampled_node_ids - state.active_node_ids:
            log(INFO, "Client %s dropped out", self.node_ids.index(node_id))

        return ret

    def unmask_stage(self, grid: Grid, context: LegacyContext, state: WorkflowState) -> bool:
        # Call parent's unmask_stage to remove the masks and obtain the aggregate.
        ret = super().unmask_stage(grid, context, state)
        log(INFO, "unmask_stage: Completed, Active nodes: %s", len(state.active_node_ids))
        return ret
    
    def share_keys_stage(self, grid: Grid, context: LegacyContext, state: WorkflowState) -> bool:
        # Call parent's share_keys_stage to execute the key-sharing logic.
        ret = super().share_keys_stage(grid, context, state)
        log(INFO, "share_keys_stage: Completed, Active nodes: %s", len(state.active_node_ids))
        return ret

    def simulate_dropped_clients(self, state):
        _num_dropped = self.num_dropped
        self.node_ids = list(state.active_node_ids)
        # Only drop nodes if the resulting active count remains above the protocol threshold.
        for node_id in self.node_ids.copy():
            # Check that removing one won't reduce the active count below the required threshold.
            if _num_dropped > 0 and (len(state.active_node_ids) - 1) >= state.threshold and random.choice([True, False]):
                client_index = self.node_ids.index(node_id)
                # state.nid_to_fitins[self.node_ids[client_index]]["fitins.config"]["index"] = client_index
                # state.nid_to_fitins[self.node_ids[client_index]]["fitins.config"]["drop"] = True
                state.active_node_ids.remove(node_id)
                log(INFO, "Client %s marked as dropped out", client_index)
                _num_dropped -= 1