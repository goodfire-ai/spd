# %%
"""Minimal single-script version of causal importance decision tree training."""

from spd.dashboard.core.acts import Activations
from spd.dashboard.core.compute import FlatActivations
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig
from spd.dashboard.core.save import DashboardData
from spd.dashboard.core.trees import DecisionTreesData

# %% ----------------------- Configuration -----------------------

CONFIG = ComponentDashboardConfig(
    model_path="wandb:goodfire/spd/runs/lxs77xye",
    batch_size=4,
    n_batches=4,
    context_length=16,
)

# %% ----------------------- get activations -----------------------

ACTIVATIONS: Activations = Activations.generate(config=CONFIG)
FLAT_ACTIVATIONS: FlatActivations = FlatActivations.create(ACTIVATIONS)


# %% ----------------------- Train Decision Trees -----------------------

TREES: DecisionTreesData = DecisionTreesData.create(
    flat_acts=FLAT_ACTIVATIONS,
    config=CONFIG,
)


# %% ----------------------- save -----------------------

DASHBOARD_DATA: DashboardData = DashboardData(
    config=CONFIG,
    activations=ACTIVATIONS,
    trees=TREES,
    metadata={"description": "Minimal CI DT example"},
)

DASHBOARD_DATA.save()
