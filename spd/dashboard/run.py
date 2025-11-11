# %%
"""Minimal single-script version of causal importance decision tree training."""

from zanj import ZANJ

from spd.dashboard.core.acts import Activations
from spd.dashboard.core.compute import FlatActivations
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig
from spd.dashboard.core.save import DashboardData, IndexSummaries
from spd.dashboard.core.trees import DecisionTreesData


def main(config: ComponentDashboardConfig) -> None:
    # get activations
    activations: Activations = Activations.generate(config=config)
    flat_activations: FlatActivations = FlatActivations.create(
        activations, activation_threshold=config.activation_threshold
    )

    # train trees
    trees: DecisionTreesData = DecisionTreesData.create(
        flat_acts=flat_activations,
        config=config,
    )

    # summarize component info into lightweight index (also embeds)
    index_summaries: IndexSummaries = IndexSummaries.from_activations(
        activations=activations, config=config
    )

    # save
    dashboard_data: DashboardData = DashboardData(
        config=config,
        activations=activations,
        index_summaries=index_summaries,
        trees=trees,
    )
    dashboard_data.save(z=ZANJ(external_array_threshold=1))


def cli() -> None:
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run component dashboard generation with minimal CI DT example."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the dashboard configuration file.",
    )
    args: argparse.Namespace = parser.parse_args()

    # Load config
    config: ComponentDashboardConfig = ComponentDashboardConfig.from_file(path=args.config)

    # Run main function
    main(config)


if __name__ == "__main__":
    cli()