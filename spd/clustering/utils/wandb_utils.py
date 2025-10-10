"""WandB utilities for clustering runs."""

import wandb_workspaces.workspaces as ws

from spd.log import logger

# Default template for clustering workspaces
# This template should be created in WandB with appropriate panels for clustering metrics
CLUSTERING_WORKSPACE_TEMPLATE = (
    "https://wandb.ai/goodfire/spd-cluster?nw=..."  # TODO: create template
)


def create_clustering_workspace_view(
    ensemble_hash: str,
    project: str = "spd-cluster",
    entity: str = "goodfire",
) -> str:
    """Create WandB workspace view for clustering runs.

    Args:
        ensemble_hash: Unique identifier for this ensemble
        project: WandB project name
        entity: WandB entity (team/user) name

    Returns:
        URL to workspace view
    """
    # For now, we'll create a simple workspace without a template
    # In the future, we can create a template workspace with appropriate panels
    # and reference it here (similar to the SPD workspace templates)

    # Create a basic workspace
    workspace = ws.Workspace(entity=entity, project=project)
    workspace.name = f"Clustering - {ensemble_hash}"

    # Filter for runs with this ensemble_hash
    # Runs will be tagged with wandb_group=f"ensemble-{ensemble_hash}"
    try:
        workspace.runset_settings.filters = [
            ws.Tags("group").isin([f"ensemble-{ensemble_hash}"]),
        ]
    except Exception as e:
        logger.warning(f"Could not set workspace filters: {e}")
        # Fallback: try using tags instead
        try:
            workspace.runset_settings.filters = [
                ws.Tags("tags").isin(["clustering"]),
            ]
        except Exception as e2:
            logger.warning(f"Could not set fallback workspace filters: {e2}")
            # Continue without filters

    try:
        workspace.save_as_new_view()
        return workspace.url
    except Exception as e:
        logger.error(f"Could not create workspace view: {e}")
        # Return a default URL to the project
        return f"https://wandb.ai/{project}"
