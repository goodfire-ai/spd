import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.app.backend.api import TokenLayerCosineSimilarityData
from spd.app.backend.services.run_context_service import RunContextService
from spd.log import logger
from spd.utils.distributed_utils import get_device

DEVICE = get_device()


class GeometryService:
    def __init__(self, run_context_service: RunContextService):
        self.run_context_service = run_context_service

    def get_subcomponent_cosine_sims(
        self, layer: str, component_idx: int
    ) -> TokenLayerCosineSimilarityData:
        assert (run := self.run_context_service.train_run_context) is not None, (
            "Run context not found"
        )
        assert (components := run.cm.components.get(layer)) is not None, f"Layer {layer} not found"

        logger.info(f"component index: {component_idx}")

        assert (cluster_ctx := self.run_context_service.cluster_run_context) is not None
        layer_components = cluster_ctx.clustering_shape.module_component_groups[layer]

        component_subcomponent_indices = torch.tensor(
            layer_components[component_idx], device=DEVICE, dtype=torch.long
        )

        assert component_subcomponent_indices.ndim == 1, "Nonzero indices must be 1D"
        n_nonzero = component_subcomponent_indices.shape[0]

        # =========================================================#
        # Where C is the number of subcomponents in the component #
        # =========================================================#

        u_singular_vectors: Float[torch.Tensor, "C d_in"] = components.U[
            component_subcomponent_indices
        ]
        logger.info(f"{u_singular_vectors.shape=}")
        u_pairwise_cosine_similarities = pairwise_cosine_similarities(u_singular_vectors)
        assert u_pairwise_cosine_similarities.shape == (n_nonzero, n_nonzero)

        v_singular_vectors: Float[torch.Tensor, "d_out C"] = components.V[
            :, component_subcomponent_indices
        ]
        logger.info(f"{v_singular_vectors.T.shape=}")
        v_pairwise_cosine_similarities = pairwise_cosine_similarities(v_singular_vectors.T)
        logger.info(f"{v_pairwise_cosine_similarities.shape=}")
        assert v_pairwise_cosine_similarities.shape == (n_nonzero, n_nonzero)

        # Zero out diagonal for display (self-similarity is always 1 and distracting)
        u_pairwise_cosine_similarities.fill_diagonal_(0.0)
        v_pairwise_cosine_similarities.fill_diagonal_(0.0)

        logger.info(f"U pairwise cosine similarities: {u_pairwise_cosine_similarities.shape}")
        logger.info(f"V pairwise cosine similarities: {v_pairwise_cosine_similarities.shape}")

        return TokenLayerCosineSimilarityData(
            input_singular_vectors=u_pairwise_cosine_similarities.tolist(),
            output_singular_vectors=v_pairwise_cosine_similarities.tolist(),
            component_indices=component_subcomponent_indices.tolist(),
        )


def pairwise_cosine_similarities(vectors: Float[Tensor, "n d"]) -> Float[Tensor, "n n"]:
    return F.cosine_similarity(vectors[:, None, :], vectors[None, :, :], dim=-1)
