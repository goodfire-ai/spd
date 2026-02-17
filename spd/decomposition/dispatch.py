from spd.decomposition import Decomposition
from spd.decomposition.configs import (
    CLTDecompositionConfig,
    SPDDecompositionConfig,
    TargetDecompositionConfig,
    TranscoderDecompositionConfig,
)
from spd.decomposition.spd import SPDDecomposition


def decomposition_from_id(id: str) -> Decomposition:
    if id.startswith("s-"):
        return SPDDecomposition(f"goodfire/spd/{id}", 0.0)
    elif id.startswith("clt-"):
        raise NotImplementedError("CLT decomposition not implemented yet")
    elif id.startswith("tc-"):
        raise NotImplementedError("Transcoder decomposition not implemented yet")
    elif id.startswith("molt-"):
        raise NotImplementedError("MOLT decomposition not implemented yet")

    raise ValueError(f"Unsupported decomposition ID: {id}")


def decomposition_from_config(config: TargetDecompositionConfig) -> Decomposition:
    match config:
        case SPDDecompositionConfig():
            return SPDDecomposition.from_config(config)
        case CLTDecompositionConfig():
            raise NotImplementedError("CLT decomposition not implemented yet")
        case TranscoderDecompositionConfig():
            raise NotImplementedError("Transcoder decomposition not implemented yet")
