"""wrapper for Conv1D, see https://github.com/goodfire-ai/spd/issues/139"""

from transformers.pytorch_utils import Conv1D as RadfordConv1D

__all__ = ["RadfordConv1D"]
