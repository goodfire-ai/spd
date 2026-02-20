"""SLURM parallelization infrastructure for attribution graph generation."""

from .manifest import JobManifest, PromptStatus

__all__ = ["JobManifest", "PromptStatus"]
