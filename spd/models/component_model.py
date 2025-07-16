import fnmatch
from collections.abc import Mapping
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, override

import einops
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.configs import Config
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.general_utils import load_pretrained
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class ComponentModel(nn.Module):
    """Wrapper around an arbitrary model for running SPD.

    The underlying *base model* can be any subclass of `nn.Module` (e.g.
    `LlamaForCausalLM`, `AutoModelForCausalLM`) as long as its sub-module names
    match the patterns you pass in `target_module_patterns`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_module_patterns: list[str],
        C: int,
        n_ci_mlp_neurons: int,
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()
        self.model = base_model
        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr
        self.components = self.create_target_components(
            target_module_patterns=target_module_patterns, C=C
        )

        gate_class = GateMLP if n_ci_mlp_neurons > 0 else Gate
        gate_kwargs = {"C": C}
        if n_ci_mlp_neurons > 0:
            gate_kwargs["n_ci_mlp_neurons"] = n_ci_mlp_neurons

        self.gates = nn.ModuleDict({name: gate_class(**gate_kwargs) for name in self.components})

    def create_target_components(self, target_module_patterns: list[str], C: int) -> nn.ModuleDict:
        """Create target components for the model."""
        components: dict[str, LinearComponent | EmbeddingComponent] = {}
        matched_patterns: set[str] = set()

        for name, module in self.model.named_modules():
            for pattern in target_module_patterns:
                if fnmatch.fnmatch(name, pattern):
                    matched_patterns.add(pattern)
                    if isinstance(module, nn.Linear):
                        d_out, d_in = module.weight.shape
                        # Replace "." with "-" in the name to avoid issues with module dict keys
                        components[name.replace(".", "-")] = LinearComponent(
                            d_in=d_in, d_out=d_out, C=C, bias=module.bias
                        )
                    elif isinstance(module, nn.Embedding):
                        components[name.replace(".", "-")] = EmbeddingComponent(
                            vocab_size=module.num_embeddings,
                            embedding_dim=module.embedding_dim,
                            C=C,
                        )
                    else:
                        raise ValueError(
                            f"Module '{name}' matched pattern '{pattern}' but is not nn.Linear or "
                            f"nn.Embedding. Found type: {type(module)}"
                        )
                    break

        unmatched_patterns = set(target_module_patterns) - matched_patterns
        if unmatched_patterns:
            raise ValueError(
                f"The following patterns in target_module_patterns did not match any modules: "
                f"{sorted(unmatched_patterns)}"
            )

        if not components:
            raise ValueError(
                f"No modules found matching target_module_patterns: {target_module_patterns}"
            )
        return nn.ModuleDict(components)

    @override
    def to(self, *args: Any, **kwargs: Any) -> "ComponentModel":
        """Move the model and components to a device."""
        self.model.to(*args, **kwargs)
        for component in self.components.values():
            component.to(*args, **kwargs)
        for gate in self.gates.values():
            gate.to(*args, **kwargs)
        return self

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Regular forward pass of the (target) model.

        If `model_output_attr` is set, return the attribute of the model's output.
        """
        raw_out = self.model(*args, **kwargs)
        if self.pretrained_model_output_attr is None:
            out = raw_out
        else:
            out = getattr(raw_out, self.pretrained_model_output_attr)
        return out

    @contextmanager
    def _replaced_modules(
        self,
        components: Mapping[str, LinearComponent | EmbeddingComponent],
        masks: dict[str, Float[Tensor, "... C"]] | None = None,
    ):
        """Context manager for temporarily replacing modules with components.

        Args:
            components: Dictionary mapping component names to components
            masks: Optional dictionary mapping component names to masks
        """
        old_modules = {}

        # Setup: Save old modules and replace with components
        for module_name, component in components.items():
            old_module = self.model.get_submodule(module_name)
            assert old_module is not None, f"Module {module_name} not found"

            old_modules[module_name] = old_module

            # Set mask if provided
            if masks is not None:
                component.mask = masks[module_name]

            # Replace module
            self.model.set_submodule(module_name, component)

        try:
            yield
        finally:
            # Teardown: Restore original modules and clear masks
            for module_name, old_module in old_modules.items():
                self.model.set_submodule(module_name, old_module)

            # Clear masks from all components
            for component in components.values():
                component.mask = None

    def forward_with_components(
        self,
        *args: Any,
        components: dict[str, LinearComponent | EmbeddingComponent],
        masks: dict[str, Float[Tensor, "... C"]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass with temporary component replacements.

        Args:
            components: Dictionary mapping component names to components
            masks: Optional dictionary mapping component names to masks
        """
        with self._replaced_modules(components, masks):
            return self(*args, **kwargs)

    def forward_with_pre_forward_cache_hooks(
        self, *args: Any, module_names: list[str], **kwargs: Any
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with caching at the input to the modules given by `module_names`.

        Args:
            module_names: List of module names to cache the inputs to.

        Returns:
            Tuple of (model output, cache dictionary)
        """
        cache = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def cache_hook(_: nn.Module, input: tuple[Tensor, ...], param_name: str) -> None:
            cache[param_name] = input[0]

        # Register hooks
        for module_name in module_names:
            module = self.model.get_submodule(module_name)
            assert module is not None, f"Module {module_name} not found"
            handles.append(
                module.register_forward_pre_hook(partial(cache_hook, param_name=module_name))
            )

        try:
            out = self(*args, **kwargs)
            return out, cache
        finally:
            for handle in handles:
                handle.remove()

    def forward_with_components_and_pre_forward_cache_hooks(
        self,
        *args: Any,
        components: dict[str, LinearComponent | EmbeddingComponent],
        masks: dict[str, Float[Tensor, "... C"]] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with temporary component replacements and pre-forward cache hooks.

        Args:
            components: Dictionary mapping component names to components
            masks: Optional dictionary mapping component names to masks
        """
        with self._replaced_modules(components, masks):
            return self.forward_with_pre_forward_cache_hooks(
                *args, module_names=list(components.keys()), **kwargs
            )

    def forward_with_components_and_post_forward_hooks(
        self,
        *args: Any,
        components: dict[str, LinearComponent | EmbeddingComponent],
        masks: dict[str, Float[Tensor, "... C"]] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with temporary component replacements and post-forward hooks.

        Args:
            components: Dictionary mapping component names to components
            masks: Optional dictionary mapping component names to masks
        """
        with self._replaced_modules(components, masks):
            return self.forward_with_post_forward_cache_hooks(
                *args, **kwargs, module_names=list(components.keys())
            )

    def model_method_with_components(
        self,
        *args: Any,
        method_name: str,
        components: dict[str, LinearComponent | EmbeddingComponent],
        masks: dict[str, Float[Tensor, "... C"]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Calls any method of the model with the given components.
        """
        with self._replaced_modules(components, masks):
            try:
                method = getattr(self.model, method_name)
                return method(*args, **kwargs)
            except AttributeError as e:
                raise ValueError(f"Method '{method_name}' not found in the model") from e

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> tuple[Path, Path]:
        """Download the relevant files from a wandb run.

        Returns:
            Tuple of (model_path, config_path)
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)

        return checkpoint_path, final_config_path

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["ComponentModel", Config, Path]:
        """Load a trained ComponentModel checkpoint along with its original config.

        The method supports two storage schemes:
        1.  A direct local path to the checkpoint file (plus `final_config.yaml` in
            the same directory).
        2.  A WandB reference of the form ``wandb:<entity>/<project>/runs/<run_id>``.
        """

        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            api = wandb.Api()
            run: Run = api.run(wandb_path)
            model_path, config_path = cls._download_wandb_files(wandb_path)
            out_dir = fetch_wandb_run_dir(run.id)
        else:
            model_path = Path(path)
            config_path = Path(path).parent / "final_config.yaml"
            out_dir = Path(path).parent

        model_weights = torch.load(model_path, map_location="cpu", weights_only=True)
        with open(config_path) as f:
            config = Config(**yaml.safe_load(f))

        assert config.pretrained_model_class is not None

        base_model_raw = load_pretrained(
            path_to_class=config.pretrained_model_class,
            model_path=config.pretrained_model_path,
            model_name_hf=config.pretrained_model_name_hf,
        )
        base_model = base_model_raw[0] if isinstance(base_model_raw, tuple) else base_model_raw

        comp_model = ComponentModel(
            base_model=base_model,
            target_module_patterns=config.target_module_patterns,
            C=config.C,
            n_ci_mlp_neurons=config.n_ci_mlp_neurons,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )
        comp_model.load_state_dict(model_weights)
        return comp_model, config, out_dir


def init_Vs_and_Us_(
    model: ComponentModel, components: dict[str, LinearComponent | EmbeddingComponent]
) -> None:
    """Initialize the V and U matrices.
    1. Normalize every component to 1.
    2. Take inner product with original model
    3. This gives you roughly how much overlap there is with the target model.
    4. Scale the Us by this value (we can choose either matrix)
    """
    # NOTE: This may increase memory usage if done on GPU.
    for param_name, component in components.items():
        V = component.V
        U = component.U
        target_weight = model.model.get_parameter(param_name + ".weight")
        if isinstance(component, EmbeddingComponent):
            target_weight = target_weight.T  # (d_out d_in)

        # Make V and U have unit norm in the d_in and d_out dimensions
        V.data[:] = torch.randn_like(V.data)
        U.data[:] = torch.randn_like(U.data)
        V.data[:] = V.data / V.data.norm(dim=-2, keepdim=True)
        U.data[:] = U.data / U.data.norm(dim=-1, keepdim=True)

        # Calculate inner products
        inner = einops.einsum(U, target_weight, "C d_out, d_out d_in -> C d_in")
        C_norms = einops.einsum(inner, V, "C d_in, d_in C -> C")

        # Scale U by the inner product.
        U.data[:] = U.data * C_norms.unsqueeze(-1)
