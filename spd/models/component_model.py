import fnmatch
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, override

import torch
import wandb
import yaml
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.modeling_utils import Conv1D as RadfordConv1D
from wandb.apis.public import Run

from spd.configs import Config
from spd.interfaces import LoadableModule, RunInfo
from spd.models.components import (
    Components,
    ComponentsMaskInfo,
    ComponentsOrModule,
    EmbeddingComponents,
    GateMLPs,
    GateType,
    LinearComponents,
    VectorGateMLPs,
)
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidTypes
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.general_utils import fetch_latest_local_checkpoint, resolve_class
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


@dataclass
class SPDRunInfo(RunInfo[Config]):
    """Run info from training a ComponentModel (i.e. from an SPD run)."""

    @override
    @classmethod
    def from_path(cls, path: ModelPath) -> "SPDRunInfo":
        """Load the run info from a wandb run or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                comp_model_path = fetch_latest_local_checkpoint(run_dir, prefix="model")
                config_path = run_dir / "final_config.yaml"
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                comp_model_path, config_path = ComponentModel._download_wandb_files(wandb_path)
        else:
            comp_model_path = Path(path)
            config_path = Path(path).parent / "final_config.yaml"

        with open(config_path) as f:
            config = Config(**yaml.safe_load(f))

        return cls(checkpoint_path=comp_model_path, config=config)


class ComponentModel(LoadableModule):
    """Wrapper around an arbitrary model for running SPD.

    The underlying *base model* can be any subclass of `nn.Module` (e.g.
    `LlamaForCausalLM`, `AutoModelForCausalLM`) as long as its sub-module names
    match the patterns you pass in `target_module_patterns`.
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_module_patterns: list[str],
        C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        target_module_paths = ComponentModel._get_target_module_paths(
            target_model, target_module_patterns
        )

        patched_model, components_or_modules = ComponentModel._patch_modules(
            model=target_model,
            module_paths=target_module_paths,
            C=C,
        )

        gates = ComponentModel._make_gates(gate_type, gate_hidden_dims, components_or_modules)

        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr
        self.target_module_paths = target_module_paths

        # We keep components_or_modules around to easily access the nested, inserted components
        # State_dict will pick the components up because they're attached to the target_model
        # via set_submodule
        self.components_or_modules = components_or_modules
        # We keep gates as a plain dict so it's properly typed, as ModuleDict isn't generic
        self.gates = gates

        # these are the actual registered submodules
        self.patched_model = patched_model
        self._gates = nn.ModuleDict(
            {k.replace(".", "-"): self.gates[k] for k in sorted(self.gates)}
        )

    @property
    def components(self) -> dict[str, Components]:
        return {name: cm.components for name, cm in self.components_or_modules.items()}

    def _extract_output(self, raw_output: Any) -> Any:
        """Extract the desired output from the model's raw output.

        If pretrained_model_output_attr is None, returns the raw output directly.
        If pretrained_model_output_attr starts with "idx_", returns the index specified by the
        second part of the string. E.g. "idx_0" returns the first element of the raw output.
        Otherwise, returns the specified attribute from the raw output.

        Args:
            raw_output: The raw output from the model.

        Returns:
            The extracted output.
        """
        if self.pretrained_model_output_attr is None:
            return raw_output
        elif self.pretrained_model_output_attr.startswith("idx_"):
            idx_val = int(self.pretrained_model_output_attr.split("_")[1])
            assert isinstance(raw_output, Sequence), (
                f"raw_output must be a sequence, not {type(raw_output)}"
            )
            assert idx_val < len(raw_output), (
                f"Index {idx_val} out of range for raw_output of length {len(raw_output)}"
            )
            return raw_output[idx_val]
        else:
            return getattr(raw_output, self.pretrained_model_output_attr)

    @staticmethod
    def _get_target_module_paths(model: nn.Module, target_module_patterns: list[str]) -> list[str]:
        """Find the target_module_patterns that match real modules in the target model.

        e.g. `["layers.*.mlp_in"]` ->  `["layers.1.mlp_in", "layers.2.mlp_in"]`.
        """

        names_out: list[str] = []
        matched_patterns: set[str] = set()
        for name, _ in model.named_modules():
            for pattern in target_module_patterns:
                if fnmatch.fnmatch(name, pattern):
                    matched_patterns.add(pattern)
                    names_out.append(name)

        unmatched_patterns = set(target_module_patterns) - matched_patterns
        if unmatched_patterns:
            raise ValueError(
                f"The following patterns in target_module_patterns did not match any modules: "
                f"{sorted(unmatched_patterns)}"
            )

        return names_out

    @staticmethod
    def _patch_modules(
        model: nn.Module,
        module_paths: list[str],
        C: int,
    ) -> tuple[nn.Module, dict[str, ComponentsOrModule]]:
        """Replace nn.Modules with ComponentsOrModule objects based on target_module_paths.

        This method mutates and returns `model`, and returns a dictionary of references
        to the newly inserted ComponentsOrModule objects.

        A module is modified in the target model if that module exists in module_paths.

        Args:
            model: The model to replace modules in.
            module_paths: The paths to the modules to replace.
            C: The number of components to use.

        Returns:
            A dictionary mapping module paths to the newly inserted ComponentsOrModule objects
            within `model`.

        Example:
            >>> model
            MyModel(
                (linear): Linear(in_features=10, out_features=20, bias=True)
            )
            >>> target_module_paths = ["linear"]
            >>> module_paths = ["linear"]
            >>> components_or_modules = _patch_modules(
            ...     model,
            ...     module_paths,
            ...     C=2,
            ... )
            >>> print(model)
            MyModel(
                (linear): ComponentsOrModule(
                    (target): Linear(in_features=10, out_features=20, bias=True),
                    (components): LinearComponents(C=2, d_in=10, d_out=20, bias=True),
                )
            )
        """
        components_or_modules: dict[str, ComponentsOrModule] = {}

        # Deterministic, order-preserving deduplicated list (critical for DDP param order)
        all_paths = list(dict.fromkeys(list(module_paths)))

        for module_path in all_paths:
            module = model.get_submodule(module_path)

            if isinstance(module, nn.Linear):
                d_out, d_in = module.weight.shape
                components = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=module.bias.data if module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            elif isinstance(module, nn.Embedding):
                components = EmbeddingComponents(
                    C=C,
                    vocab_size=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                )
            elif isinstance(module, RadfordConv1D):
                d_in, d_out = module.weight.shape
                components = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=module.bias.data if module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            # TODO
            # elif isinstance(module, nn.Identity):
            #     d =
            #     components = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
            else:
                raise ValueError(
                    f"Module '{module_path}' matched pattern is not nn.Linear, nn.Embedding,"
                    f"or Huggingface Conv1D. Found type: {type(module)}"
                )

            replacement = ComponentsOrModule(target=module, components=components)

            model.set_submodule(module_path, replacement)
            components_or_modules[module_path] = replacement

        return model, components_or_modules

    @staticmethod
    def _create_gate(
        target_module: nn.Module,
        component_C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a gate based on gate_type and module type."""
        if gate_type == "mlp":
            return GateMLPs(C=component_C, hidden_dims=gate_hidden_dims)
        else:
            if isinstance(target_module, nn.Linear):
                input_dim = target_module.weight.shape[1]
            elif isinstance(target_module, RadfordConv1D):
                input_dim = target_module.weight.shape[0]
            else:
                assert isinstance(target_module, nn.Embedding)
                raise ValueError("Embedding modules only supported for gate_type='mlp'")
            return VectorGateMLPs(C=component_C, input_dim=input_dim, hidden_dims=gate_hidden_dims)

    @staticmethod
    def _make_gates(
        gate_type: GateType,
        gate_hidden_dims: list[int],
        components_or_modules: dict[str, ComponentsOrModule],
    ) -> dict[str, nn.Module]:
        gates: dict[str, nn.Module] = {}

        for module_path in sorted(components_or_modules):
            component_or_module = components_or_modules[module_path]
            gates[module_path] = ComponentModel._create_gate(
                component_or_module.target,
                component_or_module.components.C,
                gate_type,
                gate_hidden_dims,
            )

        return gates

    @override
    def forward(
        self,
        *args: Any,
        mode: Literal["target", "components", "pre_forward_cache"] | None = "target",
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        module_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass of the patched model.

        NOTE: We need all the forward options in forward in order for DistributedDataParallel to
        work (https://discuss.pytorch.org/t/is-it-ok-to-use-methods-other-than-forward-in-ddp/176509).

        Args:
            mode: The type of forward pass to perform:
                - 'target': Standard forward pass of the target model
                - 'components': Forward with component replacements (requires masks)
                - 'pre_forward_cache': Forward with pre-forward caching (requires module_names)
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo
                (required for mode='components').
            module_names: List of module names to cache inputs for
                (required for mode='pre_forward_cache')

        If `pretrained_model_output_attr` is set, return the attribute of the model's output.
        """
        if mode == "components":
            assert mask_infos is not None, "mask_infos are required for mode='components'"
            return self._forward_with_components(*args, mask_infos=mask_infos, **kwargs)
        elif mode == "pre_forward_cache":
            assert module_names is not None, (
                "module_names parameter is required for mode='pre_forward_cache'"
            )
            return self._forward_with_pre_forward_cache_hooks(
                *args, module_names=module_names, **kwargs
            )
        else:
            return self._forward_target(*args, **kwargs)

    @contextmanager
    def _replaced_modules(self, mask_infos: dict[str, ComponentsMaskInfo]):
        """Set the forward_mode of ComponentOrModule objects and apply masks.

        A module's forward_mode is set to "components" if there is an entry in mask_infos for
        the module name.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo.
        """
        for module_name, c_or_m in self.components_or_modules.items():
            c_or_m.assert_pristine()

            if (mask_info := mask_infos.get(module_name)) is not None:
                c_or_m.forward_mode = ("mixed", mask_info)
            else:
                c_or_m.forward_mode = "target"
        try:
            yield
        finally:
            for c_or_m in self.components_or_modules.values():
                c_or_m.make_pristine()

    def _forward_target(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the target model."""
        for module in self.components_or_modules.values():
            module.assert_pristine()
            module.forward_mode = "target"
        try:
            out = self.patched_model(*args, **kwargs)
        finally:
            for module in self.components_or_modules.values():
                module.make_pristine()

        out = self._extract_output(out)

        return out

    def _forward_with_components(
        self, *args: Any, mask_infos: dict[str, ComponentsMaskInfo], **kwargs: Any
    ) -> Any:
        """Forward pass with temporary component replacements. `masks` is a dictionary mapping
        component paths to masks. A mask being present means that the module will be replaced
        with components, and the value of the mask will be used as the mask for the components.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo
        """
        with self._replaced_modules(mask_infos=mask_infos):
            raw_out = self.patched_model(*args, **kwargs)
            out = self._extract_output(raw_out)
            return out

    def _forward_with_pre_forward_cache_hooks(
        self, *args: Any, module_names: list[str], **kwargs: Any
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with caching at the input to the modules given by `module_names`.

        Args:
            module_names: List of module names to cache the inputs to.

        Returns:
            Tuple of (model output, cache dictionary)
        """
        cache = {}
        handles: list[RemovableHandle] = []

        def cache_hook(_: nn.Module, input: tuple[Tensor, "..."], param_name: str) -> None:
            cache[param_name] = input[0]

        # Register hooks
        for module_name in module_names:
            module = self.patched_model.get_submodule(module_name)
            handle = module.register_forward_pre_hook(partial(cache_hook, param_name=module_name))
            handles.append(handle)

        for module in self.components_or_modules.values():
            module.assert_pristine()
            module.forward_mode = "target"

        try:
            raw_out = self.patched_model(*args, **kwargs)
            out = self._extract_output(raw_out)
            return out, cache
        finally:
            for handle in handles:
                handle.remove()

            for module in self.components_or_modules.values():
                module.make_pristine()

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
    @override
    def from_run_info(cls, run_info: RunInfo[Config]) -> "ComponentModel":
        """Load a trained ComponentModel checkpoint from a run info object."""
        config = run_info.config

        # Load the target model
        model_class = resolve_class(config.pretrained_model_class)
        if config.pretrained_model_name is not None:
            assert hasattr(model_class, "from_pretrained"), (
                f"Model class {model_class} should have a `from_pretrained` method"
            )
            target_model_unpatched = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            assert issubclass(model_class, LoadableModule), (
                f"Model class {model_class} should be a subclass of LoadableModule which "
                "defines a `from_pretrained` method"
            )
            assert run_info.config.pretrained_model_path is not None
            target_model_unpatched = model_class.from_pretrained(
                run_info.config.pretrained_model_path
            )

        target_model_unpatched.eval()
        target_model_unpatched.requires_grad_(False)

        comp_model = ComponentModel(
            target_model=target_model_unpatched,
            target_module_patterns=config.target_module_patterns,
            C=config.C,
            gate_hidden_dims=config.gate_hidden_dims,
            gate_type=config.gate_type,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )

        comp_model_weights = torch.load(
            run_info.checkpoint_path, map_location="cpu", weights_only=True
        )

        comp_model.load_state_dict(comp_model_weights)
        return comp_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "ComponentModel":
        """Load a trained ComponentModel checkpoint from a local or wandb path."""
        run_info = SPDRunInfo.from_path(path)
        return cls.from_run_info(run_info)

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sigmoid_type: SigmoidTypes,
        sampling: Literal["continuous", "binomial"],
        detach_inputs: bool = False,
    ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
        """Calculate causal importances.

        Args:
            pre_weight_acts: The activations before each layer in the target model.
            sigmoid_type: Type of sigmoid to use.
            detach_inputs: Whether to detach the inputs to the gates.

        Returns:
            Tuple of (causal_importances, causal_importances_upper_leaky) dictionaries for each layer.
        """
        causal_importances = {}
        causal_importances_upper_leaky = {}

        for param_name in pre_weight_acts:
            acts = pre_weight_acts[param_name]
            gates = self.gates[param_name]

            if isinstance(gates, GateMLPs):
                gate_input = self.components[param_name].get_inner_acts(acts)
            elif isinstance(gates, VectorGateMLPs):
                gate_input = acts
            else:
                raise ValueError(f"Unknown gate type: {type(gates)}")

            if detach_inputs:
                gate_input = gate_input.detach()

            gate_output = gates(gate_input)

            if sigmoid_type == "leaky_hard":
                lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
                upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
            else:
                # For other sigmoid types, use the same function for both
                lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
                upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

            gate_output_for_lower_leaky = gate_output
            if sampling == "binomial":
                gate_output_for_lower_leaky = 1.05 * gate_output - 0.05 * torch.rand_like(
                    gate_output
                )

            causal_importances[param_name] = lower_leaky_fn(gate_output_for_lower_leaky)
            causal_importances_upper_leaky[param_name] = upper_leaky_fn(gate_output).abs()

        return causal_importances, causal_importances_upper_leaky
