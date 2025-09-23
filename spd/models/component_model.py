from collections.abc import Callable, Generator, Sequence
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
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.components import (
    Components,
    ComponentsMaskInfo,
    EmbeddingComponents,
    GateType,
    Identity,
    LinearComponents,
    MLPGates,
    VectorMLPGates,
    VectorSharedMLPGate,
)
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidTypes
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.general_utils import fetch_latest_local_checkpoint, resolve_class
from spd.utils.module_utils import get_target_module_paths
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

        self.target_model = target_model
        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr

        module_paths = get_target_module_paths(target_model, target_module_patterns)

        self.components = ComponentModel._create_components(
            target_model=target_model,
            module_paths=module_paths,
            C=C,
        )
        self._components = nn.ModuleDict(
            {k.replace(".", "-"): self.components[k] for k in sorted(self.components)}
        )

        self.gates = ComponentModel._create_gates(
            target_model=target_model,
            module_paths=module_paths,
            C=C,
            gate_type=gate_type,
            gate_hidden_dims=gate_hidden_dims,
        )
        self._gates = nn.ModuleDict(
            {k.replace(".", "-"): self.gates[k] for k in sorted(self.gates)}
        )

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        target_module = self.target_model.get_submodule(module_name)

        match target_module:
            case RadfordConv1D():
                return target_module.weight.T
            case nn.Linear() | nn.Embedding():
                return target_module.weight
            case Identity():
                p = next(self.parameters())
                return torch.eye(target_module.d, device=p.device, dtype=p.dtype)
            case _:
                raise ValueError(f"Module {target_module} not supported")

    @staticmethod
    def _create_component(
        target_module: nn.Module,
        C: int,
    ) -> Components:
        match target_module:
            case nn.Linear():
                d_out, d_in = target_module.weight.shape
                component = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case RadfordConv1D():
                d_in, d_out = target_module.weight.shape
                component = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case Identity():
                component = LinearComponents(
                    C=C,
                    d_in=target_module.d,
                    d_out=target_module.d,
                    bias=None,
                )
            case nn.Embedding():
                component = EmbeddingComponents(
                    C=C,
                    vocab_size=target_module.num_embeddings,
                    embedding_dim=target_module.embedding_dim,
                )
            case _:
                raise ValueError(f"Module {target_module} not supported")

        return component

    @staticmethod
    def _create_components(
        target_model: nn.Module,
        module_paths: list[str],
        C: int,
    ) -> dict[str, Components]:
        components: dict[str, Components] = {}
        for module_path in module_paths:
            target_module = target_model.get_submodule(module_path)
            components[module_path] = ComponentModel._create_component(target_module, C)
        return components

    @staticmethod
    def _create_gate(
        target_module: nn.Module,
        component_C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a gate based on gate_type and module type."""
        if isinstance(target_module, nn.Embedding):
            assert gate_type == "mlp", "Embedding modules only supported for gate_type='mlp'"

        if gate_type == "mlp":
            return MLPGates(C=component_C, hidden_dims=gate_hidden_dims)

        match target_module:
            case nn.Linear():
                input_dim = target_module.weight.shape[1]
            case RadfordConv1D():
                input_dim = target_module.weight.shape[0]
            case Identity():
                input_dim = target_module.d
            case _:
                raise ValueError(f"Module {type(target_module)} not supported for {gate_type=}")

        match gate_type:
            case "vector_mlp":
                return VectorMLPGates(
                    C=component_C, input_dim=input_dim, hidden_dims=gate_hidden_dims
                )
            case "shared_mlp":
                return VectorSharedMLPGate(
                    C=component_C, input_dim=input_dim, hidden_dims=gate_hidden_dims
                )

    @staticmethod
    def _create_gates(
        target_model: nn.Module,
        module_paths: list[str],
        C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
    ) -> dict[str, nn.Module]:
        gates: dict[str, nn.Module] = {}
        for module_path in module_paths:
            target_module = target_model.get_submodule(module_path)
            gates[module_path] = ComponentModel._create_gate(
                target_module,
                C,
                gate_type,
                gate_hidden_dims,
            )
        return gates

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

    @override
    def forward(
        self,
        *args: Any,
        mode: Literal["target", "components", "input_cache"] | None = "target",
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
                - 'input_cache': Forward with input caching (requires module_names)
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo
                (required for mode='components').
            module_names: List of module names to cache inputs for
                (required for mode='input_cache')

        If `pretrained_model_output_attr` is set, return the attribute of the model's output.
        """
        if mode == "components":
            assert mask_infos is not None, "mask_infos are required for mode='components'"
            return self._forward_with_components(*args, mask_infos=mask_infos, **kwargs)
        elif mode == "input_cache":
            assert module_names is not None, (
                "module_names parameter is required for mode='input_cache'"
            )
            return self._forward_with_input_cache(*args, module_names=module_names, **kwargs)
        else:
            return self._extract_output(self.target_model(*args, **kwargs))

    @contextmanager
    def _attach_forward_hooks(
        self, hooks: dict[str, Callable[..., Any]]
    ) -> Generator[None, None, None]:
        """Context manager to temporarily attach forward hooks to the target model."""
        handles: list[RemovableHandle] = []
        for module_name, hook in hooks.items():
            target_module = self.target_model.get_submodule(module_name)
            handle = target_module.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _forward_with_components(
        self, *args: Any, mask_infos: dict[str, ComponentsMaskInfo], **kwargs: Any
    ) -> Any:
        """Forward pass with temporary component replacements. `masks` is a dictionary mapping
        component paths to mask infos. A mask info being present means that the module will be replaced
        with components, and the value of the mask info will be used as the mask for the components.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo
        """

        def fwd_hook(
            _module: nn.Module,
            args: list[Any],
            kwargs: dict[Any, Any],
            _output: Any,
            components: Components,
            mask_info: ComponentsMaskInfo,
        ) -> None | Any:
            assert len(args) == 1, "Expected 1 argument"
            assert len(kwargs) == 0, "Expected no keyword arguments"
            x = args[0]
            assert isinstance(x, Tensor), "Expected input tensor"

            components_out = components(
                x,
                mask=mask_info.component_mask,
                weight_delta_and_mask=mask_info.weight_delta_and_mask,
            )

            return components_out

        hooks: dict[str, Callable[..., Any]] = {}
        for module_name, mask_info in mask_infos.items():
            components = self.components[module_name]
            hooks[module_name] = partial(fwd_hook, components=components, mask_info=mask_info)

        with self._attach_forward_hooks(hooks):
            raw_out = self.target_model(*args, **kwargs)

        return self._extract_output(raw_out)

    def _forward_with_input_cache(
        self, *args: Any, module_names: list[str], **kwargs: Any
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with caching at the input to the modules given by `module_names`.
        Args:
            module_names: List of module names to cache the inputs to.
        Returns:
            Tuple of (model output, input cache dictionary)
        """

        cache = {}

        def cache_hook(
            _module: nn.Module,
            args: list[Any],
            kwargs: dict[Any, Any],
            _output: Any,
            param_name: str,
        ) -> None:
            assert len(args) == 1, "Expected 1 argument"
            assert len(kwargs) == 0, "Expected no keyword arguments"
            x = args[0]
            assert isinstance(x, Tensor), "Expected x to be a tensor"
            cache[param_name] = x

        hooks = {
            module_name: partial(cache_hook, param_name=module_name) for module_name in module_names
        }
        with self._attach_forward_hooks(hooks):
            raw_out = self.target_model(*args, **kwargs)

        out = self._extract_output(raw_out)
        return out, cache

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
            target_model = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            assert issubclass(model_class, LoadableModule), (
                f"Model class {model_class} should be a subclass of LoadableModule which "
                "defines a `from_pretrained` method"
            )
            assert run_info.config.pretrained_model_path is not None
            target_model = model_class.from_pretrained(run_info.config.pretrained_model_path)

        target_model.eval()
        target_model.requires_grad_(False)

        if config.identity_module_patterns is not None:
            insert_identity_operations_(
                target_model, identity_patterns=config.identity_module_patterns
            )

        comp_model = ComponentModel(
            target_model=target_model,
            target_module_patterns=config.all_module_patterns(),
            C=config.C,
            gate_hidden_dims=config.gate_hidden_dims,
            gate_type=config.gate_type,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )

        comp_model_weights = torch.load(
            run_info.checkpoint_path, map_location="cpu", weights_only=True
        )

        handle_deprecated_state_dict_keys_(comp_model_weights)

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

            match gates:
                case MLPGates():
                    gate_input = self.components[param_name].get_inner_acts(acts)
                case VectorMLPGates() | VectorSharedMLPGate():
                    gate_input = acts
                case _:
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

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, " d_out d_in"]]:
        """Calculate the weight differences between the target and component weights (V@U) for each layer."""
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas


def _transform_key(key: str) -> str:
    key = key[:]  # make a copy

    # do this first to simplify the rest. All following logic assumes "target_model" naming convention
    key = key.replace("patched_model", "target_model")

    has_components = ".components." in key
    has_original = ".original." in key
    if has_components and has_original:
        raise ValueError(
            f"Key {key} has both components and original, this is technically possible but unsupported"
        )

    if has_components:
        # we need to move the path out of the target model, remove the "components" nesting,
        # normalize the path, and put it under the "_components" ModuleDict
        assert key.startswith("target_model.")
        key = key.removeprefix("target_model.")
        path, contents = key.split(".components.")
        normalized_path = path.replace(".", "-")
        key = f"_components.{normalized_path}.{contents}"

    if has_original:
        # simpler: just collapse the nesting
        key = key.replace(".original.", ".")

    return key


def handle_deprecated_state_dict_keys_(state_dict: dict[str, Tensor]) -> None:
    """Maps deprecated state dict keys to new state dict keys
    old: used nested ComponentsOrModule wrappers
    new: uses a single ModuleDict for all components
    """
    for key in list(state_dict.keys()):
        new_key = _transform_key(key)
        if new_key == key:
            continue
        assert new_key not in state_dict, f"Renamed key {key} already exists in state_dict"
        state_dict[new_key] = state_dict.pop(key)
