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
from transformers.models.gpt2.modeling_gpt2 import Conv1D as RadfordConv1D
from wandb.apis.public import Run

from spd.configs import Config
from spd.interfaces import LoadableModule, RunInfo
from spd.mask_info import ComponentsMaskInfo
from spd.models.components import (
    Components,
    EmbeddingComponents,
    GateMLPs,
    GateType,
    LayerwiseGlobalGateMLP,
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

        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr
        self.target_module_paths = target_module_paths
        self.target_model = target_model

        # Keep typed dict for program logic
        self.components: dict[str, Components] = self._build_components()
        # Also register as modules for state_dict/optim
        self._components = nn.ModuleDict(
            {
                module_path.replace(".", "-"): comp
                for module_path, comp in sorted(self.components.items())
            }
        )

        self.gates = self._build_gates(gate_type=gate_type, gate_hidden_dims=gate_hidden_dims)
        self._gates = nn.ModuleDict({k.replace(".", "-"): v for k, v in sorted(self.gates.items())})

    def _build_components(self) -> dict[str, Components]:
        """Construct component modules for each target module path.

        Returns:
            Mapping from module path to its corresponding `Components` implementation.
        """
        components: dict[str, Components] = {}
        for module_path in self.target_module_paths:
            module = self.target_model.get_submodule(module_path)
            if isinstance(module, nn.Linear):
                d_out, d_in = module.weight.shape
                components[module_path] = LinearComponents(
                    C=self.C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=module.bias.data if module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            elif isinstance(module, nn.Embedding):
                components[module_path] = EmbeddingComponents(
                    C=self.C,
                    vocab_size=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                )
            elif isinstance(module, RadfordConv1D):
                d_in, d_out = module.weight.shape
                components[module_path] = LinearComponents(
                    C=self.C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=module.bias.data if module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            else:
                raise ValueError(
                    f"Module '{module_path}' matched pattern is not nn.Linear, nn.Embedding, or Huggingface Conv1D. Found type: {type(module)}"
                )
        return components

    def _build_gates(
        self, *, gate_type: GateType, gate_hidden_dims: list[int]
    ) -> dict[str, nn.Module]:
        """Construct gates for each component based on the configured gate type.

        Args:
            gate_type: Which gate architecture to use.
            gate_hidden_dims: Hidden layer sizes for the gate MLPs.

        Returns:
            Mapping from module path to its gate `nn.Module`.
        """
        gates: dict[str, nn.Module] = {}
        for module_path, comp in self.components.items():
            original_module = self.target_model.get_submodule(module_path)
            gates[module_path] = ComponentModel._create_gate(
                original_module=original_module,
                component_C=comp.C,
                gate_type=gate_type,
                gate_hidden_dims=gate_hidden_dims,
            )
        return gates

    def get_original_module(self, module_name: str) -> nn.Module:
        return self.target_model.get_submodule(module_name)

    def get_original_weight(self, module_name: str) -> Float[Tensor, " d_out d_in"]:
        module = self.get_original_module(module_name)
        if isinstance(module, RadfordConv1D):
            return module.weight.T
        elif isinstance(module, nn.Linear | nn.Embedding):
            return module.weight
        else:
            raise TypeError(
                f"Module {module_name} not one of nn.Linear, nn.Embedding, or RadfordConv1D"
            )

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
    def _create_gate(
        original_module: nn.Module,
        component_C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a gate based on gate_type and module type."""
        if gate_type == "mlp":
            gate = GateMLPs(C=component_C, hidden_dims=gate_hidden_dims)
        else:
            assert gate_type in ["vector_mlp", "layerwise_global_mlp"], (
                f"Unknown gate type: {gate_type}"
            )
            assert not isinstance(original_module, nn.Embedding), (
                "Embedding modules only supported for gate_type='mlp'"
            )
            if isinstance(original_module, nn.Linear):
                input_dim = original_module.weight.shape[1]
            elif isinstance(original_module, RadfordConv1D):
                input_dim = original_module.weight.shape[0]
            else:
                raise ValueError(f"Module {type(original_module)} not supported for {gate_type=}")

            if gate_type == "vector_mlp":
                gate = VectorGateMLPs(
                    C=component_C, input_dim=input_dim, hidden_dims=gate_hidden_dims
                )
            else:
                assert gate_type == "layerwise_global_mlp"
                gate = LayerwiseGlobalGateMLP(
                    C=component_C, input_dim=input_dim, hidden_dims=gate_hidden_dims
                )
        return gate

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
            mask_infos: Dictionary mapping module names to ComponentMaskInfo
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
    def _component_hooks(self, mask_infos: dict[str, ComponentsMaskInfo]):
        """Temporarily override selected modules with component outputs via forward hooks."""
        assert set(mask_infos).issubset(set(self.components)), (
            f"Mask keys must be subset of component keys. Extra: {set(mask_infos) - set(self.components)}"
        )
        handles: list[RemovableHandle] = []
        try:
            for module_name, mask_info in mask_infos.items():
                module = self.get_original_module(module_name)
                comp = self.components[module_name]

                def hook(
                    _module: nn.Module,
                    input: tuple[Tensor, "..."],
                    _output: Tensor,
                    *,
                    _comp: Components = comp,
                    _mi: ComponentsMaskInfo = mask_info,
                ) -> Tensor:
                    return _comp(
                        input[0],
                        mask=_mi.component_mask,
                        weight_delta_and_mask=_mi.weight_delta_and_mask,
                    )

                handles.append(module.register_forward_hook(hook))
            yield
        finally:
            for h in handles:
                h.remove()

    def _forward_target(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the target model."""
        out = self.target_model(*args, **kwargs)
        out = self._extract_output(out)
        return out

    def _forward_with_components(
        self, *args: Any, mask_infos: dict[str, ComponentsMaskInfo], **kwargs: Any
    ) -> Any:
        """Forward pass with temporary component replacements. `masks` is a dictionary mapping
        component paths to masks. A mask being present means that the module will be replaced
        with components, and the value of the mask will be used as the mask for the components.

        Args:
            mask_infos: Dictionary mapping module names to ComponentMaskInfo
        """
        with self._component_hooks(mask_infos=mask_infos):
            raw_out = self.target_model(*args, **kwargs)
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
        cache: dict[str, Tensor] = {}
        handles: list[RemovableHandle] = []

        def cache_hook(_: nn.Module, input: tuple[Tensor, "..."], param_name: str) -> None:
            cache[param_name] = input[0]

        # Register hooks
        for module_name in module_names:
            module = self.target_model.get_submodule(module_name)
            assert module is not None, f"Module {module_name} not found"
            handles.append(
                module.register_forward_pre_hook(partial(cache_hook, param_name=module_name))
            )

        try:
            raw_out = self.target_model(*args, **kwargs)
            out = self._extract_output(raw_out)
            return out, cache
        finally:
            for handle in handles:
                handle.remove()

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
            elif isinstance(gates, VectorGateMLPs | LayerwiseGlobalGateMLP):
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
