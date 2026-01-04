from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, overload, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from spd.configs import Config, SamplingType
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.components import (
    Components,
    ComponentsMaskInfo,
    EmbeddingComponents,
    Identity,
    LinearComponents,
    MLPCiFn,
    VectorMLPCiFn,
    VectorSharedMLPCiFn,
)
from spd.models.masked_module import MaskedModule
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidType
from spd.spd_types import CiFnType, ModelPath
from spd.utils.general_utils import resolve_class, runtime_cast
from spd.utils.module_utils import ModulePathInfo, expand_module_patterns


@dataclass
class SPDRunInfo(RunInfo[Config]):
    """Run info from training a ComponentModel (i.e. from an SPD run)."""

    config_class = Config
    config_filename = "final_config.yaml"
    checkpoint_prefix = "model"


class OutputWithCache(NamedTuple):
    """Output tensor and cached activations."""

    output: Tensor
    cache: dict[str, Tensor]


@dataclass
class CIOutputs:
    lower_leaky: dict[str, Float[Tensor, "... C"]]
    upper_leaky: dict[str, Float[Tensor, "... C"]]
    pre_sigmoid: dict[str, Tensor]


class ComponentModel(LoadableModule):
    """Wrapper around an arbitrary pytorch model for running SPD.

    The underlying *base model* can be any subclass of `nn.Module` (e.g.
    `LlamaForCausalLM`, `AutoModelForCausalLM`) as long as its sub-module names
    are provided in the `module_path_info` list.

    Forward passes support optional component replacement and/or caching:
    - No args: Standard forward pass of the target model
    - With mask_infos: Components replace the specified modules via forward hooks
    - With cache_type="input": Input activations are cached for the specified modules
    - With cache_type="component_acts": Component activations are cached for the specified modules
    - Both can be used simultaneously for component forward pass with input caching

    We register components and causal importance functions (ci_fns) as modules in this class in order to have them update
    correctly when the model is wrapped in a `DistributedDataParallel` wrapper (and for other
    conveniences).
    """

    def __init__(
        self,
        target_model: nn.Module,
        module_path_info: list[ModulePathInfo],
        ci_fn_type: CiFnType,
        ci_fn_hidden_dims: list[int],
        sigmoid_type: SigmoidType,
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        self.target_model = target_model
        self.pretrained_model_output_attr = pretrained_model_output_attr
        self.module_to_c = {info.module_path: info.C for info in module_path_info}
        self.target_module_paths = list(self.module_to_c.keys())

        # Create trainable components and CI functions against the *original* target modules.
        # We patch the target model only after this, to avoid CI fn creation seeing wrappers.
        self.components = ComponentModel._create_components(
            target_model=target_model, module_to_c=self.module_to_c
        )
        self.ci_fns = ComponentModel._create_ci_fns(
            target_model=target_model,
            module_to_c=self.module_to_c,
            ci_fn_type=ci_fn_type,
            ci_fn_hidden_dims=ci_fn_hidden_dims,
        )
        self._ci_fns = nn.ModuleDict(
            {k.replace(".", "-"): self.ci_fns[k] for k in sorted(self.ci_fns)}
        )

        # Patch the target model in-place: wrap each decomposed module with MaskedModule.
        # IMPORTANT: Process paths in order of decreasing depth (deepest first) so that
        # wrapping a parent module doesn't prevent access to its children.
        # E.g., wrap "linear1.pre_identity" before "linear1".
        self._masked_modules: dict[str, MaskedModule] = {}
        sorted_paths = sorted(self.target_module_paths, key=lambda p: p.count("."), reverse=True)
        for module_path in sorted_paths:
            base_module = target_model.get_submodule(module_path)
            masked = MaskedModule(
                module_name=module_path,
                base=base_module,
                components=self.components[module_path],
            )
            _set_submodule_by_path(target_model, module_path, masked)
            self._masked_modules[module_path] = masked

        if sigmoid_type == "leaky_hard":
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
        else:
            # For other sigmoid types, use the same function for both
            self.lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
            self.upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        target_module = self.target_model.get_submodule(module_name)
        if isinstance(target_module, MaskedModule):
            target_module = target_module.base

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
        module_to_c: dict[str, int],
    ) -> dict[str, Components]:
        components: dict[str, Components] = {}
        for target_module_path, target_module_c in module_to_c.items():
            target_module = target_model.get_submodule(target_module_path)
            components[target_module_path] = ComponentModel._create_component(
                target_module, target_module_c
            )
        return components

    @staticmethod
    def _create_ci_fn(
        target_module: nn.Module,
        C: int,
        ci_fn_type: CiFnType,
        ci_fn_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a causal importance function (ci_fn) based on ci_fn_type and module type."""
        if isinstance(target_module, nn.Embedding):
            assert ci_fn_type == "mlp", "Embedding modules only supported for ci_fn_type='mlp'"

        if ci_fn_type == "mlp":
            return MLPCiFn(C=C, hidden_dims=ci_fn_hidden_dims)

        match target_module:
            case nn.Linear():
                input_dim = target_module.weight.shape[1]
            case RadfordConv1D():
                input_dim = target_module.weight.shape[0]
            case Identity():
                input_dim = target_module.d
            case _:
                raise ValueError(f"Module {type(target_module)} not supported for {ci_fn_type=}")

        match ci_fn_type:
            case "vector_mlp":
                return VectorMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)
            case "shared_mlp":
                return VectorSharedMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)

    @staticmethod
    def _create_ci_fns(
        target_model: nn.Module,
        module_to_c: dict[str, int],
        ci_fn_type: CiFnType,
        ci_fn_hidden_dims: list[int],
    ) -> dict[str, nn.Module]:
        ci_fns: dict[str, nn.Module] = {}
        for target_module_path, target_module_c in module_to_c.items():
            target_module = target_model.get_submodule(target_module_path)
            ci_fns[target_module_path] = ComponentModel._create_ci_fn(
                target_module=target_module,
                C=target_module_c,
                ci_fn_type=ci_fn_type,
                ci_fn_hidden_dims=ci_fn_hidden_dims,
            )
        return ci_fns

    def _extract_output(self, raw_output: Any) -> Tensor:
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
            out = raw_output
        elif self.pretrained_model_output_attr.startswith("idx_"):
            idx_val = int(self.pretrained_model_output_attr.split("_")[1])
            assert isinstance(raw_output, Sequence), (
                f"raw_output must be a sequence, not {type(raw_output)}"
            )
            assert idx_val < len(raw_output), (
                f"Index {idx_val} out of range for raw_output of length {len(raw_output)}"
            )
            out = raw_output[idx_val]
        else:
            out = getattr(raw_output, self.pretrained_model_output_attr)

        assert isinstance(out, Tensor), f"Expected tensor output, got {type(out)}"
        return out

    @overload
    def __call__(
        self,
        *args: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input"],
        **kwargs: Any,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        *args: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
        **kwargs: Any,
    ) -> Tensor: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        *args: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input", "none"] = "none",
        **kwargs: Any,
    ) -> Tensor | OutputWithCache:
        """Forward pass with optional component replacement and/or caching, without hooks.

        Semantics match the previous hook-based implementation:
        - `mask_infos is None and cache_type == "none"`: pure target model forward.
        - `mask_infos is None and cache_type != "none"`: cache on all decomposed modules.
        - `mask_infos is not None`: activate exactly those modules for component replacement.
          If also caching, cache only on the modules provided in `mask_infos`.
        """
        cache: dict[str, Tensor] | None = {} if cache_type != "none" else None

        if cache_type == "none":
            cache_names: set[str] = set()
        else:
            cache_names = (
                set(mask_infos.keys()) if mask_infos is not None else set(self.target_module_paths)
            )

        active_infos = mask_infos if mask_infos is not None else {}

        for name in self.target_module_paths:
            masked = self._masked_modules[name]
            is_active = name in active_infos
            is_caching = name in cache_names
            masked.set_runtime_state(
                active=is_active,
                mask_info=(active_infos[name] if is_active else None),
                cache_type=(cache_type if is_caching else "none"),
                cache=(cache if is_caching else None),
            )

        raw_out = self.target_model(*args, **kwargs)
        out = self._extract_output(raw_out)
        if cache_type == "none":
            return out
        assert cache is not None
        return OutputWithCache(output=out, cache=cache)

    def validate_masked_module_state(self) -> None:
        """Validate that all MaskedModules have consistent runtime state.

        Call this for debugging after set_runtime_state() has been called on all modules
        but before the forward pass. Useful for catching state configuration bugs.

        Raises:
            AssertionError: If any MaskedModule has inconsistent state.
        """
        for masked in self._masked_modules.values():
            masked.validate_state()

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

        if config.identity_module_info is not None:
            insert_identity_operations_(
                target_model,
                identity_module_info=config.identity_module_info,
            )

        module_path_info = expand_module_patterns(target_model, config.all_module_info)

        comp_model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_fn_hidden_dims=config.ci_fn_hidden_dims,
            ci_fn_type=config.ci_fn_type,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
            sigmoid_type=config.sigmoid_type,
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
        sampling: SamplingType,
        detach_inputs: bool = False,
    ) -> CIOutputs:
        """Calculate causal importances.

        Args:
            pre_weight_acts: The activations before each layer in the target model.
            detach_inputs: Whether to detach the inputs to the causal importance function.

        Returns:
            Tuple of (causal_importances, causal_importances_upper_leaky) dictionaries for each layer.
        """
        causal_importances_lower_leaky = {}
        causal_importances_upper_leaky = {}
        pre_sigmoid = {}

        for target_module_name in pre_weight_acts:
            input_activations = pre_weight_acts[target_module_name]
            ci_fn = self.ci_fns[target_module_name]

            match ci_fn:
                case MLPCiFn():
                    ci_fn_input = self.components[target_module_name].get_inner_acts(
                        input_activations
                    )
                case VectorMLPCiFn() | VectorSharedMLPCiFn():
                    ci_fn_input = input_activations
                case _:
                    raise ValueError(f"Unknown ci_fn type: {type(ci_fn)}")

            if detach_inputs:
                ci_fn_input = ci_fn_input.detach()

            ci_fn_output = runtime_cast(Tensor, ci_fn(ci_fn_input))

            if sampling == "binomial":
                ci_fn_output_for_lower_leaky = 1.05 * ci_fn_output - 0.05 * torch.rand_like(
                    ci_fn_output
                )
            else:
                ci_fn_output_for_lower_leaky = ci_fn_output

            lower_leaky_output = self.lower_leaky_fn(ci_fn_output_for_lower_leaky)
            assert lower_leaky_output.all() <= 1.0
            causal_importances_lower_leaky[target_module_name] = lower_leaky_output

            upper_leaky_output = self.upper_leaky_fn(ci_fn_output)
            assert upper_leaky_output.all() >= 0
            causal_importances_upper_leaky[target_module_name] = upper_leaky_output

            pre_sigmoid[target_module_name] = ci_fn_output

        return CIOutputs(
            lower_leaky=causal_importances_lower_leaky,
            upper_leaky=causal_importances_upper_leaky,
            pre_sigmoid=pre_sigmoid,
        )

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Calculate the weight differences between the target and component weights (V@U) for each layer."""
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas


def handle_deprecated_state_dict_keys_(state_dict: dict[str, Tensor]) -> None:
    """Maps deprecated state dict keys to new state dict keys"""
    for key in list(state_dict.keys()):
        new_key: str = key
        # We used to have "_gates.*", now we have "_ci_fns.*"
        if "_gates." in new_key:
            new_key = new_key.replace("_gates.", "_ci_fns.")
        # We used to have prefix "patched_model.*", now we have "target_model.*"
        if new_key.startswith("patched_model."):
            new_key = "target_model." + new_key.removeprefix("patched_model.")
        # We used to have "*.original.weight", now we have "*.weight"
        if new_key.endswith(".original.weight"):
            new_key = new_key.removesuffix(".original.weight") + ".weight"
        # We used to have "*.components.{U,V}", now we have "_components.*.{U,V}"
        if new_key.endswith(".components.U") or new_key.endswith(".components.V"):
            target_module_path: str = (
                new_key.removeprefix("target_model.")
                .removesuffix(".components.U")
                .removesuffix(".components.V")
            )
            # module path has "." replaced with "-"
            new_key = f"_components.{target_module_path.replace('.', '-')}.{new_key.split('.')[-1]}"
        # If we now store components under target_model.<path>.components.{U,V}, map legacy keys.
        if new_key.startswith("_components."):
            # _components.<dashed_path>.<U|V>  -> target_model.<dotted_path>.components.<U|V>
            parts = new_key.split(".")
            if len(parts) == 3 and parts[0] == "_components" and parts[2] in {"U", "V"}:
                dotted_path = parts[1].replace("-", ".")
                new_key = f"target_model.{dotted_path}.components.{parts[2]}"
        # replace if modified
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)


def _set_submodule_by_path(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    """Set `root.<module_path>` to `new_module`, supporting ModuleList/Sequential numeric segments."""
    if module_path == "":
        raise ValueError("module_path cannot be empty")
    parts = module_path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = _get_child_module(parent, part)
    last = parts[-1]
    if last.isdigit():
        # Parent is a container type (ModuleList/Sequential) that supports indexing
        parent[int(last)] = new_module  # pyright: ignore[reportIndexIssue]
    else:
        setattr(parent, last, new_module)


def _get_child_module(parent: nn.Module, part: str) -> nn.Module:
    if part.isdigit():
        # Parent is a container type (ModuleList/Sequential) that supports indexing
        return parent[int(part)]  # pyright: ignore[reportIndexIssue]
    return parent.get_submodule(part)
