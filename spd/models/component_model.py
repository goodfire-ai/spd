from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, overload, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from spd.configs import Config, SamplingType
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.batch_and_loss_fns import RunBatch, make_run_batch
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


class ComponentModel(nn.Module):
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
        run_batch: RunBatch,
        module_path_info: list[ModulePathInfo],
        ci_fn_type: CiFnType,
        ci_fn_hidden_dims: list[int],
        sigmoid_type: SigmoidType,
    ):
        super().__init__()
        self._run_batch: RunBatch = run_batch

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        self.target_model = target_model
        self.module_to_c = {info.module_path: info.C for info in module_path_info}
        self.target_module_paths = list(self.module_to_c.keys())

        self.components = ComponentModel._create_components(
            target_model=target_model,
            module_to_c=self.module_to_c,
        )
        self._components = nn.ModuleDict(
            {k.replace(".", "-"): self.components[k] for k in sorted(self.components)}
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

        if sigmoid_type == "leaky_hard":
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
        else:
            # For other sigmoid types, use the same function for both
            self.lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
            self.upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

    @classmethod
    def from_run_info(cls, run_info: RunInfo[Config]) -> "ComponentModel":
        """Load a trained ComponentModel from a run info object."""
        config = run_info.config

        model_class = resolve_class(config.pretrained_model_class)
        if config.pretrained_model_name is not None:
            assert hasattr(model_class, "from_pretrained")
            target_model = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            assert issubclass(model_class, LoadableModule)
            assert config.pretrained_model_path is not None
            target_model = model_class.from_pretrained(config.pretrained_model_path)

        target_model.eval()
        target_model.requires_grad_(False)

        if config.identity_module_info is not None:
            insert_identity_operations_(
                target_model,
                identity_module_info=config.identity_module_info,
            )

        module_path_info = expand_module_patterns(target_model, config.all_module_info)

        run_batch = make_run_batch(config.output_extract)

        comp_model = ComponentModel(
            target_model=target_model,
            run_batch=run_batch,
            module_path_info=module_path_info,
            ci_fn_hidden_dims=config.ci_fn_hidden_dims,
            ci_fn_type=config.ci_fn_type,
            sigmoid_type=config.sigmoid_type,
        )

        weights = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=True)
        handle_deprecated_state_dict_keys_(weights)
        comp_model.load_state_dict(weights)
        return comp_model

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> "ComponentModel":
        """Load a trained ComponentModel from a wandb or local path."""
        run_info = SPDRunInfo.from_path(path)
        return cls.from_run_info(run_info)

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

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["component_acts"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["input"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
    ) -> Tensor: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input", "none"] = "none",
    ) -> Tensor | OutputWithCache:
        """Forward pass with optional component replacement and/or input caching.

        This method handles the following 4 cases:
        1. mask_infos is None and cache_type is "none": Regular forward pass.
        2. mask_infos is None and cache_type is "input" or "component_acts": Forward pass with
            caching on all modules in self.target_module_paths.
        3. mask_infos is not None and cache_type is "input" or "component_acts": Forward pass with
            component replacement and caching on the modules provided in mask_infos.
        4. mask_infos is not None and cache_type is "none": Forward pass with component replacement
            on the modules provided in mask_infos and no caching.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo.
                If provided, those modules will be replaced with their components.
            cache_type: If "input" or "component_acts", cache the inputs or component acts to the
                modules provided in mask_infos. If "none", no caching is done. If mask_infos is None,
                cache the inputs or component acts to all modules in self.target_module_paths.

        Returns:
            OutputWithCache object if cache_type is "input" or "component_acts", otherwise the
            model output tensor.
        """
        if mask_infos is None and cache_type == "none":
            return self._run_batch(self.target_model, batch)

        cache: dict[str, Tensor] = {}
        hooks: dict[str, Callable[..., Any]] = {}

        hook_module_names = list(mask_infos.keys()) if mask_infos else self.target_module_paths

        for module_name in hook_module_names:
            mask_info = mask_infos[module_name] if mask_infos else None
            components = self.components[module_name] if mask_info else None

            hooks[module_name] = partial(
                self._components_and_cache_hook,
                module_name=module_name,
                components=components,
                mask_info=mask_info,
                cache_type=cache_type,
                cache=cache,
            )

        with self._attach_forward_hooks(hooks):
            out: Tensor = self._run_batch(self.target_model, batch)

        match cache_type:
            case "input" | "component_acts":
                return OutputWithCache(output=out, cache=cache)
            case "none":
                return out

    def _components_and_cache_hook(
        self,
        _module: nn.Module,
        args: list[Any],
        kwargs: dict[Any, Any],
        output: Any,
        module_name: str,
        components: Components | None,
        mask_info: ComponentsMaskInfo | None,
        cache_type: Literal["component_acts", "input", "none"],
        cache: dict[str, Tensor],
    ) -> Any | None:
        """Unified hook function that handles both component replacement and caching.

        Args:
            module: The module being hooked
            args: Module forward args
            kwargs: Module forward kwargs
            output: Module forward output
            module_name: Name of the module in the target model
            components: Component replacement (if using components)
            mask_info: Mask information (if using components)
            cache_type: Whether to cache the component acts, input, or none
            cache: Cache dictionary to populate (if cache_type is not None)

        Returns:
            If using components: modified output (or None to keep original)
            If not using components: None (keeps original output)
        """
        assert len(args) == 1, "Expected 1 argument"
        assert len(kwargs) == 0, "Expected no keyword arguments"
        x = args[0]
        assert isinstance(x, Tensor), "Expected input tensor"

        if cache_type == "input":
            cache[module_name] = x

        if components is not None and mask_info is not None:
            assert isinstance(output, Tensor), (
                f"Only supports single-tensor outputs, got {type(output)}"
            )

            component_acts_cache = {} if cache_type == "component_acts" else None
            components_out = components(
                x,
                mask=mask_info.component_mask,
                weight_delta_and_mask=mask_info.weight_delta_and_mask,
                component_acts_cache=component_acts_cache,
            )
            if component_acts_cache is not None:
                for k, v in component_acts_cache.items():
                    cache[f"{module_name}_{k}"] = v

            if mask_info.routing_mask == "all":
                return components_out

            return torch.where(mask_info.routing_mask[..., None], components_out, output)

        # No component replacement - keep original output
        return None

    @contextmanager
    def _attach_forward_hooks(self, hooks: dict[str, Callable[..., Any]]) -> Generator[None]:
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
                    ci_fn_input = self.components[target_module_name].get_component_acts(
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

    def get_all_component_acts(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        """Compute component activations (v_i^T @ x) for all layers.

        Args:
            pre_weight_acts: Dict mapping layer name to input activations.

        Returns:
            Dict mapping layer name to component activations tensor.
        """
        return {
            layer: self.components[layer].get_component_acts(acts)
            for layer, acts in pre_weight_acts.items()
            if layer in self.components
        }

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
        # replace if modified
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)
