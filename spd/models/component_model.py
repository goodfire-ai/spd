from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, overload, override

import torch
import wandb
import yaml
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as RadfordConv1D
from wandb.apis.public import Run

from spd.configs import Config, SamplingType
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.components import (
    CiFnType,
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
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.general_utils import fetch_latest_local_checkpoint, resolve_class, runtime_cast
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
    match the patterns you pass in `target_module_patterns`.

    Forward passes support optional component replacement and/or input caching:
    - No args: Standard forward pass of the target model
    - With mask_infos: Components replace the specified modules via forward hooks
    - With cache_type="input": Input activations are cached for the specified modules
    - Both can be used simultaneously for component forward pass with input caching

    We register components and causal importance functions (ci_fns) as modules in this class in order to have them update
    correctly when the model is wrapped in a `DistributedDataParallel` wrapper (and for other
    conveniences).
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_module_patterns: list[str],
        C: int,
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
        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr
        self.target_module_paths = get_target_module_paths(target_model, target_module_patterns)

        self.components = ComponentModel._create_components(
            target_model=target_model,
            target_module_paths=self.target_module_paths,
            C=C,
        )
        self._components = nn.ModuleDict(
            {k.replace(".", "-"): self.components[k] for k in sorted(self.components)}
        )

        self.ci_fns = ComponentModel._create_ci_fns(
            target_model=target_model,
            target_module_paths=self.target_module_paths,
            C=C,
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
        target_module_paths: list[str],
        C: int,
    ) -> dict[str, Components]:
        components: dict[str, Components] = {}
        for target_module_path in target_module_paths:
            target_module = target_model.get_submodule(target_module_path)
            components[target_module_path] = ComponentModel._create_component(target_module, C)
        return components

    @staticmethod
    def _create_ci_fn(
        target_module: nn.Module,
        component_C: int,
        ci_fn_type: CiFnType,
        ci_fn_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a causal importance function (ci_fn) based on ci_fn_type and module type."""
        if isinstance(target_module, nn.Embedding):
            assert ci_fn_type == "mlp", "Embedding modules only supported for ci_fn_type='mlp'"

        if ci_fn_type == "mlp":
            return MLPCiFn(C=component_C, hidden_dims=ci_fn_hidden_dims)

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
                return VectorMLPCiFn(
                    C=component_C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims
                )
            case "shared_mlp":
                return VectorSharedMLPCiFn(
                    C=component_C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims
                )

    @staticmethod
    def _create_ci_fns(
        target_model: nn.Module,
        target_module_paths: list[str],
        C: int,
        ci_fn_type: CiFnType,
        ci_fn_hidden_dims: list[int],
    ) -> dict[str, nn.Module]:
        ci_fns: dict[str, nn.Module] = {}
        for target_module_path in target_module_paths:
            target_module = target_model.get_submodule(target_module_path)
            ci_fns[target_module_path] = ComponentModel._create_ci_fn(
                target_module,
                C,
                ci_fn_type,
                ci_fn_hidden_dims,
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
        cache_type: Literal["input", "output"],
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
        cache_type: Literal["input", "none", "output"] = "none",
        **kwargs: Any,
    ) -> Tensor | OutputWithCache:
        """Forward pass with optional component replacement and/or input caching.

        This method handles the following 4 cases:
        1. mask_infos is None and cache_type is "none": Regular forward pass.
        2. mask_infos is None and cache_type is "input": Forward pass with input caching on
            all modules in self.target_module_paths.
        3. mask_infos is not None and cache_type is "input": Forward pass with component replacement
            and input caching on the modules provided in mask_infos.
        4. mask_infos is not None and cache_type is "none": Forward pass with component replacement
            on the modules provided in mask_infos and no caching.

        We use the same _components_and_cache_hook for cases 2, 3, and 4, and don't use any hooks
        for case 1.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo.
                If provided, those modules will be replaced with their components.
            cache_type: If "input", cache the inputs to the modules provided in mask_infos. If
                mask_infos is None, cache the inputs to all modules in self.target_module_paths.

        Returns:
            OutputWithCache object if cache_type is "input", otherwise the model output tensor.
        """
        if mask_infos is None and cache_type == "none":
            # No hooks needed. Do a regular forward pass of the target model.
            return self._extract_output(self.target_model(*args, **kwargs))

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
            raw_out = self.target_model(*args, **kwargs)

        out = self._extract_output(raw_out)
        match cache_type:
            case "input":
                return OutputWithCache(output=out, cache=cache)
            case "none":
                return out
            case "output":
                return OutputWithCache(output=out, cache=cache)

    def _components_and_cache_hook(
        self,
        _module: nn.Module,
        args: list[Any],
        kwargs: dict[Any, Any],
        output: Any,
        module_name: str,
        components: Components | None,
        mask_info: ComponentsMaskInfo | None,
        cache_type: Literal["input", "none", "output"],
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
            cache_type: Whether to cache the input
            cache: Cache dictionary to populate (if cache_type is not None)

        Returns:
            If using components: modified output (or None to keep original)
            If not using components: None (keeps original output)
        """
        assert len(args) == 1, "Expected 1 argument"
        assert len(kwargs) == 0, "Expected no keyword arguments"
        x = args[0]
        assert isinstance(x, Tensor), "Expected input tensor"
        assert cache_type in ["input", "none", "output"], "Expected cache_type to be 'input', 'none', or 'output'"

        if components is not None and mask_info is not None:
            assert isinstance(output, Tensor), (
                f"Only supports single-tensor outputs, got {type(output)}"
            )

            components_out = components(
                x,
                mask=mask_info.component_mask,
                weight_delta_and_mask=mask_info.weight_delta_and_mask,
            )

            if mask_info.routing_mask == "all":
                output = components_out
            else:
                output = torch.where(mask_info.routing_mask[..., None], components_out, output)

        if cache_type == "input":
            cache[module_name] = x

        if cache_type == "output":
            cache[module_name] = output

        return output

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
            target_module_patterns=config.all_module_patterns,
            C=config.C,
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

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, " d_out d_in"]]:
        """Calculate the weight differences between the target and component weights (V@U) for each layer."""
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas

    @contextmanager
    def cache_modules(self, module_paths: list[str]) -> Generator[tuple[dict[str, Tensor], dict[str, Tensor]]]:
        """Context manager to cache the inputs to the specified modules."""
        input_cache: dict[str, Tensor] = {}
        output_cache: dict[str, Tensor] = {}

        def _make_hook(module_path: str) -> Callable[[nn.Module, tuple[Any, ...], dict[Any, Any], Any], Any]:
            def _cache_hook(
                module: nn.Module, args: tuple[Any, ...], kwargs: dict[Any, Any], output: Any
            ) -> Any:
                """Hook to cache the input to the module."""
                assert len(args) > 0, f"Expected at least 1 argument for module {module} path {module_path}"
                input_cache[module_path] = args[0]
                output_cache[module_path] = output
                return output

            return _cache_hook

        handles: list[RemovableHandle] = []
        for module_path in module_paths:
            module = self.target_model.get_submodule(module_path)
            handle = module.register_forward_hook(_make_hook(module_path), with_kwargs=True)
            handles.append(handle)
        try:
            yield input_cache, output_cache
        finally:
            for handle in handles:
                handle.remove()


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
