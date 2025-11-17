# ruff: noqa: E402, I001
# %%
from jaxtyping import Float, Int
from collections import defaultdict
import torch
from typing import Any
import plotly.graph_objects as go
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.configs import Config, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import ComponentsMaskInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, runtime_cast
from torch import Tensor
from transformers import LlamaForCausalLM

from spd.utils.module_utils import get_target_module_paths

# %%

device = get_device()

# %%

RUN = "wandb:goodfire/spd/runs/9gf5ud48"

config = SPDRunInfo.from_path(RUN).config
task_config = config.task_config
assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

model = ComponentModel.from_run_info(SPDRunInfo.from_path(RUN))
model.to(device)
model.eval()

# %%

tmodel = runtime_cast(LlamaForCausalLM, model.target_model)
# %%

train_data_config = DatasetConfig(
    name=task_config.dataset_name,
    hf_tokenizer_path=config.tokenizer_name,
    split=task_config.train_data_split,
    n_ctx=task_config.max_seq_len,
    is_tokenized=task_config.is_tokenized,
    streaming=task_config.streaming,
    column_name=task_config.column_name,
    shuffle_each_epoch=task_config.shuffle_each_epoch,
    seed=None,
)


tokenizer: Any
data_loader, tokenizer = create_data_loader(
    dataset_config=train_data_config,
    batch_size=1,
    buffer_size=task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
data_loader_iter = iter(data_loader)


pgd_config = PGDReconLossConfig(
    init="random",
    step_size=0.1,
    n_steps=20,
    mask_scope="shared_across_batch",
)

# %%


def sort_layer(lname: str):
    if lname == "lm_head":
        return 10_000
    if lname == "model.norm":
        return 9_999

    assert lname.startswith("model.layers."), f"Unknown layer: {lname}"
    lname = lname.removeprefix("model.layers.")

    if "." in lname:
        layer_str, rest = lname.split(".", maxsplit=1)
        layer = int(layer_str)
        assert layer in [0, 1, 2, 3], f"Unknown layer: {lname}"
        if rest == "input_layernorm":
            return (layer * 1000) + 1
        if rest == "self_attn.q_proj":
            return (layer * 1000) + 2
        if rest == "self_attn.k_proj":
            return (layer * 1000) + 3
        if rest == "self_attn.v_proj":
            return (layer * 1000) + 4
        if rest == "self_attn.o_proj":
            return (layer * 1000) + 5
        if rest == "post_attention_layernorm":
            return (layer * 1000) + 6
        if rest == "mlp.down_proj":
            return (layer * 1000) + 7

    return (int(lname) * 1000) + 8


def visualize_seq_layer_metric(
    vals: dict[str, Tensor],
    val_name: str,
    title: str,
    x_labels: list[str],
    hover_data: dict[str, list[str]] | None = None,
    imshow_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Visualize a matrix of values, with hover data for each token.

    Args:
        vals: A dictionary of layer names to values.
        title: The title of the plot.
        hover_data: A dictionary of layer names to lists of hover data, one string per token.
        imshow_kwargs: Keyword arguments for the heatmap (e.g., colorscale, zmin, zmax).
    """
    layer_names = sorted(vals.keys(), key=sort_layer)
    matrix = torch.stack([vals[layer] for layer in layer_names])
    matrix_np = matrix.cpu().detach().numpy()

    # Prepare custom hover text if provided
    customdata = None
    hovertemplate = None
    if hover_data is not None:
        # Create a matrix of hover text strings
        hover_matrix: list[list[str]] = []
        for layer in layer_names:
            assert layer in hover_data
            hover_matrix.append(hover_data[layer])
        customdata = hover_matrix
        hovertemplate = (
            "<b>Layer:</b> %{y}<br>"
            "<b>Position:</b> %{x}<br>"
            f"<b>{val_name}:</b> %{{z:.4f}}<br>"
            "<br>%{customdata}<extra></extra>"
        )

    # Extract common imshow kwargs for plotly
    colorscale = imshow_kwargs.get("cmap", "Viridis") if imshow_kwargs else "Viridis"
    zmin = imshow_kwargs.get("vmin") if imshow_kwargs else None
    zmax = imshow_kwargs.get("vmax") if imshow_kwargs else None

    assert len(x_labels) == matrix_np.shape[1], (
        "x_labels must match the number of columns in the matrix"
    )
    assert len(layer_names) == matrix_np.shape[0], (
        "layer_names must match the number of rows in the matrix"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_np,
            y=layer_names,
            x=[f"{i}: [{x}]" for i, x in enumerate(x_labels)],
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            customdata=customdata,
            hovertemplate=hovertemplate,
            hoverongaps=False,
            showscale=False,
        )
    )

    fig.update_layout(
        title=title,
        width=12000,
        height=len(layer_names) * 40,
        yaxis=dict(title="Layer"),
        xaxis=dict(title="Token Position"),
    )

    fig.show()


# %%


def mse(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).pow(2).mean(dim=-1)


def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def l2_norm_ratio(a: Tensor, b: Tensor) -> Tensor:
    return torch.norm(a, dim=-1) / torch.norm(b, dim=-1)


def get_layer_outputs(
    batch: Int[Tensor, "1 seq"],
    paths: list[str],
    mask_infos: dict[str, ComponentsMaskInfo] | None = None,
) -> dict[str, Tensor]:
    with model.cache_modules(paths) as (_, output_cache):
        model(batch, mask_infos=mask_infos)
    layers = {}
    for path in paths:
        layers[path] = output_cache[path]
    return layers


def get_gpd_masks(model: ComponentModel, config: Config, batch: Int[Tensor, "1 seq"]):
    target_output = model(batch, cache_type="input")
    ci = model.calc_causal_importances(
        pre_weight_acts=target_output.cache,
        detach_inputs=False,
        sampling=config.sampling,
    )

    _, _, pgd_mask_infos = pgd_masked_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci.lower_leaky,
        weight_deltas=model.calc_weight_deltas(),
        target_out=target_output.output,
        output_loss_type=config.output_loss_type,
        routing="all",
        pgd_config=pgd_config,
    )

    return pgd_mask_infos


paths = get_target_module_paths(
    model.target_model,
    [
        # "model.layers.*.mlp",
        "model.layers.*.mlp.down_proj",
        # "model.layers.*.mlp.*proj",
        "model.layers.*.self_attn.o_proj",
        # "model.layers.*.*layernorm",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.layers.3",
        "model.norm",
    ],
)


def get_logits_lens_toks_and_projections(
    vectors: dict[str, Float[Tensor, "s d"]], topk: int = 10
) -> dict[str, list[list[tuple[int, float]]]]:
    assert next(iter(vectors.values())).ndim == 2, "Expected 2D tensor"
    logits: dict[str, list[list[tuple[int, float]]]] = defaultdict(list)
    for layer, seq_vec in vectors.items():
        print(layer)
        for tok_vec in seq_vec:
            top_logits = runtime_cast(Tensor, tmodel.lm_head(tok_vec)).topk(topk, dim=-1)  # pyright: ignore[reportCallIssue]  # noqa: F821
            logits[layer].append(
                list(zip(top_logits.indices.tolist(), top_logits.values.tolist(), strict=True))
            )
    return logits


# %%

batch = extract_batch_data(next(data_loader_iter)).to(device)[0:1]
actual_seq_toks = [tokenizer.decode(tok) for tok in batch[0]]

pgd_mask_infos = get_gpd_masks(model, config, batch)
target_outputs = {k: v[0] for k, v in get_layer_outputs(batch, paths).items()}
pgd_outputs = {k: v[0] for k, v in get_layer_outputs(batch, paths, pgd_mask_infos).items()}
cos_sim_val = {
    layer: cosine_similarity(target_outputs[layer], pgd_outputs[layer]) for layer in paths
}
diffs = {layer: pgd_outputs[layer] - target_outputs[layer] for layer in paths}
diff_logits = get_logits_lens_toks_and_projections(diffs, topk=5)
target_logits = get_logits_lens_toks_and_projections(target_outputs, topk=5)
pgd_logits = get_logits_lens_toks_and_projections(pgd_outputs, topk=5)

# %%

hover_data = defaultdict[str, list[str]](list)
for layer in diff_logits:
    for diff_data, target_data, pgd_data in zip(
        diff_logits[layer], target_logits[layer], pgd_logits[layer], strict=False
    ):
        lines = []
        for name, logit_lens_topk in [
            ("target", target_data),
            ("pgd", pgd_data),
            ("diff", diff_data),
        ]:
            lines.append(f"{name}:")
            for tok_idx, prob in logit_lens_topk:
                lines.append(f"{tokenizer.decode(tok_idx):<10} {prob:.2f}")
            lines.append("")
        hover_data[layer].append("<br>".join(lines))

# %%

visualize_seq_layer_metric(
    vals=cos_sim_val,
    val_name="cos_similarity",
    hover_data=hover_data,
    title="cos_similarity between pgd and target hidden states",
    imshow_kwargs={"cmap": "RdBu", "vmin": -1, "vmax": 1},
    x_labels=actual_seq_toks,
)

# for fn, title, imshow_kwargs in [
# %%
"_".join([tokenizer.decode(tok) for tok in batch[0]])
# %%
tokenizer.decode(batch[0])
# %%
