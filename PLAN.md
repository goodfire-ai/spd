# Multi-step PPGD Implementation Plan

## Problem

Standard PGD (20 inner steps, step_size=0.1) converges nicely to PGDReconLoss ~0.5-1 but is ~2.5x slower per step. Persistent PGD (1 Adam step per training step) is fast but the adversary lags behind the moving model — eval PGDReconLoss stays wild and high (~3.0).

## Approach

Add K extra source-only warmup steps per training step. These reuse the already-computed target output and CI (the expensive parts), running only cheap component model forward passes to update PPGD sources before they're used for the "real" loss computation. Sources get K+1 total updates per step (K warmup + 1 from the existing flow).

## Current flow (1 PPGD step per training step)

```
for each training step:
    weight_deltas = model.calc_weight_deltas()
    ppgd_batch_grads = zeros

    for each microbatch:
        target_out, ci = forward(microbatch)           # expensive
        sources = ppgd_state.get_effective_sources()
        losses = compute_losses(..., sources)           # includes PPGD forward
        ppgd_grads = autograd.grad(ppgd_loss, sources)
        ppgd_batch_grads += ppgd_grads
        total_loss.backward()                           # model param gradients

    ppgd_state.step(ppgd_batch_grads)                   # 1 source update
    optimizer.step()
```

## Proposed flow (K+1 PPGD steps per training step)

```
for each training step:
    weight_deltas = model.calc_weight_deltas()
    ppgd_batch_grads = zeros

    for each microbatch:
        target_out, ci = forward(microbatch)           # expensive (done once)

        # NEW: K warmup steps with detached model state
        detached_ci = {k: v.detach() for ...}
        detached_target = target_out.detach()
        detached_wd = {k: v.detach() for ...}
        for _ in range(K):
            warmup_loss = persistent_pgd_recon_loss(
                ..., detached_ci, detached_target, detached_wd)
            warmup_grads = get_grads(warmup_loss, retain_graph=False)
            ppgd_state.step(warmup_grads)

        # Existing: compute all losses with warmed-up sources
        losses = compute_losses(..., ppgd_state.get_effective_sources())
        ppgd_grads = autograd.grad(ppgd_loss, sources, retain_graph=True)
        ppgd_batch_grads += ppgd_grads
        total_loss.backward()

    ppgd_state.step(ppgd_batch_grads)
    optimizer.step()
```

## Changes

### 1. Config (`spd/configs.py`)

Add `n_warmup_steps: int = 0` to both PPGD config classes:

```python
class PersistentPGDReconLossConfig(LossMetricConfig):
    classname: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"
    optimizer: Annotated[PGDOptimizerConfig, Field(discriminator="type")]
    scope: PersistentPGDSourceScope
    use_sigmoid_parameterization: bool = False
    n_warmup_steps: int = 0       # NEW

class PersistentPGDReconSubsetLossConfig(LossMetricConfig):
    classname: Literal["PersistentPGDReconSubsetLoss"] = "PersistentPGDReconSubsetLoss"
    optimizer: Annotated[PGDOptimizerConfig, Field(discriminator="type")]
    scope: PersistentPGDSourceScope
    use_sigmoid_parameterization: bool = False
    routing: ...
    n_warmup_steps: int = 0       # NEW
```

### 2. `PersistentPGDState.get_grads()` (`spd/persistent_pgd.py`)

Add `retain_graph` parameter. Currently always `True` because the loss tensor is reused in `.backward()`. Warmup steps don't call `.backward()`, so `False` saves memory.

```python
def get_grads(
    self, loss: Float[Tensor, ""], r1_coeff: float = 0.0, retain_graph: bool = True
) -> tuple[PPGDSources, float]:
    grads = torch.autograd.grad(
        loss, source_values, retain_graph=retain_graph, create_graph=use_r1
    )
    ...
```

### 3. Training loop (`spd/run_spd.py`)

Add warmup loop inside the microbatch loop, between CI computation (line ~292) and `compute_losses()` (line ~294).

```python
# After target_model_output and ci, before compute_losses:
for ppgd_cfg in persistent_pgd_configs:
    if ppgd_cfg.n_warmup_steps == 0:
        continue
    detached_ci = {k: v.detach() for k, v in ci.lower_leaky.items()}
    detached_target = target_model_output.output.detach()
    detached_wd = (
        {k: v.detach() for k, v in weight_deltas.items()}
        if config.use_delta_component else None
    )
    for _ in range(ppgd_cfg.n_warmup_steps):
        with bf16_autocast(enabled=config.autocast_bf16):
            match ppgd_cfg:
                case PersistentPGDReconLossConfig():
                    warmup_loss = persistent_pgd_recon_loss(
                        model=component_model,
                        ppgd_sources=ppgd_states[ppgd_cfg].get_effective_sources(),
                        output_loss_type=config.output_loss_type,
                        batch=microbatch,
                        target_out=detached_target,
                        ci=detached_ci,
                        weight_deltas=detached_wd,
                    )
                case PersistentPGDReconSubsetLossConfig():
                    warmup_loss = persistent_pgd_recon_subset_loss(
                        model=component_model,
                        ppgd_sources=ppgd_states[ppgd_cfg].get_effective_sources(),
                        output_loss_type=config.output_loss_type,
                        batch=microbatch,
                        target_out=detached_target,
                        ci=detached_ci,
                        weight_deltas=detached_wd,
                        routing=ppgd_cfg.routing,
                    )
        warmup_grads, _ = ppgd_states[ppgd_cfg].get_grads(
            warmup_loss, r1_coeff=ppgd_cfg.r1_coeff, retain_graph=False
        )
        ppgd_states[ppgd_cfg].step(warmup_grads)
```

### 4. YAML config

```yaml
- coeff: 0.5
  classname: PersistentPGDReconLoss
  n_warmup_steps: 4              # 4 extra source updates before the real step
  optimizer:
    type: adam
    lr: 0.01
    beta1: 0.8
    beta2: 0.99
    eps: 1.0e-08
  scope:
    type: repeat_across_batch
    n_sources: 8
  use_sigmoid_parameterization: false
  r1_coeff: 1
```

## Design details

### Why detaching ci/target_out/weight_deltas matters

Without detaching, each warmup forward pass builds a graph connecting sources AND model parameters to the loss. This means:
- The graph holds all model intermediates — memory blowup proportional to K
- `autograd.grad(loss, sources)` traverses the full graph unnecessarily

With detaching, the graph is only: sources → mask interpolation → component model forward → loss. Model weights participate in the forward pass but aren't tracked for gradients, so the graph is small and freed after each warmup step.

### Why `component_model` (not DDP `wrapped_model`) is correct for warmup

- Warmup doesn't compute model parameter gradients (no `.backward()`)
- PPGD source gradients are already all-reduced in `get_grads()` via `all_reduce(g, op=ReduceOp.SUM)`
- DDP wrapping is only needed for `.backward()` to sync param gradients

### Cost estimate

Each warmup step: 1 component model forward pass + `autograd.grad` through a small graph. The expensive parts (target model forward, CI computation, `.backward()` for model params) stay at 1x. For K=4 on a 2L model, expect ~20-40% wall time overhead vs ~20x for full 20-step PGD.
