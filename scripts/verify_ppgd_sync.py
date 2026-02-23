"""Verify PersistentPGD source initialization and gradient sync across ranks.

Run: torchrun --standalone --nproc_per_node=2 scripts/verify_ppgd_sync.py
"""

import torch
from torch.distributed import ReduceOp

from spd.configs import (
    PerBatchPerPositionScope,
    PersistentPGDReconLossConfig,
    RepeatAcrossBatchScope,
    ScheduleConfig,
    SignPGDConfig,
)
from spd.persistent_pgd import PersistentPGDState
from spd.utils.distributed_utils import (
    all_reduce,
    cleanup_distributed,
    gather_all_tensors,
    get_distributed_state,
    init_distributed,
)


def make_cfg(scope: RepeatAcrossBatchScope | PerBatchPerPositionScope):
    return PersistentPGDReconLossConfig(
        optimizer=SignPGDConfig(
            lr_schedule=ScheduleConfig(start_val=0.1, fn_type="constant"),
        ),
        scope=scope,
        coeff=1.0,
    )


def main():
    init_distributed()
    state = get_distributed_state()
    assert state is not None
    rank = state.rank

    torch.manual_seed(42)

    device = f"cuda:{state.local_rank}"
    module_to_c = {"mod_a": 4, "mod_b": 8}
    batch_dims = (4, 16)  # microbatch=4, seq_len=16

    # --- Test 1: RepeatAcrossBatch (sources should be identical) ---
    cfg_repeat = make_cfg(RepeatAcrossBatchScope(n_sources=2))
    ppgd_repeat = PersistentPGDState(
        module_to_c=module_to_c,
        batch_dims=batch_dims,
        device=device,
        use_delta_component=False,
        cfg=cfg_repeat,
        output_loss_type="kl",
    )

    for name, src in ppgd_repeat.sources.items():
        gathered = gather_all_tensors(src.detach())
        assert torch.allclose(gathered[0], gathered[1]), (
            f"FAIL: RepeatAcrossBatch sources differ for {name}"
        )
    print(f"[rank {rank}] PASS: RepeatAcrossBatch sources identical across ranks")

    # --- Test 2: PerBatchPerPosition (sources should differ) ---
    torch.manual_seed(42)  # Reset seed so the only difference is rank-seeding
    cfg_perbatch = make_cfg(PerBatchPerPositionScope())
    ppgd_perbatch = PersistentPGDState(
        module_to_c=module_to_c,
        batch_dims=batch_dims,
        device=device,
        use_delta_component=False,
        cfg=cfg_perbatch,
        output_loss_type="kl",
    )

    for name, src in ppgd_perbatch.sources.items():
        gathered = gather_all_tensors(src.detach())
        assert not torch.allclose(gathered[0], gathered[1]), (
            f"FAIL: PerBatchPerPosition sources are identical for {name}"
        )
    print(f"[rank {rank}] PASS: PerBatchPerPosition sources differ across ranks")

    # --- Test 3: Gradient sync behavior ---
    # RepeatAcrossBatch: manual grad + all_reduce should give same result on all ranks
    loss_repeat = sum(s.sum() for s in ppgd_repeat.sources.values())
    grads_repeat_raw = torch.autograd.grad(
        loss_repeat, list(ppgd_repeat.sources.values()), retain_graph=True
    )
    for g in grads_repeat_raw:
        g_synced = all_reduce(g.clone(), op=ReduceOp.AVG)
        gathered = gather_all_tensors(g_synced)
        assert torch.allclose(gathered[0], gathered[1]), "FAIL: synced grads differ"
    print(f"[rank {rank}] PASS: RepeatAcrossBatch grads identical after all_reduce")

    # PerBatchPerPosition: use rank-dependent loss to produce different gradients
    rank_loss = sum((s * (rank + 1)).sum() for s in ppgd_perbatch.sources.values())
    grads_perbatch = torch.autograd.grad(
        rank_loss, list(ppgd_perbatch.sources.values()), retain_graph=True
    )
    for g in grads_perbatch:
        gathered = gather_all_tensors(g.detach())
        # rank0 grads = 1.0, rank1 grads = 2.0 → should differ
        assert not torch.allclose(gathered[0], gathered[1]), (
            "FAIL: PerBatchPerPosition grads are identical (should diverge)"
        )
    print(f"[rank {rank}] PASS: PerBatchPerPosition grads differ across ranks (no sync)")

    # --- Test 4: After step, sources diverge for PerBatchPerPosition ---
    ppgd_perbatch.step(dict(zip(ppgd_perbatch.sources.keys(), grads_perbatch, strict=True)))
    for name, src in ppgd_perbatch.sources.items():
        gathered = gather_all_tensors(src.detach())
        assert not torch.allclose(gathered[0], gathered[1]), (
            f"FAIL: sources identical after step for {name}"
        )
    print(f"[rank {rank}] PASS: PerBatchPerPosition sources still differ after step")

    print(f"\n[rank {rank}] ALL TESTS PASSED")
    cleanup_distributed()


if __name__ == "__main__":
    main()
