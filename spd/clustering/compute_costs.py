from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.clustering.math.merge_matrix import GroupMerge


def compute_merge_costs(
    coact: Bool[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    alpha: float = 1.0,
    rank_cost: Callable[[float], float] = lambda _: 1.0,
) -> Float[Tensor, "k_groups k_groups"]:
    r"""Compute MDL costs for merge matrices

    $$
        F(P_i, P_j)
        = \alpha |s_i| r(P_j) + \alpha |s_j| r(P_i)
        - s_i s_j ( \alpha r(P_i) + \alpha r(P_j) + c )
        = \alpha (
            |s_i| r(P_j)
            + |s_j| r(P_i)
            - s_i s_j ( r(P_i) + r(P_j) + c/\alpha )
        )
    $$

    """
    device: torch.device = coact.device
    ranks: Float[Tensor, " k_groups"] = merges.components_per_group.to(device=device).float()
    s_diag: Float[Tensor, " k_groups"] = torch.diag(coact).to(device=device)
    # dbg_auto(ranks)
    # dbg_auto(diag)
    # dbg_auto(coact)
    # dbg_auto(diag @ ranks.unsqueeze(1))
    # dbg_auto(ranks @ diag.unsqueeze(1))
    # dbg_auto(ranks.unsqueeze(0) + ranks.unsqueeze(1))
    term_sipj: Float[Tensor, "k_groups k_groups"] = s_diag.view(-1, 1) * ranks.view(1, -1)
    rank_sum: Float[Tensor, "k_groups k_groups"] = ranks.view(-1, 1) + ranks.view(1, -1)
    # TODO: use dynamic rank computation
    return alpha * (
        term_sipj + term_sipj.T - (rank_sum + (rank_cost(merges.k_groups) / alpha)) * coact
    )


def recompute_coacts_merge_pair(
    coact: Float[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    merge_pair: tuple[int, int],
    activation_mask: Bool[Tensor, "samples k_groups"],
) -> tuple[
    GroupMerge,
    Float[Tensor, "k_groups-1 k_groups-1"],
    Bool[Tensor, "samples k_groups"],
]:
    # check shape
    k_groups: int = coact.shape[0]
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"

    # activations of the new merged group
    activation_mask_grp: Bool[Tensor, " samples"] = (
        activation_mask[:, merge_pair[0]] + activation_mask[:, merge_pair[1]]
    )

    # coactivations with the new merged group
    # dbg_tensor(activation_mask_grp)
    # dbg_tensor(activation_mask)
    coact_with_merge: Float[Tensor, " k_groups"] = (
        activation_mask_grp.float() @ activation_mask.float()
    )
    new_group_idx: int = min(merge_pair)
    remove_idx: int = max(merge_pair)
    new_group_self_coact: float = activation_mask_grp.float().sum().item()
    # dbg_tensor(coact_with_merge)

    # assemble the merge pair
    merge_new: GroupMerge = merges.merge_groups(
        merge_pair[0],
        merge_pair[1],
    )
    # `merge_groups` will set `old_to_new_idx` to be an actual dict for `merge_new`
    old_to_new_idx: dict[int | None, int | None] = merge_new.old_to_new_idx  # pyright: ignore[reportAssignmentType]
    assert old_to_new_idx[None] == new_group_idx, (
        "New group index should be the minimum of the merge pair"
    )
    assert old_to_new_idx[new_group_idx] is None
    assert old_to_new_idx[remove_idx] is None
    # TODO: check that the rest are in order? probably not necessary

    # reindex coactivations
    coact_temp: Float[Tensor, "k_groups k_groups"] = coact.clone()
    # add in the similarities with the new group
    coact_temp[new_group_idx, :] = coact_with_merge
    coact_temp[:, new_group_idx] = coact_with_merge
    # delete the old group
    mask: Bool[Tensor, " k_groups"] = torch.ones(
        coact_temp.shape[0], dtype=torch.bool, device=coact_temp.device
    )
    mask[remove_idx] = False
    coact_new: Float[Tensor, "k_groups-1 k_groups-1"] = coact_temp[mask, :][:, mask]
    # add in the self-coactivation of the new group
    coact_new[new_group_idx, new_group_idx] = new_group_self_coact
    # dbg_tensor(coact_new)

    # reindex mask
    activation_mask_new: Float[Tensor, "samples ..."] = activation_mask.clone()
    # add in the new group
    activation_mask_new[:, new_group_idx] = activation_mask_grp
    # remove the old group
    activation_mask_new = activation_mask_new[:, mask]

    # dbg_tensor(activation_mask_new)

    return (
        merge_new,
        coact_new,
        activation_mask_new,
    )


def recompute_coacts_pop_group(
    coact: Float[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    component_idx: int,
    activation_mask: Bool[Tensor, "n_samples k_groups"],
    activation_mask_orig: Bool[Tensor, "n_samples n_components"],
) -> tuple[
    GroupMerge,
    Float[Tensor, "k_groups+1 k_groups+1"],
    Bool[Tensor, "n_samples k_groups+1"],
]:
    # sanity check dims
    # ==================================================
    # dbg_tensor(coact)
    # dbg_tensor(activation_mask)
    # dbg_tensor(activation_mask_orig)

    k_groups: int = coact.shape[0]
    n_samples: int = activation_mask.shape[0]
    # n_components: int = activation_mask_orig.shape[1]
    k_groups_new: int = k_groups + 1
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"
    assert activation_mask.shape[1] == k_groups, (
        "Activation mask must match coactivation matrix shape"
    )
    assert n_samples == activation_mask_orig.shape[0], (
        "Activation mask original must match number of samples"
    )

    # get the activations we need
    # ==================================================
    # which group does the component belong to?
    group_idx: int = int(merges.group_idxs[component_idx].item())
    group_size_old: int = int(merges.components_per_group[group_idx].item())
    group_size_new: int = group_size_old - 1

    # activations of component we are popping out
    acts_pop: Bool[Tensor, " samples"] = activation_mask_orig[:, component_idx]

    # activations of the "remainder" -- everything other than the component we are popping out,
    # in the group we're popping it out of
    # dims: dict[str, int] = dict(
    # 	n_samples=n_samples,
    # 	k_groups=k_groups,
    # 	k_groups_new=k_groups_new,
    # 	group_size_old=group_size_old,
    # 	group_size_new=group_size_new,
    # 	n_components=n_components,
    # )
    # dbg_auto(dims)
    # dbg_tensor(acts_pop)
    # dbg_tensor(merges.components_in_group(group_idx))

    acts_remainder: Bool[Tensor, " samples"] = (
        activation_mask_orig[
            :, [i for i in merges.components_in_group(group_idx) if i != component_idx]
        ]
        .max(dim=-1)
        .values
    )

    # assemble the new activation mask
    # ==================================================
    # first concat the popped-out component onto the end
    activation_mask_new: Bool[Tensor, " samples k_groups+1"] = torch.cat(
        [activation_mask, acts_pop.unsqueeze(1)],
        dim=1,
    )
    # then replace the group we are popping out of with the remainder
    activation_mask_new[:, group_idx] = acts_remainder

    # dbg_tensor(acts_remainder)
    # dbg_tensor(activation_mask_new)

    # assemble the new coactivation matrix
    # ==================================================
    coact_new: Float[Tensor, "k_groups+1 k_groups+1"] = torch.full(
        (k_groups_new, k_groups_new),
        fill_value=float("nan"),
        dtype=coact.dtype,
        device=coact.device,
    )
    # copy in the old coactivation matrix
    coact_new[:k_groups, :k_groups] = coact.clone()
    # compute new coactivations we need
    coact_pop: Float[Tensor, " k_groups"] = acts_pop.float() @ activation_mask_new.float()
    coact_remainder: Float[Tensor, " k_groups"] = (
        acts_remainder.float() @ activation_mask_new.float()
    )

    # dbg_tensor(coact)
    # dbg_tensor(coact_new)
    # dbg_tensor(coact_pop)
    # dbg_tensor(coact_remainder)

    # replace the relevant rows and columns
    coact_new[group_idx, :] = coact_remainder
    coact_new[:, group_idx] = coact_remainder
    coact_new[-1, :] = coact_pop
    coact_new[:, -1] = coact_pop

    # assemble the new group merge
    # ==================================================
    group_idxs_new: Int[Tensor, " k_groups+1"] = merges.group_idxs.clone()
    # the popped-out component is now its own group
    new_group_idx: int = k_groups_new - 1
    group_idxs_new[component_idx] = new_group_idx
    merge_new: GroupMerge = GroupMerge(
        group_idxs=group_idxs_new,
        k_groups=k_groups_new,
    )

    # sanity check
    assert merge_new.components_per_group.shape == (k_groups_new,), (
        "New merge must have k_groups+1 components"
    )
    assert merge_new.components_per_group[new_group_idx] == 1, (
        "New group must have exactly one component"
    )
    assert merge_new.components_per_group[group_idx] == group_size_new, (
        "Old group must have one less component"
    )

    # return
    # ==================================================
    return (
        merge_new,
        coact_new,
        activation_mask_new,
    )
