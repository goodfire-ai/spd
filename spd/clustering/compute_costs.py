import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.clustering.math.merge_matrix import GroupMerge


def compute_mdl_cost(
    acts: Float[Tensor, " k_groups"],
    merges: GroupMerge,
    alpha: float = 1.0,
) -> float:
    r"""Compute MDL costs for merge matrices

    $$
        MDL = \sum_{i \in \N_k} s_i ( \log(k) + \alpha r(P_i) )
    $$

    where:
     - $s_i$ activation of component $i$, $s_j$ activation of component $j$
     - $r(P_i)$ rank of component $i$, $r(P_j)$ rank of component $j$
     - $k$ is the total number of components
    """

    k_groups: int = acts.shape[0]
    assert k_groups == merges.k_groups, "Merges must match activation vector shape"

    return (
        (acts * (math.log2(k_groups) + alpha * merges.components_per_group.to(device=acts.device)))
        .sum()
        .item()
    )


def compute_merge_costs(
    coact: Float[Tensor, "k_groups k_groups"],
    merges: GroupMerge,
    alpha: float = 1.0,
) -> Float[Tensor, "k_groups k_groups"]:
    r"""Compute MDL costs for merge matrices

    $$
        F(P_i, P_j)
        = \alpha |s_i| r(P_i) + \alpha |s_j| r(P_j)
            - s_i s_j ( \alpha r(P_i) + \alpha r(P_j) + c )
        = \alpha (
            |s_i| r(P_i)
            + |s_j| r(P_j)
            - s_i s_j ( r(P_i) + r(P_j) + c/\alpha )
        )
    $$

    new version from nathu 2025-08-11 16:48

    $$
        (s_\Sigma - s_i - s_j) log((c-1)/c)
        + s_{i,j} log(c-1) - s_i log(c) - s_j log(c)
        + alpha ( s_{i,j} r(P_{i,j}) - s_i r(P_i) - s_j r(P_j) )
    $$
    where:
     - $s_\Sigma$ average activation of all components
     - $s_i$ activation of component $i$, $s_j$ activation of component $j$
     - $s_{i,j}$ activation of the merged component $i,j$
     - $r(P_i)$ rank of component $i$, $r(P_j)$ rank of component $j$
     - $r(P_{i,j})$ rank of the merged component $i,j$

    """
    k_groups: int = coact.shape[0]
    assert coact.shape[1] == k_groups, "Coactivation matrix must be square"
    assert merges.k_groups == k_groups, "Merges must match coactivation matrix shape"

    device: torch.device = coact.device
    ranks: Float[Tensor, " k_groups"] = merges.components_per_group.to(device=device).float()
    s_diag: Float[Tensor, " k_groups"] = torch.diag(coact).to(device=device)
    # term_si_rpj: Float[Tensor, "k_groups k_groups"] = s_diag.view(-1, 1) * ranks.view(1, -1)
    # term_si_rpj: Float[Tensor, "k_groups k_groups"] = s_diag.view(-1, 1) * (ranks.view(1, -1) + 1/alpha)
    term_si_rpi: Float[Tensor, " k_groups"] = s_diag * ranks
    # dbg_auto(term_si_rpi)
    rank_sum: Float[Tensor, "k_groups k_groups"] = ranks.view(-1, 1) + ranks.view(1, -1)
    # TODO: use dynamic rank computation
    # return alpha * (
    #     term_si_rpj  # |s_i| r(P_j)
    #     + term_si_rpj.T  # |s_j| r(P_i)
    #     - coact * ( # s_i s_j
    #         rank_sum  # r(P_i) + r(P_j)
    #         + (rank_cost(merges.k_groups) / alpha) # c / alpha
    #     )
    # )

    coact_OR: Float[Tensor, "k_groups k_groups"] = s_diag.view(-1, 1) + s_diag.view(1, -1) - coact

    # reduce penalty for sending dictionary by 1
    # (s_\Sigma - s_i - s_j) log((c-1)/c)
    # delta of cost for sending index, in expectation
    # + s_{i,j} log(c-1) - s_i log(c) - s_j log(c)
    # delta of cost for sending ranks, in expectation
    # + alpha ( s_{i,j} r(P_{i,j}) - s_i r(P_i) - s_j r(P_j)

    s_other: Float[Tensor, "k_groups k_groups"] = (
        s_diag.sum() - s_diag.view(-1, 1) - s_diag.view(1, -1)
    ) * math.log2((k_groups - 1) / k_groups)

    bits_local: Float[Tensor, "k_groups k_groups"] = (
        coact_OR * math.log2(k_groups - 1)
        - s_diag.view(-1, 1) * math.log2(k_groups)
        - s_diag.view(1, -1) * math.log2(k_groups)
    )

    penalty: Float[Tensor, "k_groups k_groups"] = (
        coact_OR * rank_sum  # s_{i,j} r(P_{i,j})
        - term_si_rpi.view(-1, 1)  # s_i r(P_i)
        - term_si_rpi.view(1, -1)  # s_j r(P_j)
    )

    output: Float[Tensor, "k_groups k_groups"] = s_other + bits_local + alpha * penalty
    return output


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
    coact_with_merge: Float[Tensor, " k_groups"] = (
        activation_mask_grp.float() @ activation_mask.float()
    )
    new_group_idx: int = min(merge_pair)
    remove_idx: int = max(merge_pair)
    new_group_self_coact: float = activation_mask_grp.float().sum().item()

    # assemble the merge pair
    merge_new: GroupMerge = merges.merge_groups(
        merge_pair[0],
        merge_pair[1],
    )
    # TODO: we don't use this index for anything, and could reconstruct it from the merge pair if needed. get rid of it
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

    # reindex mask
    activation_mask_new: Float[Tensor, "samples ..."] = activation_mask.clone()
    # add in the new group
    activation_mask_new[:, new_group_idx] = activation_mask_grp
    # remove the old group
    activation_mask_new = activation_mask_new[:, mask]

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

    k_groups: int = coact.shape[0]
    n_samples: int = activation_mask.shape[0]
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
