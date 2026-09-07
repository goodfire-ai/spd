"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteDims` / `SiteSpec` are the per-site shape primitives (configured name+C,
matrix dimensions, and the combined shape-carrying spec); `Factorization` (`Dense` |
`ExpertBlocked`) says how a site's V/U factor its matrix, and every consumer whose
behavior depends on the kind matches on it; `ComponentStacks` is the trainable
master pytree, grouped by target-declared semantic role; `init_component_stacks` seeds it.
These are domain-neutral — they depend only on the site shapes and the V/U arrays — so they
live here rather than inside `model.py` (whose `DecomposedModel` Protocol references
`ComponentStacks`/`SiteSpec`) or any one target. Executing a decomposed site is placement's
business, above: `decomposed_linear.site_forward`.
"""

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Generic, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from typing_extensions import TypeVar

from param_decomp.core.axes import Axes, SemanticAxis
from param_decomp.core.nonlinearity import (
    KVHeads,
    Neurons,
    NonlinearityPartition,
    QueryHeads,
)


def activation_axes(ndim: int, feature: SemanticAxis) -> Axes:
    """THE semantic axis names of a waist activation `[batch, *positions, feature]`.
    Placement lookups are exact-name; every consumer derives the tuple here so a
    misspelled feature axis (silent replication) has no second spelling to hide in.
    The waist comes in exactly TWO shapes (`model.py`) — positionless or one position
    axis — so the position vocabulary is the enumeration below, not an open family."""
    match ndim:
        case 2:
            return ("batch", feature)
        case 3:
            return ("batch", "position", feature)
        case _:
            raise AssertionError(ndim)


@dataclass(frozen=True)
class SiteC:
    """A decomposed site as configured: its torch-module-path name and its C.

    The shape-carrying `SiteSpec` is derived from this plus the target's config."""

    name: str
    C: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NarrowCI:
    """One expert-blocked site's CI (or mask), NARROW: the routed slots' scores and the
    router indices that give them meaning, travelling as ONE value. `values[.., m·c + j]`
    scores global component `router_indices[.., m]·c + j` (`c = values.shape[-1] //
    router_indices.shape[-1]`; the site's flat C = `n_experts·c`). Unrouted components'
    CI is zero by definition — structurally absent, never computed — and a narrow tensor
    separated from its routing is unrepresentable: consumers reduce, gather, and shard
    on this bundle. No production code materializes the full `[.., C]` view; the
    tiny-scale equivalence oracle builds it from the bundle inside its own test module."""

    values: Array
    router_indices: Array
    n_experts: int = field(metadata=dict(static=True))

    @property
    def c_per_expert(self) -> int:
        k_c, k = self.values.shape[-1], self.router_indices.shape[-1]
        assert k_c % k == 0, (k_c, k)
        return k_c // k

    @property
    def C(self) -> int:
        return self.n_experts * self.c_per_expert

    def map_values(self, f: "Callable[[Array], Array]") -> "NarrowCI":
        """The same routed slots under a pointwise map — indices carried through."""
        return NarrowCI(f(self.values), self.router_indices, self.n_experts)


SiteCI = Array | NarrowCI
"""One site's CI value at the CI/mask boundary: full emission is the bare `[*leading, C]`
array (dense sites); narrow emission is the `NarrowCI` bundle (expert-blocked sites).
Consumers that reduce over — or reshard — the component axis dispatch on this union;
pointwise consumers map over `values` and carry the indices through."""


def map_site_ci(f: "Callable[[Array], Array]", value: SiteCI) -> SiteCI:
    """Apply a pointwise map to one site's CI values, whichever emission it carries."""
    match value:
        case NarrowCI():
            return value.map_values(f)
        case jax.Array():
            return f(value)


def site_ci_values(value: SiteCI) -> Array:
    """One site's CI values as the bare array — the full `[.., C]` tensor, or a narrow
    site's `[.., k·c]` routed values. For consumers whose reduction is emission-exact
    on the values alone (thresholds, per-token sums: unrouted entries are exactly 0)."""
    match value:
        case NarrowCI():
            return value.values
        case jax.Array():
            return value


def site_ci_leading(value: SiteCI) -> tuple[int, ...]:
    """One site's waist leading shape, whichever emission it carries."""
    return site_ci_values(value).shape[:-1]


def narrow_component_sums(bundle: NarrowCI, data: Array) -> Array:
    """fp32 scatter-sum of per-(token, slot) `data` (shaped like `bundle.values`) onto
    the site's FULL component axis — `out[e·c + j] = Σ_{routed (n, m): ids[n,m]=e}
    data[n, m·c + j]`. Spelled as a one-hot contraction over the (possibly sharded)
    leading axes, with NO leading collapse: the ellipsis contraction is unambiguous
    where a flattening reshape is not (only the unsharded minor slot axis splits, as in
    `narrow_routed_counts`). Partial sums stay shard-local and reduce globally once, and
    the `[C]` vector exists only as this reduction's output, never as a per-token
    tensor. The contraction is an fp32 scatter-sum spelled as a matmul, so it pins
    HIGHEST precision: the default would run it as a reduced-precision (TF32) dot and
    round the fp32 `data` it exists to sum exactly."""
    assert data.shape == bundle.values.shape, (data.shape, bundle.values.shape)
    k = bundle.router_indices.shape[-1]
    slots = data.astype(jnp.float32).reshape(*data.shape[:-1], k, bundle.c_per_expert)
    one_hot = jax.nn.one_hot(bundle.router_indices, bundle.n_experts, dtype=jnp.float32)
    if jax.sharding.get_abstract_mesh().empty:
        per_expert = jnp.einsum(
            "...kc,...ke->ec", slots, one_hot, precision=jax.lax.Precision.HIGHEST
        )
    else:
        # The token contraction spans the dp-sharded lead, so the output's sharding is
        # the contraction's to declare: shard-local partials, one global reduction.
        per_expert = jnp.einsum(
            "...kc,...ke->ec",
            slots,
            one_hot,
            precision=jax.lax.Precision.HIGHEST,
            out_sharding=P(None, None),
        )
    return per_expert.reshape(bundle.C)


def narrow_routed_counts(bundle: NarrowCI) -> Array:
    """fp32 per-EXPERT routed-token counts `[n_experts]` — the complement against the
    leading-axis extent prices the unrouted (token, component) pairs, which share one
    count across an expert's block. Ellipsis reduction, no leading collapse: the full
    sum over (possibly sharded) leading axes is unambiguous where a flattening reshape
    is not."""
    one_hot = jax.nn.one_hot(bundle.router_indices, bundle.n_experts, dtype=jnp.float32)
    return jnp.einsum("...ke->e", one_hot)


def narrow_component_maxes(bundle: NarrowCI, data: Array) -> Array:
    """fp32 segment-max of per-(token, slot) `data` (shaped like `bundle.values`) onto
    the site's FULL component axis — exactly the max over the full-width view, where an
    unrouted (token, component) entry is zero: an expert some token left unrouted takes
    `max(routed, 0)` across its block, and a never-routed expert's block is exactly 0.
    The `[C]` vector exists only as this reduction's output, never as a per-token
    tensor. Under a placed (explicit-sharding) mesh the token flattening keeps the
    leading axis's own spec and the scatter-max declares its replicated output — each
    shard maxes its tokens locally, one cross-shard max combines them."""
    assert data.shape == bundle.values.shape, (data.shape, bundle.values.shape)
    n = math.prod(bundle.values.shape[:-1])
    k = bundle.router_indices.shape[-1]
    data = data.astype(jnp.float32)
    routed_init = jnp.full((bundle.n_experts, bundle.c_per_expert), -jnp.inf, jnp.float32)
    if jax.sharding.get_abstract_mesh().empty:
        flat = data.reshape(n * k, bundle.c_per_expert)
        ids = bundle.router_indices.reshape(n * k)
        routed = routed_init.at[ids].max(flat)
    else:
        lead = jax.typeof(data).sharding.spec[0]
        flat = jax.lax.reshape(data, (n * k, bundle.c_per_expert), out_sharding=P(lead, None))
        ids = jax.lax.reshape(bundle.router_indices, (n * k,), out_sharding=P(lead))
        # jax's basearray stub lags the runtime signature: `out_sharding` exists on the
        # scatter ops but not in the shipped .pyi.
        routed = routed_init.at[ids].max(flat, out_sharding=P(None, None))  # pyright: ignore[reportCallIssue]
    unrouted_somewhere = narrow_routed_counts(bundle) < n
    full = jnp.where(unrouted_somewhere[:, None], jnp.maximum(routed, 0.0), routed)
    return full.reshape(bundle.C)


def require_full_emission(value: SiteCI) -> Array:
    """The full-emission arm of one site's CI. A consumer calling this has no narrow
    arm: each call site is an enumerated gap — a narrow site reaching it needs dispatch
    on the `NarrowCI` bundle, never a scatter to `[.., C]`."""
    match value:
        case NarrowCI():
            raise NotImplementedError(
                "this consumer has no narrow-emission arm; it must dispatch on the "
                "NarrowCI bundle, not materialize the full component axis"
            )
        case jax.Array():
            return value


DENSE_V_AXES: Axes = ("stack", "d_in", "C")
DENSE_U_AXES: Axes = ("stack", "C", "d_out")
DENSE_DELTA_AXES: Axes = ("stack", "d_out", "d_in")

EXPERT_V_AXES: Axes = ("stack", "expert", "d_in", "C_block")
EXPERT_U_AXES: Axes = ("stack", "expert", "C_block", "d_out")
EXPERT_DELTA_AXES: Axes = ("stack", "expert", "d_out", "d_in")


@dataclass(frozen=True, kw_only=True)
class Dense:
    """The whole-matrix factorization: the site's `W [d_out, d_in]` is factored as
    `V [d_in, C] @ U [C, d_out]`, and a group of such sites persists as one 3-D stack
    per factor."""

    d_in: int
    d_out: int
    C: int

    @property
    def v_axes(self) -> Axes:
        return DENSE_V_AXES

    @property
    def u_axes(self) -> Axes:
        return DENSE_U_AXES

    @property
    def delta_axes(self) -> Axes:
        return DENSE_DELTA_AXES

    def v_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.d_in, self.C)

    def u_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.C, self.d_out)

    def delta_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.d_out, self.d_in)


@dataclass(frozen=True, kw_only=True)
class ExpertBlocked:
    """The expert-local factorization, for a site whose weight matrix is a stack of
    per-expert blocks. Component `(e, k)` reads and writes only expert `e`'s block:
    each expert gets its own factors `V_e [d_in, c_per_expert]` and
    `U_e [c_per_expert, d_out]`, and a group of such sites persists as one 4-D stack
    per factor (`[stack, expert, ...]`). Note that `d_in` and `d_out` here are the
    dimensions of ONE expert's block, not of the whole site. Which side of the site
    concatenates the expert blocks (the output for gate/up matrices, the input for
    down matrices) is deliberately absent: only the forward computation needs it, so it is
    declared by the target where the site's linears are built
    (`linear_plan.ExpertContraction`). At the engine's mask/CI boundary the site still
    has one flat component axis of size `C = n_experts * c_per_expert`, ordered
    expert-major."""

    n_experts: int
    d_in: int
    d_out: int
    c_per_expert: int

    @property
    def C(self) -> int:
        return self.n_experts * self.c_per_expert

    @property
    def v_axes(self) -> Axes:
        return EXPERT_V_AXES

    @property
    def u_axes(self) -> Axes:
        return EXPERT_U_AXES

    @property
    def delta_axes(self) -> Axes:
        return EXPERT_DELTA_AXES

    def v_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.n_experts, self.d_in, self.c_per_expert)

    def u_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.n_experts, self.c_per_expert, self.d_out)

    def delta_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.n_experts, self.d_out, self.d_in)


type Factorization = Dense | ExpertBlocked
"""How one site's V/U factor its matrix. Every consumer whose behavior depends on the
kind (init, placement transitions, muon labeling, the linear primitive, faithfulness
deltas) matches on this union, so a new kind fails loudly wherever it lacks an arm.
All sites of one semantic group share one factorization (`vu_groups`)."""


@dataclass(frozen=True, kw_only=True)
class SiteDims:
    d_in: int
    d_out: int

    def dense(self, C: int) -> Dense:
        return Dense(d_in=self.d_in, d_out=self.d_out, C=C)


@dataclass(frozen=True)
class SiteSpec:
    """One decomposed site: its name, how its V/U factor its matrix, and its
    target-declared persistence group. The factorization is the stored shape truth;
    `C` (the flat per-site component count) is derived from it, and the dense-only
    `d_in`/`d_out` views assert the site is dense — an expert-blocked site has no
    single fused view here, because the factorization carries no orientation."""

    name: str
    factorization: Factorization
    group: str
    nonlinearity_partition: NonlinearityPartition | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        match self.nonlinearity_partition:
            case QueryHeads(head_count=head_count) | KVHeads(head_count=head_count):
                assert self.d_out % head_count == 0, self
            case Neurons() | None:
                pass

    @property
    def C(self) -> int:
        return self.factorization.C

    @property
    def d_in(self) -> int:
        assert isinstance(self.factorization, Dense), self
        return self.factorization.d_in

    @property
    def d_out(self) -> int:
        assert isinstance(self.factorization, Dense), self
        return self.factorization.d_out


def nonlinearity_partitions(sites: tuple[SiteSpec, ...]) -> dict[str, NonlinearityPartition]:
    return {s.name: s.nonlinearity_partition for s in sites if s.nonlinearity_partition is not None}


@dataclass(frozen=True)
class SiteComponents:
    """The two rank-one factor matrices for one decomposed site."""

    V: Array
    U: Array


# site name -> (target-declared group, slot on the group's stack axis)
SiteSlots = tuple[tuple[str, str, int], ...]

# The V/U leaf type: `Array` for the real fp32 masters (the default — so bare `ComponentStacks`
# means `ComponentStacks[Array]` and no call site needs the parameter), or `NamedSharding` for
# the same-structure placement tree `placement.component_stacks_shardings` returns for
# `jax.jit(out_shardings=...)`.
VULeaf = TypeVar("VULeaf", default=Array)


@dataclass(frozen=True)
class VUGroup:
    """One semantic persistence group: its sites in canonical order, and the one
    factorization they all share. The factorization lives here — established once, when
    the grouping is built — so no consumer ever has to re-derive it from a member."""

    factorization: Factorization
    specs: tuple[SiteSpec, ...]


def vu_groups(sites: tuple[SiteSpec, ...]) -> dict[str, VUGroup]:
    """Sites grouped by the target's semantic persistence group. A group's sites must
    all share one factorization, because they persist as slots of one homogeneous
    stack."""
    grouped: dict[str, list[SiteSpec]] = {}
    for spec in sites:
        grouped.setdefault(spec.group, []).append(spec)
    groups: dict[str, VUGroup] = {}
    for name, specs in grouped.items():
        factorizations = {spec.factorization for spec in specs}
        assert len(factorizations) == 1, (
            f"component group {name!r} mixes factorizations: {factorizations}"
        )
        groups[name] = VUGroup(factorization=factorizations.pop(), specs=tuple(specs))
    return groups


def group_factorizations(sites: tuple[SiteSpec, ...]) -> dict[str, Factorization]:
    """Each semantic group's factorization, keyed by group name."""
    return {name: group.factorization for name, group in vu_groups(sites).items()}


def site_slots_for(sites: tuple[SiteSpec, ...]) -> SiteSlots:
    """The canonical site→(group, slot) mapping in site order."""
    by_name: dict[str, tuple[str, int]] = {}
    for name, group in vu_groups(sites).items():
        for slot, spec in enumerate(group.specs):
            by_name[spec.name] = (name, slot)
    return tuple((spec.name, *by_name[spec.name]) for spec in sites)


@cache
def slot_index(site_slots: SiteSlots) -> dict[str, tuple[str, int]]:
    """site name -> (group, slot), cached per `SiteSlots` value."""
    return {name: (group, slot) for name, group, slot in site_slots}


class ComponentStacks(eqx.Module, Generic[VULeaf]):
    """The trainable V/U masters: one homogeneous stack per target-declared semantic group.

    A group holds `(Vs [g, d_in, C], Us [g, C, d_out])`; `site_slots` maps each site to
    its slot. LM targets declare matrix kind as the group, making each scan input a leaf.
    Toy targets may declare independent per-site groups. Placement is separate: a rule may
    shard the stack axis for ownership or shard matrix dimensions instead.

    `stack_pads` enumerates the persist-layer PAD slots: a stack-sharding placement whose
    extent the real stack length does not tile pads each stack with trailing all-zero
    slots (`pad_component_stacks`), and this field carries that fact as data — entries
    only for groups with a nonzero pad, `()` = unpadded — so no consumer ever infers a
    pad from a shape. Pad slots exist only between the persist layer and the entry
    boundaries that strip them; `site_slots` never names them.

    Leaves are fp32 master Arrays (`ComponentStacks[Array]`) or `NamedSharding`s in the
    same-structure placement tree `placement.component_stacks_shardings` returns
    (`ComponentStacks[NamedSharding]`). This module is placement-FREE: the per-group row
    lookup and its boundary validation live in `placement.py`, above."""

    stacks: dict[str, tuple[VULeaf, VULeaf]]
    site_slots: SiteSlots = eqx.field(static=True)
    stack_pads: tuple[tuple[str, int], ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        groups = {group for _, group, _ in self.site_slots}
        assert all(group in groups and pad > 0 for group, pad in self.stack_pads), (
            self.stack_pads,
            sorted(groups),
        )

    def pad_of(self, group: str) -> int:
        return dict(self.stack_pads).get(group, 0)

    def slot_of(self, name: str) -> tuple[str, int]:
        return slot_index(self.site_slots)[name]

    def site(self: "ComponentStacks[Array]", name: str) -> SiteComponents:
        group, slot = self.slot_of(name)
        Vs, Us = self.stacks[group]
        return SiteComponents(V=Vs[slot], U=Us[slot])

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.site_slots)

    def sites_items(self: "ComponentStacks[Array]") -> Iterator[tuple[str, SiteComponents]]:
        """Named site components in canonical site order."""
        for name, _, _ in self.site_slots:
            yield name, self.site(name)

    def group_lengths(self) -> dict[str, int]:
        """Stack length per semantic group, available from eval-shape trees."""
        lengths: dict[str, int] = {}
        for _name, group, slot in self.site_slots:
            lengths[group] = max(lengths.get(group, 0), slot + 1)
        return lengths


def pad_component_stacks(
    stacks: "ComponentStacks[Array]", pads: "Mapping[str, int]"
) -> "ComponentStacks[Array]":
    """Append each group's declared pad count as trailing ALL-ZERO stack slots — the one
    constructor of a padded persist tree. Pads are enumerated on `stack_pads`, never
    inferred; entries of 0 are dropped so `()` stays the single spelling of unpadded."""
    assert stacks.stack_pads == (), f"already padded: {stacks.stack_pads}"
    assert pads.keys() <= stacks.stacks.keys(), (sorted(pads), sorted(stacks.stacks))
    padded: dict[str, tuple[Array, Array]] = {}
    for group, (Vs, Us) in stacks.stacks.items():
        pad = pads.get(group, 0)
        if pad == 0:
            padded[group] = (Vs, Us)
            continue
        padded[group] = (
            jnp.concatenate([Vs, jnp.zeros((pad, *Vs.shape[1:]), Vs.dtype)]),
            jnp.concatenate([Us, jnp.zeros((pad, *Us.shape[1:]), Us.dtype)]),
        )
    return ComponentStacks(
        stacks=padded,
        site_slots=stacks.site_slots,
        stack_pads=tuple((group, pad) for group, pad in pads.items() if pad > 0),
    )


EXPERT_U_INIT_FAN_IN: Literal["site", "block"] = "site"
"""Which fan-in sets an expert-blocked U's init scale. `"site"` draws
`U_e ~ N(0, C^-1/2)` with `C = n_experts * c_per_expert`: a site whose output SUMS the
expert blocks (the down orientation) then starts with the same output variance as a
dense site. `"block"` is the live alternative — the dense rule applied to each block's
own fan-in, `U_e ~ N(0, c_per_expert^-1/2)`, which instead variance-matches a site
whose output CONCATENATES the blocks (the gate/up orientation). The factorization
deliberately carries no orientation, so one choice applies to every expert U; flipping
this constant is the whole change."""


def init_stack_arrays(sites: tuple[SiteSpec, ...], key: Array) -> dict[str, tuple[Array, Array]]:
    """Seed each semantic group's V/U stacks, drawing one (V, U) key pair per site in
    site order. V scales by its fan-in (`d_in^-1/2`, the per-expert block `d_in` for
    expert-blocked groups); U scales by `C^-1/2` for dense groups and by
    `EXPERT_U_INIT_FAN_IN`'s choice for expert-blocked groups."""
    site_keys = jax.random.split(key, (len(sites), 2))
    site_index = {spec.name: idx for idx, spec in enumerate(sites)}
    stacked: dict[str, tuple[Array, Array]] = {}
    for name, group in vu_groups(sites).items():
        idxs = jnp.array([site_index[spec.name] for spec in group.specs])
        v_keys, u_keys = site_keys[idxs, 0], site_keys[idxs, 1]
        match group.factorization:
            case Dense(d_in=d_in, d_out=d_out, C=c):
                Vs = jax.vmap(lambda k, s=(d_in, c): jax.random.normal(k, s))(v_keys)
                Us = jax.vmap(lambda k, s=(c, d_out): jax.random.normal(k, s))(u_keys)
                stacked[name] = (Vs * d_in**-0.5, Us * c**-0.5)
            case ExpertBlocked(n_experts=n_experts, d_in=d_in, d_out=d_out, c_per_expert=c) as f:
                Vs = jax.vmap(lambda k, s=(n_experts, d_in, c): jax.random.normal(k, s))(v_keys)
                Us = jax.vmap(lambda k, s=(n_experts, c, d_out): jax.random.normal(k, s))(u_keys)
                match EXPERT_U_INIT_FAN_IN:
                    case "site":
                        u_fan_in = f.C
                    case "block":
                        u_fan_in = c
                stacked[name] = (Vs * d_in**-0.5, Us * u_fan_in**-0.5)
    return stacked


def component_stacks_from_site_arrays(
    sites: tuple[SiteSpec, ...], vu: dict[str, tuple[Array, Array]]
) -> ComponentStacks:
    assert tuple(vu) == tuple(spec.name for spec in sites), (tuple(vu), sites)
    stacks = {
        name: (
            jnp.stack([vu[spec.name][0] for spec in group.specs]),
            jnp.stack([vu[spec.name][1] for spec in group.specs]),
        )
        for name, group in vu_groups(sites).items()
    }
    return ComponentStacks(stacks=stacks, site_slots=site_slots_for(sites))


def component_stacks_from_sites(vu: dict[str, tuple[Array, Array]]) -> ComponentStacks:
    """Build independently grouped component leaves from explicit per-site arrays."""
    sites = tuple(
        SiteSpec(
            name=name,
            factorization=Dense(d_in=V.shape[0], d_out=U.shape[1], C=V.shape[1]),
            group=name,
        )
        for name, (V, U) in vu.items()
    )
    return component_stacks_from_site_arrays(sites, vu)


def init_component_stacks(sites: tuple[SiteSpec, ...], key: Array) -> ComponentStacks:
    """Small random fp32 V/U per site at the scales `init_stack_arrays` documents, built
    directly in the stacked persistence layout; the weight-delta channel carries the
    faithfulness residual at init (before faithfulness warmup)."""
    return ComponentStacks(stacks=init_stack_arrays(sites, key), site_slots=site_slots_for(sites))
