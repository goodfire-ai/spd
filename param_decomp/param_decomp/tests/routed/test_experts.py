"""The routed-experts primitive against its dense oracle.

The dense reference computes every expert for every token and zeroes the unselected via
the dense routing vector; the routed path computes the selected jobs only. Forward AND
backward must agree: the backward comparison pins the custom-VJP sort/unsort gathers
against plain autodiff through the dense reference. The fixture keeps two experts EMPTY
so zero-size groups stay exercised.

Tolerances are nonzero because reduction order differs: the routed path contracts
per-expert row blocks and combines in fp32 after the down matmul, the dense oracle
folds routing before one all-expert contraction.

The decomposed-seam primitives (`sort_jobs` / `gather_job_blocks` / `sum_jobs` and the
EP siblings incl. `ep_grouped_matmul` over a provenance-carrying trained weight) get
the same treatment: a masked decomposed mini-site vs its dense spelling, values and
gradients. The EP test runs on the simulated 8-device `(data, tp)` mesh with the weight
`reduced`-typed the way `prepare_compute_weights` types the real residents.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.routed.experts import (
    GroupedMatmulBackend,
    combine_jobs,
    ep_gather_job_blocks,
    ep_gather_tokens,
    ep_grouped_matmul,
    ep_sort_job_values,
    ep_sort_jobs,
    ep_sum_jobs,
    expert_sharded_jobs,
    gather_job_blocks,
    gather_tokens,
    grouped_matmul,
    routed_jobs,
    scatter_jobs,
    sort_jobs,
    sum_jobs,
    transposed_grouped_matmul,
    unsort_jobs,
)

# D and DI are 64-multiples so every backend arm can serve the fixtures:
# `transposed_grouped_matmul` (the split arm's d_weights) tiles d_in/d_out at 64.
E, K, T, D, DI = 8, 2, 24, 64, 128


def _fixture() -> tuple[Array, Array, Array, Array, Array, Array]:
    ks = jax.random.split(jax.random.PRNGKey(0), 5)
    x = jax.random.normal(ks[0], (T, D))
    w_gate = jax.random.normal(ks[1], (E, D, DI)) * D**-0.5
    w_up = jax.random.normal(ks[2], (E, D, DI)) * D**-0.5
    w_down = jax.random.normal(ks[3], (E, DI, D)) * DI**-0.5
    # Deterministic assignments: distinct experts per token; experts E-2 and E-1 EMPTY.
    first = jnp.arange(T) % (E - 2)
    top_idx = jnp.stack([first, (first + 1) % (E - 2)], axis=1)
    top_w = jax.nn.softmax(jax.random.normal(ks[4], (T, K)), axis=-1)
    return x, w_gate, w_up, w_down, top_idx, top_w


def _routed_moe(
    x: Float[Array, "T d"],
    w_gate: Array,
    w_up: Array,
    w_down: Array,
    top_idx: Int[Array, "T k"],
    top_w: Float[Array, "T k"],
) -> Float[Array, "T d"]:
    jobs = routed_jobs(top_idx, E)
    x_jobs = gather_tokens(x, jobs)
    gate = grouped_matmul(x_jobs, w_gate, jobs.group_sizes, "ragged_dot")
    up = grouped_matmul(x_jobs, w_up, jobs.group_sizes, "ragged_dot")
    down = grouped_matmul(jax.nn.silu(gate) * up, w_down, jobs.group_sizes, "ragged_dot")
    return combine_jobs(down, jobs, top_w)


def _dense_moe(
    x: Float[Array, "T d"],
    w_gate: Array,
    w_up: Array,
    w_down: Array,
    top_idx: Int[Array, "T k"],
    top_w: Float[Array, "T k"],
) -> Float[Array, "T d"]:
    routing = jnp.sum(jax.nn.one_hot(top_idx, E) * top_w[..., None], axis=-2)
    gate = jnp.einsum("td,edh->teh", x, w_gate)
    up = jnp.einsum("td,edh->teh", x, w_up)
    return jnp.einsum("teh,ehd->td", jax.nn.silu(gate) * up * routing[..., None], w_down)


def test_jobs_schedule_counts_experts_with_static_shapes():
    _x, _wg, _wu, _wd, top_idx, _tw = _fixture()
    jobs = routed_jobs(top_idx, E)
    assert jobs.sort_idx.shape == jobs.inv_sort_idx.shape == (T * K,)
    np.testing.assert_array_equal(
        jobs.group_sizes, np.bincount(np.asarray(top_idx).ravel(), minlength=E)
    )
    np.testing.assert_array_equal(jobs.group_sizes[-2:], [0, 0])


def test_forward_matches_dense_oracle():
    args = _fixture()
    routed = jax.jit(_routed_moe)(*args)
    dense = _dense_moe(*args)
    np.testing.assert_allclose(routed, dense, rtol=1e-5, atol=1e-6)


def test_backward_matches_dense_autodiff():
    x, w_gate, w_up, w_down, top_idx, top_w = _fixture()

    def routed_loss(x: Array, wg: Array, wu: Array, wd: Array, tw: Array) -> Array:
        return jnp.sum(jnp.cos(_routed_moe(x, wg, wu, wd, top_idx, tw)))

    def dense_loss(x: Array, wg: Array, wu: Array, wd: Array, tw: Array) -> Array:
        return jnp.sum(jnp.cos(_dense_moe(x, wg, wu, wd, top_idx, tw)))

    argnums = (0, 1, 2, 3, 4)
    routed_grads = jax.grad(routed_loss, argnums)(x, w_gate, w_up, w_down, top_w)
    dense_grads = jax.grad(dense_loss, argnums)(x, w_gate, w_up, w_down, top_w)
    for routed_grad, dense_grad in zip(routed_grads, dense_grads, strict=True):
        np.testing.assert_allclose(routed_grad, dense_grad, rtol=1e-4, atol=1e-6)


def test_gather_then_unsort_round_trips_exactly():
    x, _wg, _wu, _wd, top_idx, _tw = _fixture()
    jobs = routed_jobs(top_idx, E)
    per_token = unsort_jobs(gather_tokens(x, jobs), jobs)
    np.testing.assert_array_equal(per_token, jnp.broadcast_to(x[:, None], (T, K, D)))


def test_scatter_jobs_hits_selected_slots_and_zeros_the_rest():
    x, w_gate, _wu, _wd, top_idx, _tw = _fixture()
    jobs = routed_jobs(top_idx, E)
    gate_jobs = grouped_matmul(gather_tokens(x, jobs), w_gate, jobs.group_sizes, "ragged_dot")
    scattered = np.asarray(scatter_jobs(gate_jobs, jobs, E))
    dense_gate = np.asarray(jnp.einsum("td,edh->teh", x, w_gate))
    selected = np.asarray(jnp.any(jax.nn.one_hot(top_idx, E, dtype=bool), axis=-2))
    np.testing.assert_allclose(scattered[selected], dense_gate[selected], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(scattered[~selected], 0.0)


def test_tokamax_arm_matches_the_ragged_dot_oracle():
    """On CPU tokamax auto-selects its XLA reference implementation, so forward and both
    grads must agree with `jax.lax.ragged_dot` exactly."""
    x, w_gate, _wu, _wd, top_idx, _tw = _fixture()
    jobs = routed_jobs(top_idx, E)
    lhs = gather_tokens(x, jobs)

    def loss(backend: GroupedMatmulBackend, l: jax.Array, w: jax.Array):  # noqa: E741
        return grouped_matmul(l, w, jobs.group_sizes, backend).sum()

    np.testing.assert_array_equal(
        grouped_matmul(lhs, w_gate, jobs.group_sizes, "tokamax"),
        grouped_matmul(lhs, w_gate, jobs.group_sizes, "ragged_dot"),
    )
    for tok, ref in zip(
        jax.grad(partial(loss, "tokamax"), (0, 1))(lhs, w_gate),
        jax.grad(partial(loss, "ragged_dot"), (0, 1))(lhs, w_gate),
        strict=True,
    ):
        np.testing.assert_array_equal(tok, ref)


# ── the grouped-matmul backend arms ───────────────────────────────────────────

_RAGGED_CONTRACTING_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


def test_transposed_grouped_matmul_matches_the_ragged_contracting_oracle():
    """The d_weights kernel against `jax.lax.ragged_dot_general`'s ragged-contracting
    spelling: unaligned group sizes (none a block multiple), a single-job group, a
    group spanning multiple job blocks, and EMPTY groups — which must come back
    exactly zero (the tokamax triton spelling fails exactly there, tokamax#887).
    fp32 tolerances are reduction-order noise; both paths accumulate in fp32."""
    sizes = jnp.array([7, 0, 19, 1, 0, 26, 37, 0], jnp.int32)
    n_jobs = int(sizes.sum())
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    lhs = jax.random.normal(k1, (n_jobs, D))
    dout = jax.random.normal(k2, (n_jobs, DI))
    oracle = jax.lax.ragged_dot_general(lhs, dout, sizes, _RAGGED_CONTRACTING_DIM_NUMS)
    out = jax.jit(partial(transposed_grouped_matmul, out_dtype=jnp.float32))(lhs, dout, sizes)
    np.testing.assert_allclose(out, oracle, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(out)[np.asarray(sizes) == 0], 0.0)


def test_transposed_grouped_matmul_bf16_matches_the_fp32_oracle():
    """bf16 inputs, fp32 accumulation, bf16 output — against the fp32 oracle over the
    same quantized inputs. The tolerance is the OUTPUT rounding (bf16 mantissa,
    ~0.4% relative) plus reduction-order noise; the input quantization is common to
    both sides."""
    sizes = jnp.array([7, 0, 19, 1, 0, 26, 37, 0], jnp.int32)
    n_jobs = int(sizes.sum())
    k1, k2 = jax.random.split(jax.random.PRNGKey(4))
    lhs = jax.random.normal(k1, (n_jobs, D), jnp.bfloat16)
    dout = jax.random.normal(k2, (n_jobs, DI), jnp.bfloat16)
    oracle = jax.lax.ragged_dot_general(
        lhs, dout, sizes, _RAGGED_CONTRACTING_DIM_NUMS, preferred_element_type=jnp.float32
    )
    out = transposed_grouped_matmul(lhs, dout, sizes, out_dtype=jnp.bfloat16)
    np.testing.assert_allclose(out.astype(jnp.float32), oracle, rtol=2e-2, atol=2e-2)


def test_tokamax_split_vjp_arm_matches_the_ragged_dot_oracle():
    """Forward and both gradients of the split-VJP arm against `jax.lax.ragged_dot`
    autodiff. The forward resolves through the same tokamax chain as the `tokamax`
    arm; the gradients exercise the arm's own product spellings (TRANS_RHS d_input,
    `transposed_grouped_matmul` d_weights)."""
    x, w_gate, _wu, _wd, top_idx, _tw = _fixture()
    jobs = routed_jobs(top_idx, E)
    lhs = gather_tokens(x, jobs)

    def loss(backend: GroupedMatmulBackend, l: jax.Array, w: jax.Array):  # noqa: E741
        return jnp.sum(jnp.cos(grouped_matmul(l, w, jobs.group_sizes, backend)))

    np.testing.assert_array_equal(
        grouped_matmul(lhs, w_gate, jobs.group_sizes, "tokamax_split_vjp"),
        grouped_matmul(lhs, w_gate, jobs.group_sizes, "ragged_dot"),
    )
    for split, ref in zip(
        jax.jit(jax.grad(partial(loss, "tokamax_split_vjp"), (0, 1)))(lhs, w_gate),
        jax.grad(partial(loss, "ragged_dot"), (0, 1))(lhs, w_gate),
        strict=True,
    ):
        np.testing.assert_allclose(split, ref, rtol=1e-5, atol=1e-6)


# ── the decomposed seam: job-space masks, weights, and component matmuls ──────

C = 64  # components per expert block


def _decomposed_fixture() -> tuple[Array, Array, Array, Array, Array, Array]:
    ks = jax.random.split(jax.random.PRNGKey(1), 5)
    x = jax.random.normal(ks[0], (T, D))
    v = jax.random.normal(ks[1], (E, D, C)) * D**-0.5
    u = jax.random.normal(ks[2], (E, C, DI)) * C**-0.5
    mask = jax.random.uniform(ks[3], (T, E, C))
    first = jnp.arange(T) % (E - 2)
    top_idx = jnp.stack([first, (first + 1) % (E - 2)], axis=1)
    top_w = jax.nn.softmax(jax.random.normal(ks[4], (T, K)), axis=-1)
    return x, v, u, mask, top_idx, top_w


def _routed_masked_site(
    backend: GroupedMatmulBackend,
    x: Array,
    v: Array,
    u: Array,
    mask: Array,
    top_idx: Array,
    top_w: Array,
) -> Array:
    """One masked decomposed site with routing weights folded into its input, summed
    per token — the routed decomposed arm's whole vocabulary in one expression."""
    jobs = routed_jobs(top_idx, E)
    x_jobs = gather_tokens(x, jobs) * sort_jobs(top_w, jobs)[:, None]
    acts = grouped_matmul(x_jobs, v, jobs.group_sizes, backend) * gather_job_blocks(mask, jobs)
    return sum_jobs(grouped_matmul(acts, u, jobs.group_sizes, backend), jobs)


def _dense_masked_site(
    x: Array, v: Array, u: Array, mask: Array, top_idx: Array, top_w: Array
) -> Array:
    routing = jnp.sum(jax.nn.one_hot(top_idx, E) * top_w[..., None], axis=-2)
    acts = jnp.einsum("te,td,edc->tec", routing, x, v) * mask
    return jnp.einsum("tec,ech->th", acts, u)


_SEAM_BACKENDS: tuple[GroupedMatmulBackend, ...] = ("ragged_dot", "tokamax", "tokamax_split_vjp")


@pytest.mark.parametrize("backend", _SEAM_BACKENDS)
def test_masked_decomposed_site_matches_dense_oracle(backend: GroupedMatmulBackend):
    args = _decomposed_fixture()
    routed = jax.jit(partial(_routed_masked_site, backend))(*args)
    dense = _dense_masked_site(*args)
    np.testing.assert_allclose(routed, dense, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", _SEAM_BACKENDS)
def test_masked_decomposed_site_backward_matches_dense_autodiff(backend: GroupedMatmulBackend):
    x, v, u, mask, top_idx, top_w = _decomposed_fixture()

    def routed_loss(x: Array, v: Array, u: Array, mask: Array, tw: Array) -> Array:
        return jnp.sum(jnp.cos(_routed_masked_site(backend, x, v, u, mask, top_idx, tw)))

    def dense_loss(x: Array, v: Array, u: Array, mask: Array, tw: Array) -> Array:
        return jnp.sum(jnp.cos(_dense_masked_site(x, v, u, mask, top_idx, tw)))

    argnums = (0, 1, 2, 3, 4)
    routed_grads = jax.jit(jax.grad(routed_loss, argnums))(x, v, u, mask, top_w)
    dense_grads = jax.grad(dense_loss, argnums)(x, v, u, mask, top_w)
    for routed_grad, dense_grad in zip(routed_grads, dense_grads, strict=True):
        np.testing.assert_allclose(routed_grad, dense_grad, rtol=1e-4, atol=1e-6)


def test_gather_job_blocks_grad_lands_on_selected_rows_only():
    x, v, _u, mask, top_idx, _tw = _decomposed_fixture()
    jobs = routed_jobs(top_idx, E)
    xv = grouped_matmul(gather_tokens(x, jobs), v, jobs.group_sizes, "ragged_dot")

    def loss(mask: Array) -> Array:
        return jnp.sum(xv * gather_job_blocks(mask, jobs))

    d_mask = jax.grad(loss)(mask)
    selected = np.asarray(jnp.any(jax.nn.one_hot(top_idx, E, dtype=bool), axis=-2))
    np.testing.assert_array_equal(np.asarray(d_mask)[~selected], 0.0)
    assert np.abs(np.asarray(d_mask)[selected]).min() > 0.0


def test_sum_jobs_is_the_weightless_combine():
    x, _v, _u, _mask, top_idx, _tw = _decomposed_fixture()
    jobs = routed_jobs(top_idx, E)
    y = gather_tokens(x, jobs)
    np.testing.assert_allclose(
        sum_jobs(y, jobs), combine_jobs(y, jobs, jnp.ones((T, K))), rtol=1e-6
    )


multidevice = pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")


def _uniform_top_idx(key: Array, b: int, t: int, k: int, e: int) -> Array:
    """Every token's top-k from random logits — all experts busy with high probability."""
    logits = jax.random.normal(key, (b, t, e))
    return jnp.argsort(-logits, axis=-1)[..., :k].astype(jnp.int32)


def _skewed_top_idx(b: int, t: int) -> Array:
    """Skewed routing over e=4 experts on s=2 shards, k=2: batch row 0 routes every
    token to shard 0's experts (0, 1) — shard 1's cell there is EMPTY — and the other
    rows lean on expert 0 and never touch expert 3, so it is empty everywhere."""
    row0 = jnp.broadcast_to(jnp.array([0, 1], jnp.int32), (t, 2))
    rest = jnp.stack([jnp.zeros(t, jnp.int32), 1 + jnp.arange(t, dtype=jnp.int32) % 2], axis=1)
    return jnp.concatenate([row0[None], jnp.broadcast_to(rest[None], (b - 1, t, 2))], axis=0)


_EP_BACKENDS: tuple[GroupedMatmulBackend, ...] = ("ragged_dot", "tokamax_split_vjp")


@multidevice
@pytest.mark.multidevice
@pytest.mark.parametrize("routing", ("uniform", "skewed_empty"))
@pytest.mark.parametrize("backend", _EP_BACKENDS)
def test_ep_decomposed_site_matches_dense_with_provenance_typed_weights(
    backend: GroupedMatmulBackend, routing: str
):
    """The EP spelling of the masked decomposed site on the (data=4, tp=2) mesh, with
    V/U `reduced`-typed over `data` exactly as the placed compute residents are: values
    and gradients (d-input, dV, dU, d-mask, d-weights) match the dense reference, and
    the V/U cotangents surface through the entry-boundary reshard that applied the tag.
    Both wired backend arms serve both matmul orientations (V: the d_in→c product, U:
    the c→d_out product); the skewed routing keeps one expert empty everywhere and one
    (batch row, shard) cell entirely empty. d/c/di are 64-multiples so the split arm's
    `transposed_grouped_matmul` can tile them, and distinct so no transposition hides."""
    devices = np.asarray(jax.devices()[:8]).reshape(4, 2)
    mesh = Mesh(devices, ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)
    b, t, k, e, d, c, di, s = 4, 8, 2, 4, 64, 128, 192, 2
    ks = jax.random.split(jax.random.PRNGKey(2), 6)
    with jax.set_mesh(mesh):
        batch_spec = NamedSharding(mesh, P("data", None, None))
        x = jax.device_put(jax.random.normal(ks[0], (b, t, d)), batch_spec)
        # variance-matched inits keep `out` O(1): `cos` in the losses amplifies benign
        # reduction-order noise chaotically at large arguments.
        v_master = jax.device_put(
            jax.random.normal(ks[1], (e, d, c)) * d**-0.5,
            NamedSharding(mesh, P("tp", None, "data")),
        )
        u_master = jax.device_put(
            jax.random.normal(ks[2], (e, c, di)) * c**-0.5,
            NamedSharding(mesh, P("tp", "data", None)),
        )
        mask = jax.device_put(
            jax.random.uniform(ks[3], (b, t, e, c)),
            NamedSharding(mesh, P("data", None, None, None)),
        )
        match routing:
            case "uniform":
                top_idx = _uniform_top_idx(ks[4], b, t, k, e)
            case "skewed_empty":
                top_idx = _skewed_top_idx(b, t)
            case _:
                raise AssertionError(routing)
        top_idx = jax.device_put(top_idx, batch_spec)
        top_w = jax.device_put(
            jax.nn.softmax(jax.random.normal(ks[5], (b, t, k)), axis=-1), batch_spec
        )

        def entry(master: Array, spec: P) -> Array:
            # the masters→resident gather, typed `reduced` over the gathered axis —
            # `placement.materialize_reduced_weights`'s move.
            return jax.sharding.reshard(
                jax.lax.optimization_barrier(master),
                NamedSharding(mesh, P(*spec, reduced=frozenset({"data"}))),
            )

        def routed_loss(x: Array, vm: Array, um: Array, mask: Array, tw: Array) -> Array:
            v = entry(vm, P("tp", None, None))
            u = entry(um, P("tp", None, None))
            jobs = expert_sharded_jobs(top_idx, e, s)
            x_jobs = ep_gather_tokens(x, jobs, "tp") * ep_sort_jobs(tw, jobs, "tp")[..., None]
            acts = ep_grouped_matmul(x_jobs, v, jobs, "tp", backend) * ep_gather_job_blocks(
                mask, jobs, "tp"
            )
            out = ep_sum_jobs(
                ep_grouped_matmul(acts, u, jobs, "tp", backend),
                jobs,
                "tp",
                P("data", None, None),
            )
            return jnp.sum(jnp.cos(out))

        def dense_loss(x: Array, vm: Array, um: Array, mask: Array, tw: Array) -> Array:
            v = jax.sharding.reshard(vm, NamedSharding(mesh, P(None, None, None)))
            u = jax.sharding.reshard(um, NamedSharding(mesh, P(None, None, None)))
            m = jax.sharding.reshard(mask, NamedSharding(mesh, P("data", None, None, None)))
            routing = jnp.sum(jax.nn.one_hot(top_idx, e) * tw[..., None], axis=-2)
            acts = (
                jnp.einsum(
                    "bte,btd,edc->btec",
                    routing,
                    x,
                    v,
                    out_sharding=NamedSharding(mesh, P("data", None, None, None)),
                )
                * m
            )
            out = jnp.einsum(
                "btec,ech->bth",
                acts,
                u,
                out_sharding=NamedSharding(mesh, P("data", None, None)),
            )
            return jnp.sum(jnp.cos(out))

        argnums = (0, 1, 2, 3, 4)
        routed_grads = jax.jit(jax.grad(routed_loss, argnums))(x, v_master, u_master, mask, top_w)
        dense_grads = jax.jit(jax.grad(dense_loss, argnums))(x, v_master, u_master, mask, top_w)
    for routed_grad, dense_grad in zip(routed_grads, dense_grads, strict=True):
        # The routed job-order sum and the dense einsum reassociate; on an entry that
        # nearly cancels out of O(1) terms they can land a few fp32 ulps (at scale 1)
        # apart, and where they land differs by host ISA (x86 FMA contraction vs ARM).
        # The floor is a few ulps at the operands' scale, not at the entry's own.
        np.testing.assert_allclose(
            np.asarray(routed_grad), np.asarray(dense_grad), rtol=1e-4, atol=1e-5
        )


@multidevice
@pytest.mark.multidevice
def test_ep_replicated_row_gathers_slice_the_schedule_before_gathering():
    """`ep_gather_tokens` / `ep_sort_job_values` / `ep_sort_jobs` gather a
    shard-replicated row table by THIS shard's schedule rows; gathering every shard's
    rows and then resharding to the expert-shard axis is the same function of the same
    inputs. Forward values agree bit for bit (pure data movement), and the custom-VJP
    cotangents match autodiff through the all-shard spelling."""
    devices = np.asarray(jax.devices()[:8]).reshape(4, 2)
    mesh = Mesh(devices, ("data", "tp"), axis_types=(AxisType.Explicit,) * 2)
    b, t, k, e, d, s = 4, 8, 2, 4, 64, 2
    ks = jax.random.split(jax.random.PRNGKey(5), 6)
    with jax.set_mesh(mesh):
        batch_spec = NamedSharding(mesh, P("data", None, None))
        x = jax.device_put(jax.random.normal(ks[0], (b, t, d)), batch_spec)
        values = jax.device_put(jax.random.normal(ks[1], (b, t, k, d)), batch_spec)
        slots = jax.device_put(jax.random.normal(ks[2], (b, t, k)), batch_spec)
        top_idx = jax.device_put(_uniform_top_idx(ks[3], b, t, k, e), batch_spec)
        row_cotangent = jax.device_put(
            jax.random.normal(ks[4], (b, s, t * k, d)),
            NamedSharding(mesh, P("data", "tp", None, None)),
        )
        slot_cotangent = jax.device_put(
            jax.random.normal(ks[5], (b, s, t * k)), NamedSharding(mesh, P("data", "tp", None))
        )
        jobs = expert_sharded_jobs(top_idx, e, s)

        def all_shard_gather(rows: Array, row_of_job: Array) -> Array:
            gathered = jnp.take_along_axis(rows[:, None], row_of_job[..., None], axis=2)
            return jax.sharding.reshard(gathered, NamedSharding(mesh, P("data", "tp", None, None)))

        cases = (
            (
                lambda a: ep_gather_tokens(a, jobs, "tp"),
                lambda a: all_shard_gather(a, jobs.sort_idx // k),
                x,
                row_cotangent,
            ),
            (
                lambda a: ep_sort_job_values(a, jobs, "tp"),
                lambda a: all_shard_gather(a.reshape(b, t * k, d), jobs.sort_idx),
                values,
                row_cotangent,
            ),
            (
                lambda a: ep_sort_jobs(a, jobs, "tp"),
                lambda a: all_shard_gather(a.reshape(b, t * k, 1), jobs.sort_idx)[..., 0],
                slots,
                slot_cotangent,
            ),
        )
        for routed, reference, arg, cotangent in cases:
            out, routed_vjp = jax.jit(lambda a, f=routed: jax.vjp(f, a))(arg)
            expected, reference_vjp = jax.jit(lambda a, f=reference: jax.vjp(f, a))(arg)
            assert out.sharding == expected.sharding
            np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
            (grad,) = routed_vjp(cotangent)
            (expected_grad,) = reference_vjp(cotangent)
            np.testing.assert_allclose(
                np.asarray(grad), np.asarray(expected_grad), rtol=1e-5, atol=1e-6
            )
