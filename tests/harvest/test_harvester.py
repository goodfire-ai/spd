"""Tests for the Harvester class and extract_firing_windows."""

import random
from pathlib import Path

import pytest
import torch

from spd.harvest.harvester import Harvester, extract_firing_windows
from spd.harvest.reservoir import WINDOW_PAD_SENTINEL

DEVICE = torch.device("cpu")

# Small dimensions for fast tests
LAYER_NAMES = ["layer_0", "layer_1"]
C_PER_LAYER = {"layer_0": 4, "layer_1": 4}
N_TOTAL = 8  # sum of c_per_layer values
VOCAB_SIZE = 10
CI_THRESHOLD = 0.5
MAX_EXAMPLES = 5
CONTEXT_TOKENS_PER_SIDE = 1
WINDOW = 2 * CONTEXT_TOKENS_PER_SIDE + 1  # 3


def _make_harvester() -> Harvester:
    return Harvester(
        layer_names=LAYER_NAMES,
        c_per_layer=C_PER_LAYER,
        vocab_size=VOCAB_SIZE,
        ci_threshold=CI_THRESHOLD,
        max_examples_per_component=MAX_EXAMPLES,
        context_tokens_per_side=CONTEXT_TOKENS_PER_SIDE,
        device=DEVICE,
    )


class TestInit:
    def test_tensor_shapes(self):
        h = _make_harvester()
        assert h.firing_counts.shape == (N_TOTAL,)
        assert h.ci_sums.shape == (N_TOTAL,)
        assert h.cooccurrence_counts.shape == (N_TOTAL, N_TOTAL)
        assert h.input_cooccurrence.shape == (N_TOTAL, VOCAB_SIZE)
        assert h.input_marginals.shape == (VOCAB_SIZE,)
        assert h.output_cooccurrence.shape == (N_TOTAL, VOCAB_SIZE)
        assert h.output_marginals.shape == (VOCAB_SIZE,)
        assert h.reservoir.tokens.shape == (N_TOTAL, MAX_EXAMPLES, WINDOW)
        assert h.reservoir.ci.shape == (N_TOTAL, MAX_EXAMPLES, WINDOW)
        assert h.reservoir.acts.shape == (N_TOTAL, MAX_EXAMPLES, WINDOW)
        assert h.reservoir.n_items.shape == (N_TOTAL,)
        assert h.reservoir.n_seen.shape == (N_TOTAL,)

    def test_tensors_on_correct_device(self):
        h = _make_harvester()
        assert h.firing_counts.device == DEVICE
        assert h.reservoir.tokens.device == DEVICE
        assert h.cooccurrence_counts.device == DEVICE

    def test_layer_offsets(self):
        h = _make_harvester()
        assert h.layer_offsets == {"layer_0": 0, "layer_1": 4}

    def test_tensors_initialized_to_zero(self):
        h = _make_harvester()
        assert h.firing_counts.sum() == 0
        assert h.ci_sums.sum() == 0
        assert h.cooccurrence_counts.sum() == 0
        assert h.reservoir.n_items.sum() == 0
        assert h.reservoir.n_seen.sum() == 0
        assert h.total_tokens_processed == 0

    def test_reservoir_tokens_initialized_to_sentinel(self):
        h = _make_harvester()
        assert (h.reservoir.tokens == WINDOW_PAD_SENTINEL).all()


class TestReservoirAdd:
    def test_fills_up_to_k(self):
        h = _make_harvester()
        k = h.reservoir.k
        comp = 2

        for i in range(k):
            comp_idx = torch.tensor([comp])
            tokens = torch.full((1, WINDOW), i, dtype=torch.long)
            ci = torch.ones(1, WINDOW)
            acts = torch.ones(1, WINDOW) * 0.5
            h.reservoir.add(comp_idx, tokens, ci, acts)

        assert h.reservoir.n_items[comp] == k
        assert h.reservoir.n_seen[comp] == k
        # All slots should be filled with distinct values
        for i in range(k):
            assert h.reservoir.tokens[comp, i, 0].item() == i

    def test_replacement_after_k(self):
        h = _make_harvester()
        k = h.reservoir.k
        comp = 0

        random.seed(42)
        n_extra = 100
        for i in range(k + n_extra):
            comp_idx = torch.tensor([comp])
            tokens = torch.full((1, WINDOW), i, dtype=torch.long)
            ci = torch.ones(1, WINDOW)
            acts = torch.ones(1, WINDOW)
            h.reservoir.add(comp_idx, tokens, ci, acts)

        assert h.reservoir.n_items[comp] == k
        assert h.reservoir.n_seen[comp] == k + n_extra

    def test_n_items_never_exceeds_k(self):
        h = _make_harvester()
        k = h.reservoir.k
        comp = 1

        random.seed(0)
        for i in range(k * 10):
            comp_idx = torch.tensor([comp])
            tokens = torch.full((1, WINDOW), i % VOCAB_SIZE, dtype=torch.long)
            ci = torch.ones(1, WINDOW)
            acts = torch.ones(1, WINDOW)
            h.reservoir.add(comp_idx, tokens, ci, acts)

        assert h.reservoir.n_items[comp] == k
        assert h.reservoir.n_seen[comp] == k * 10

    def test_multiple_components_in_one_call(self):
        h = _make_harvester()
        comp_idx = torch.tensor([0, 0, 3, 3, 3])
        tokens = torch.arange(5 * WINDOW).reshape(5, WINDOW)
        ci = torch.ones(5, WINDOW)
        acts = torch.ones(5, WINDOW)
        h.reservoir.add(comp_idx, tokens, ci, acts)

        assert h.reservoir.n_items[0] == 2
        assert h.reservoir.n_seen[0] == 2
        assert h.reservoir.n_items[3] == 3
        assert h.reservoir.n_seen[3] == 3
        # Other components untouched
        assert h.reservoir.n_items[1] == 0
        assert h.reservoir.n_items[2] == 0

    def test_independent_component_tracking(self):
        h = _make_harvester()
        k = h.reservoir.k

        # Fill component 0 to capacity
        for i in range(k):
            h.reservoir.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), i, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )

        # Add one item to component 1
        h.reservoir.add(
            torch.tensor([1]),
            torch.full((1, WINDOW), 99, dtype=torch.long),
            torch.ones(1, WINDOW),
            torch.ones(1, WINDOW),
        )

        assert h.reservoir.n_items[0] == k
        assert h.reservoir.n_seen[0] == k
        assert h.reservoir.n_items[1] == 1
        assert h.reservoir.n_seen[1] == 1

    def test_written_data_matches_input(self):
        h = _make_harvester()
        tokens = torch.tensor([[7, 8, 9]])
        ci = torch.tensor([[0.1, 0.2, 0.3]])
        acts = torch.tensor([[1.0, 2.0, 3.0]])
        h.reservoir.add(torch.tensor([2]), tokens, ci, acts)

        assert torch.equal(h.reservoir.tokens[2, 0], tokens[0])
        assert torch.allclose(h.reservoir.ci[2, 0], ci[0])
        assert torch.allclose(h.reservoir.acts[2, 0], acts[0])


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_all_fields(self, tmp_path: Path):
        h = _make_harvester()

        # Put some data in the harvester
        h.firing_counts[0] = 10.0
        h.firing_counts[3] = 5.0
        h.ci_sums[0] = 2.5
        h.cooccurrence_counts[0, 3] = 7.0
        h.input_cooccurrence[0, 2] = 15
        h.input_marginals[2] = 100
        h.output_cooccurrence[0, 5] = 0.3
        h.output_marginals[5] = 1.0
        h.total_tokens_processed = 500

        # Add a reservoir entry
        h.reservoir.add(
            torch.tensor([0]),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0.9, 0.8, 0.7]]),
            torch.tensor([[1.0, 2.0, 3.0]]),
        )

        path = tmp_path / "harvester.pt"
        h.save(path)
        loaded = Harvester.load(path, device=DEVICE)

        assert loaded.layer_names == h.layer_names
        assert loaded.c_per_layer == h.c_per_layer
        assert loaded.vocab_size == h.vocab_size
        assert loaded.ci_threshold == h.ci_threshold
        assert loaded.max_examples_per_component == h.max_examples_per_component
        assert loaded.context_tokens_per_side == h.context_tokens_per_side
        assert loaded.total_tokens_processed == h.total_tokens_processed
        assert loaded.layer_offsets == h.layer_offsets

        for field in [
            "firing_counts",
            "ci_sums",
            "cooccurrence_counts",
            "input_cooccurrence",
            "input_marginals",
            "output_cooccurrence",
            "output_marginals",
        ]:
            assert torch.equal(getattr(loaded, field), getattr(h, field).cpu()), field

        # Check reservoir fields roundtrip
        assert torch.equal(loaded.reservoir.tokens, h.reservoir.tokens.cpu())
        assert torch.equal(loaded.reservoir.ci, h.reservoir.ci.cpu())
        assert torch.equal(loaded.reservoir.acts, h.reservoir.acts.cpu())
        assert torch.equal(loaded.reservoir.n_items, h.reservoir.n_items.cpu())
        assert torch.equal(loaded.reservoir.n_seen, h.reservoir.n_seen.cpu())

    def test_load_to_specific_device(self, tmp_path: Path):
        h = _make_harvester()
        path = tmp_path / "harvester.pt"
        h.save(path)
        loaded = Harvester.load(path, device=torch.device("cpu"))
        assert loaded.device == torch.device("cpu")
        assert loaded.firing_counts.device == torch.device("cpu")


class TestMerge:
    def test_accumulators_sum(self):
        h1 = _make_harvester()
        h2 = _make_harvester()

        h1.firing_counts[0] = 10.0
        h2.firing_counts[0] = 20.0
        h1.ci_sums[1] = 3.0
        h2.ci_sums[1] = 7.0
        h1.cooccurrence_counts[0, 1] = 5.0
        h2.cooccurrence_counts[0, 1] = 3.0
        h1.input_cooccurrence[0, 2] = 10
        h2.input_cooccurrence[0, 2] = 5
        h1.input_marginals[2] = 100
        h2.input_marginals[2] = 200
        h1.output_cooccurrence[0, 0] = 0.5
        h2.output_cooccurrence[0, 0] = 0.3
        h1.output_marginals[0] = 1.0
        h2.output_marginals[0] = 2.0
        h1.total_tokens_processed = 100
        h2.total_tokens_processed = 200

        h1.merge(h2)

        assert h1.firing_counts[0] == 30.0
        assert h1.ci_sums[1] == 10.0
        assert h1.cooccurrence_counts[0, 1] == 8.0
        assert h1.input_cooccurrence[0, 2] == 15
        assert h1.input_marginals[2] == 300
        assert h1.output_cooccurrence[0, 0] == 0.8
        assert h1.output_marginals[0] == 3.0
        assert h1.total_tokens_processed == 300

    def test_merge_asserts_matching_structure(self):
        h1 = _make_harvester()
        h_different = Harvester(
            layer_names=["other"],
            c_per_layer={"other": 4},
            vocab_size=VOCAB_SIZE,
            ci_threshold=CI_THRESHOLD,
            max_examples_per_component=MAX_EXAMPLES,
            context_tokens_per_side=CONTEXT_TOKENS_PER_SIDE,
            device=DEVICE,
        )
        with pytest.raises(AssertionError):
            h1.merge(h_different)

    def test_merge_reservoir_both_underfilled(self):
        h1 = _make_harvester()
        h2 = _make_harvester()

        # Add 2 items to component 0 in h1
        for i in range(2):
            h1.reservoir.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), i, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )
        # Add 2 items to component 0 in h2
        for i in range(2):
            h2.reservoir.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), 10 + i, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )

        h1.merge(h2)

        # Both underfilled: 2 + 2 = 4, which is < k=5
        assert h1.reservoir.n_items[0] == 4
        assert h1.reservoir.n_seen[0] == 4

    def test_merge_reservoir_n_seen_sums(self):
        h1 = _make_harvester()
        h2 = _make_harvester()
        k = MAX_EXAMPLES

        # Fill h1 to capacity and add more
        random.seed(42)
        for i in range(k + 10):
            h1.reservoir.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), i % VOCAB_SIZE, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )
        # Fill h2 similarly
        for i in range(k + 5):
            h2.reservoir.add(
                torch.tensor([0]),
                torch.full((1, WINDOW), i % VOCAB_SIZE, dtype=torch.long),
                torch.ones(1, WINDOW),
                torch.ones(1, WINDOW),
            )

        seen_before = h1.reservoir.n_seen[0].item() + h2.reservoir.n_seen[0].item()
        h1.merge(h2)

        assert h1.reservoir.n_items[0] == k
        assert h1.reservoir.n_seen[0] == seen_before

    def test_merge_preserves_other_components(self):
        h1 = _make_harvester()
        h2 = _make_harvester()

        h1.reservoir.add(
            torch.tensor([0]),
            torch.full((1, WINDOW), 1, dtype=torch.long),
            torch.ones(1, WINDOW),
            torch.ones(1, WINDOW),
        )
        h2.reservoir.add(
            torch.tensor([3]),
            torch.full((1, WINDOW), 2, dtype=torch.long),
            torch.ones(1, WINDOW),
            torch.ones(1, WINDOW),
        )

        h1.merge(h2)

        assert h1.reservoir.n_items[0] == 1
        assert h1.reservoir.n_items[3] == 1


class TestBuildResults:
    def _make_harvester_with_firings(self) -> Harvester:
        """Create a harvester with some data for build_results testing."""
        h = _make_harvester()

        # Simulate processing: component 0 fires, component 1 fires, rest don't
        h.total_tokens_processed = 100
        h.firing_counts[0] = 10.0
        h.firing_counts[1] = 5.0
        h.ci_sums[0] = 2.0
        h.ci_sums[1] = 1.0

        # Set up token stats so PMI computation works
        h.input_cooccurrence[0, 0] = 8
        h.input_cooccurrence[1, 1] = 3
        h.input_marginals[0] = 50
        h.input_marginals[1] = 30
        h.output_cooccurrence[0, 0] = 5.0
        h.output_cooccurrence[1, 1] = 2.0
        h.output_marginals[0] = 20.0
        h.output_marginals[1] = 15.0

        # Add reservoir examples for component 0
        for i in range(3):
            h.reservoir.add(
                torch.tensor([0]),
                torch.tensor([[i, i + 1, i + 2]]),
                torch.tensor([[0.9, 0.8, 0.7]]),
                torch.tensor([[1.0, 2.0, 3.0]]),
            )

        # Add one reservoir example for component 1
        h.reservoir.add(
            torch.tensor([1]),
            torch.tensor([[5, 6, 7]]),
            torch.tensor([[0.6, 0.7, 0.8]]),
            torch.tensor([[0.5, 1.0, 1.5]]),
        )

        return h

    def test_yields_only_firing_components(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        # Only components 0 and 1 have firings
        keys = {r.component_key for r in results}
        assert keys == {"layer_0:0", "layer_0:1"}

    def test_skips_zero_firing_components(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        # Components 2-7 have zero firings
        keys = {r.component_key for r in results}
        for cidx in range(2, 4):
            assert f"layer_0:{cidx}" not in keys
        for cidx in range(4):
            assert f"layer_1:{cidx}" not in keys

    def test_component_data_structure(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        comp0 = next(r for r in results if r.component_key == "layer_0:0")
        assert comp0.layer == "layer_0"
        assert comp0.component_idx == 0
        assert abs(comp0.mean_ci - 2.0 / 100) < 1e-6
        assert len(comp0.activation_examples) == 3
        assert comp0.input_token_pmi is not None
        assert comp0.output_token_pmi is not None

    def test_activation_examples_have_correct_data(self):
        h = self._make_harvester_with_firings()
        results = list(h.build_results(pmi_top_k_tokens=3))

        comp0 = next(r for r in results if r.component_key == "layer_0:0")
        ex = comp0.activation_examples[0]
        # All tokens are non-sentinel, so all should be kept
        assert len(ex.token_ids) > 0
        assert len(ex.ci_values) == len(ex.token_ids)
        assert len(ex.component_acts) == len(ex.token_ids)

    def test_second_layer_component_keys(self):
        h = _make_harvester()
        h.total_tokens_processed = 100
        # Fire component at flat index 5 = layer_1:1
        h.firing_counts[5] = 8.0
        h.ci_sums[5] = 1.6
        h.input_marginals[0] = 50
        h.input_cooccurrence[5, 0] = 4
        h.output_marginals[0] = 10.0
        h.output_cooccurrence[5, 0] = 2.0

        h.reservoir.add(
            torch.tensor([5]),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0.9, 0.8, 0.7]]),
            torch.tensor([[1.0, 2.0, 3.0]]),
        )

        results = list(h.build_results(pmi_top_k_tokens=3))
        assert len(results) == 1
        assert results[0].component_key == "layer_1:1"
        assert results[0].layer == "layer_1"
        assert results[0].component_idx == 1

    def test_no_results_when_nothing_fires(self):
        h = _make_harvester()
        h.total_tokens_processed = 100
        results = list(h.build_results(pmi_top_k_tokens=3))
        assert results == []

    def test_sentinel_tokens_stripped_from_examples(self):
        h = _make_harvester()
        h.total_tokens_processed = 100
        h.firing_counts[0] = 5.0
        h.ci_sums[0] = 1.0
        h.input_marginals[0] = 50
        h.input_cooccurrence[0, 0] = 3
        h.output_marginals[0] = 10.0
        h.output_cooccurrence[0, 0] = 1.0

        # Manually write a reservoir entry with a sentinel in it
        h.reservoir.tokens[0, 0] = torch.tensor([WINDOW_PAD_SENTINEL, 5, 6])
        h.reservoir.ci[0, 0] = torch.tensor([0.0, 0.8, 0.9])
        h.reservoir.acts[0, 0] = torch.tensor([0.0, 1.0, 2.0])
        h.reservoir.n_items[0] = 1
        h.reservoir.n_seen[0] = 1

        results = list(h.build_results(pmi_top_k_tokens=3))
        assert len(results) == 1
        ex = results[0].activation_examples[0]
        # The sentinel token should be stripped by the mask in build_results
        assert WINDOW_PAD_SENTINEL not in ex.token_ids
        assert len(ex.token_ids) == 2


class TestProcessBatch:
    def test_updates_total_tokens(self):
        h = _make_harvester()
        B, S = 2, 4
        batch = torch.randint(0, VOCAB_SIZE, (B, S))
        ci = torch.zeros(B, S, N_TOTAL)
        output_probs = torch.zeros(B, S, VOCAB_SIZE)
        subcomp_acts = torch.zeros(B, S, N_TOTAL)

        h.process_batch(batch, ci, output_probs, subcomp_acts)
        assert h.total_tokens_processed == B * S

    def test_firing_counts_accumulate(self):
        h = _make_harvester()
        B, S = 1, 2
        batch = torch.zeros(B, S, dtype=torch.long)
        ci = torch.zeros(B, S, N_TOTAL)
        # Component 0 fires at both positions (ci > threshold=0.5)
        ci[0, 0, 0] = 0.9
        ci[0, 1, 0] = 0.8
        output_probs = torch.zeros(B, S, VOCAB_SIZE)
        subcomp_acts = torch.zeros(B, S, N_TOTAL)

        h.process_batch(batch, ci, output_probs, subcomp_acts)
        assert h.firing_counts[0] == 2.0
        assert h.firing_counts[1] == 0.0

    def test_ci_sums_accumulate(self):
        h = _make_harvester()
        B, S = 1, 1
        batch = torch.zeros(B, S, dtype=torch.long)
        ci = torch.zeros(B, S, N_TOTAL)
        ci[0, 0, 2] = 0.75
        output_probs = torch.zeros(B, S, VOCAB_SIZE)
        subcomp_acts = torch.zeros(B, S, N_TOTAL)

        h.process_batch(batch, ci, output_probs, subcomp_acts)
        assert h.ci_sums[2].item() == 0.75

    def test_cooccurrence_counts(self):
        h = _make_harvester()
        B, S = 1, 1
        batch = torch.zeros(B, S, dtype=torch.long)
        ci = torch.zeros(B, S, N_TOTAL)
        # Components 0 and 2 both fire at the same position
        ci[0, 0, 0] = 0.9
        ci[0, 0, 2] = 0.8
        output_probs = torch.zeros(B, S, VOCAB_SIZE)
        subcomp_acts = torch.zeros(B, S, N_TOTAL)

        h.process_batch(batch, ci, output_probs, subcomp_acts)
        assert h.cooccurrence_counts[0, 2] == 1.0
        assert h.cooccurrence_counts[2, 0] == 1.0
        # Self-cooccurrence
        assert h.cooccurrence_counts[0, 0] == 1.0
        assert h.cooccurrence_counts[2, 2] == 1.0


class TestExtractFiringWindows:
    def test_center_window(self):
        batch = torch.tensor([[10, 11, 12, 13, 14]])  # [1, 5]
        ci = torch.zeros(1, 5, 2)
        ci[0, 2, 0] = 0.9
        acts = torch.ones(1, 5, 2)

        tok_w, ci_w, _ = extract_firing_windows(
            batch,
            ci,
            acts,
            batch_idx=torch.tensor([0]),
            seq_idx=torch.tensor([2]),
            comp_idx=torch.tensor([0]),
            context_tokens_per_side=1,
        )

        assert tok_w.shape == (1, 3)
        assert tok_w[0].tolist() == [11, 12, 13]
        assert ci_w[0, 1].item() == pytest.approx(0.9)

    def test_left_boundary_padding(self):
        batch = torch.tensor([[10, 11, 12]])
        ci = torch.zeros(1, 3, 1)
        acts = torch.zeros(1, 3, 1)

        tok_w, _, _ = extract_firing_windows(
            batch,
            ci,
            acts,
            batch_idx=torch.tensor([0]),
            seq_idx=torch.tensor([0]),
            comp_idx=torch.tensor([0]),
            context_tokens_per_side=2,
        )

        assert tok_w.shape == (1, 5)
        assert tok_w[0, 0] == WINDOW_PAD_SENTINEL
        assert tok_w[0, 1] == WINDOW_PAD_SENTINEL
        assert tok_w[0, 2] == 10
        assert tok_w[0, 3] == 11
        assert tok_w[0, 4] == 12

    def test_right_boundary_padding(self):
        batch = torch.tensor([[10, 11, 12]])
        ci = torch.zeros(1, 3, 1)
        acts = torch.zeros(1, 3, 1)

        tok_w, _, _ = extract_firing_windows(
            batch,
            ci,
            acts,
            batch_idx=torch.tensor([0]),
            seq_idx=torch.tensor([2]),
            comp_idx=torch.tensor([0]),
            context_tokens_per_side=2,
        )

        assert tok_w[0, 0] == 10
        assert tok_w[0, 1] == 11
        assert tok_w[0, 2] == 12
        assert tok_w[0, 3] == WINDOW_PAD_SENTINEL
        assert tok_w[0, 4] == WINDOW_PAD_SENTINEL

    def test_multiple_firings(self):
        batch = torch.tensor([[0, 1, 2, 3, 4]])
        ci = torch.zeros(1, 5, 3)
        acts = torch.zeros(1, 5, 3)

        tok_w, _, _ = extract_firing_windows(
            batch,
            ci,
            acts,
            batch_idx=torch.tensor([0, 0]),
            seq_idx=torch.tensor([1, 3]),
            comp_idx=torch.tensor([0, 2]),
            context_tokens_per_side=1,
        )

        assert tok_w.shape == (2, 3)
        assert tok_w[0].tolist() == [0, 1, 2]
        assert tok_w[1].tolist() == [2, 3, 4]

    def test_ci_and_acts_index_correct_component(self):
        batch = torch.tensor([[0, 1, 2]])
        ci = torch.zeros(1, 3, 4)
        ci[0, 1, 2] = 0.5
        acts = torch.zeros(1, 3, 4)
        acts[0, 1, 2] = 7.0

        _, ci_w, act_w = extract_firing_windows(
            batch,
            ci,
            acts,
            batch_idx=torch.tensor([0]),
            seq_idx=torch.tensor([1]),
            comp_idx=torch.tensor([2]),
            context_tokens_per_side=0,
        )

        assert ci_w[0, 0].item() == pytest.approx(0.5)
        assert act_w[0, 0].item() == pytest.approx(7.0)
