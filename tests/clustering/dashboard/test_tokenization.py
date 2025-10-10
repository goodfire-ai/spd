"""Tests for fast tokenization utilities."""

import numpy as np
import pytest
import torch
from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.clustering.dashboard.core.tokenization import (
    attach_vocab_arr,
    simple_batch_decode,
)

# pyright: reportAttributeAccessIssue=false, reportUnknownParameterType=false


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    """Load SimpleStories tokenizer for testing."""
    return AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")


@pytest.fixture
def tokenizer_with_vocab(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Tokenizer with vocab_arr already attached."""
    attach_vocab_arr(tokenizer)
    return tokenizer


class TestAttachVocabArr:
    """Test attach_vocab_arr function."""

    def test_creates_vocab_arr_attribute(self, tokenizer: PreTrainedTokenizer):
        """Test that vocab_arr attribute is created."""
        assert not hasattr(tokenizer, "vocab_arr")
        attach_vocab_arr(tokenizer)
        assert hasattr(tokenizer, "vocab_arr")

    def test_vocab_arr_shape(self, tokenizer: PreTrainedTokenizer):
        """Test vocab_arr has correct shape."""
        attach_vocab_arr(tokenizer)
        assert tokenizer.vocab_arr.shape == (tokenizer.vocab_size,)

    def test_vocab_arr_dtype(self, tokenizer: PreTrainedTokenizer):
        """Test vocab_arr has bytes dtype."""
        attach_vocab_arr(tokenizer)
        # Should be a fixed-length bytes string (S<length>)
        assert tokenizer.vocab_arr.dtype.kind == "U"

    def test_vocab_arr_matches_convert_ids_to_tokens(self, tokenizer: PreTrainedTokenizer):
        """Test that vocab_arr entries match convert_ids_to_tokens."""
        attach_vocab_arr(tokenizer)

        # Test first 100 tokens
        for token_id in range(min(100, tokenizer.vocab_size)):
            expected = tokenizer.convert_ids_to_tokens(token_id)
            actual = tokenizer.vocab_arr[token_id]
            assert actual == expected, f"Mismatch at token_id={token_id}"

    def test_vocab_arr_all_tokens_retrievable(self, tokenizer: PreTrainedTokenizer):
        """Test that all tokens can be retrieved from vocab_arr."""
        attach_vocab_arr(tokenizer)

        # Test random sample of tokens
        rng = np.random.default_rng(seed=42)
        sample_ids = rng.integers(0, tokenizer.vocab_size, size=50)

        for token_id in sample_ids:
            # Convert numpy int to Python int
            expected = tokenizer.convert_ids_to_tokens(int(token_id))
            actual = tokenizer.vocab_arr[token_id]
            assert actual == expected

    def test_idempotent(self, tokenizer: PreTrainedTokenizer):
        """Test that calling attach_vocab_arr twice doesn't break."""
        attach_vocab_arr(tokenizer)
        vocab_arr_1 = tokenizer.vocab_arr.copy()

        attach_vocab_arr(tokenizer)
        vocab_arr_2 = tokenizer.vocab_arr

        np.testing.assert_array_equal(vocab_arr_1, vocab_arr_2)


class TestSimpleBatchDecode:
    """Test simple_batch_decode function."""

    def test_requires_vocab_arr(self, tokenizer: PreTrainedTokenizer):
        """Test that simple_batch_decode fails without vocab_arr."""
        batch = np.array([[1, 2, 3]])
        with pytest.raises(AssertionError, match="vocab_arr"):
            simple_batch_decode(tokenizer, batch)

    def test_basic_decoding(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test basic batch decoding."""
        # Create a simple batch
        batch = np.array([[10, 20, 30], [40, 50, 60]])

        result = simple_batch_decode(tokenizer_with_vocab, batch)

        # Check shape matches input
        assert result.shape == batch.shape

        # Check dtype is bytes
        assert result.dtype.kind == "U"

    def test_matches_convert_ids_to_tokens(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test that output matches tokenizer.convert_ids_to_tokens."""
        batch = np.array([[10, 20, 30], [40, 50, 60]])

        result = simple_batch_decode(tokenizer_with_vocab, batch)

        # Compare with expected output
        for i, seq in enumerate(batch):
            for j, token_id in enumerate(seq):
                # Convert numpy int to Python int
                expected = tokenizer_with_vocab.convert_ids_to_tokens(int(token_id))
                actual = result[i, j]
                assert actual == expected

    def test_with_random_batches(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test with random token IDs."""
        rng = np.random.default_rng(seed=42)
        vocab_size = tokenizer_with_vocab.vocab_size

        # Test various batch sizes
        for batch_size, seq_len in [(1, 10), (4, 8), (16, 32)]:
            batch = rng.integers(0, vocab_size, size=(batch_size, seq_len))
            result = simple_batch_decode(tokenizer_with_vocab, batch)

            assert result.shape == batch.shape

            # Spot check a few tokens
            for i in range(min(3, batch_size)):
                for j in range(min(3, seq_len)):
                    # Convert numpy int to Python int
                    expected = tokenizer_with_vocab.convert_ids_to_tokens(int(batch[i, j]))
                    actual = result[i, j]
                    assert actual == expected

    def test_with_torch_tensor(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test that it works with torch tensors."""
        batch_tensor: Int[Tensor, "batch seq"] = torch.tensor([[10, 20, 30], [40, 50, 60]])

        # Convert to numpy
        batch_np = batch_tensor.numpy()
        result = simple_batch_decode(tokenizer_with_vocab, batch_np)

        assert result.shape == batch_tensor.shape

    def test_single_sequence(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test with a single sequence."""
        batch = np.array([[5, 10, 15, 20]])
        result = simple_batch_decode(tokenizer_with_vocab, batch)

        assert result.shape == (1, 4)

    def test_special_tokens(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test with special tokens (BOS, EOS, PAD, etc)."""
        # Get special token IDs
        special_ids = []
        if hasattr(tokenizer_with_vocab, "bos_token_id") and tokenizer_with_vocab.bos_token_id:
            special_ids.append(tokenizer_with_vocab.bos_token_id)
        if hasattr(tokenizer_with_vocab, "eos_token_id") and tokenizer_with_vocab.eos_token_id:
            special_ids.append(tokenizer_with_vocab.eos_token_id)
        if hasattr(tokenizer_with_vocab, "pad_token_id") and tokenizer_with_vocab.pad_token_id:
            special_ids.append(tokenizer_with_vocab.pad_token_id)

        if special_ids:
            batch = np.array([special_ids])
            result = simple_batch_decode(tokenizer_with_vocab, batch)

            for i, token_id in enumerate(special_ids):
                expected = tokenizer_with_vocab.convert_ids_to_tokens(token_id)
                actual = result[0, i]
                assert actual == expected


class TestRoundTripping:
    """Test encoding then decoding produces expected results."""

    def test_encode_decode_round_trip(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test encoding text then decoding produces original tokens."""
        text = "Hello world, how are you?"

        # Encode to token IDs
        token_ids = tokenizer_with_vocab.encode(text, return_tensors="np")  # type: ignore

        # Decode using simple_batch_decode
        decoded_tokens = simple_batch_decode(tokenizer_with_vocab, token_ids)

        # Compare with expected tokens
        expected_tokens = tokenizer_with_vocab.convert_ids_to_tokens(token_ids[0])

        for i, expected in enumerate(expected_tokens):
            actual = decoded_tokens[0, i]
            assert actual == expected

    def test_batch_encode_decode(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test batch encoding then decoding."""
        texts = ["Hello world", "This is a test", "Another example"]

        # Set pad token if not present
        if tokenizer_with_vocab.pad_token is None:
            tokenizer_with_vocab.pad_token = tokenizer_with_vocab.eos_token

        # Encode to token IDs
        encoded = tokenizer_with_vocab(texts, padding=True, truncation=True, return_tensors="np")  # pyright: ignore[reportCallIssue]
        token_ids = encoded["input_ids"]  # type: ignore

        # Decode using simple_batch_decode
        decoded_tokens = simple_batch_decode(tokenizer_with_vocab, token_ids)

        # Verify shape
        assert decoded_tokens.shape == token_ids.shape

        # Compare with expected tokens for each sequence
        for i, token_seq in enumerate(token_ids):
            expected_tokens = tokenizer_with_vocab.convert_ids_to_tokens(list(token_seq))
            for j, expected in enumerate(expected_tokens):
                actual = decoded_tokens[i, j]
                assert actual == expected


class TestPerformance:
    """Test performance characteristics (not strict timing, just sanity checks)."""

    def test_handles_large_vocab(self, tokenizer: PreTrainedTokenizer):
        """Test that attach_vocab_arr works with large vocabularies."""
        attach_vocab_arr(tokenizer)
        # Just verify it completes without error
        assert tokenizer.vocab_arr.shape[0] == tokenizer.vocab_size

    def test_handles_large_batches(self, tokenizer_with_vocab: PreTrainedTokenizer):
        """Test that simple_batch_decode handles large batches."""
        rng = np.random.default_rng(seed=42)
        # Large batch
        batch = rng.integers(0, tokenizer_with_vocab.vocab_size, size=(128, 512))

        result = simple_batch_decode(tokenizer_with_vocab, batch)
        assert result.shape == batch.shape
