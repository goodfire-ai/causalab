"""
pytest unit-tests for LM_units.py

These tests assume:
* model_units.py and featurizers.py have already been patched as in previous
  steps (feature-bounds checks, mutable-default fixes, etc.).
* No actual LLM weights are loaded; we stay entirely on synthetic data.
"""

from __future__ import annotations

import pytest
import torch

import neural.LM_units as LM
import neural.featurizers as F


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng():
    g = torch.Generator()
    g.manual_seed(0)
    return g


# --------------------------------------------------------------------------- #
#  1. Mutable-default featurizer fix                                           #
# --------------------------------------------------------------------------- #
def test_residualstream_featurizers_unique():
    rs1 = LM.ResidualStream(layer=0, token_indices=[0])
    rs2 = LM.ResidualStream(layer=0, token_indices=[0])
    assert rs1.featurizer is not rs2.featurizer


def test_attentionhead_featurizers_unique():
    ah1 = LM.AttentionHead(layer=0, head=5, token_indices=[0])
    ah2 = LM.AttentionHead(layer=0, head=5, token_indices=[0])
    assert ah1.featurizer is not ah2.featurizer


# --------------------------------------------------------------------------- #
#  2. AttentionHead index structure                                            #
# --------------------------------------------------------------------------- #
def test_attentionhead_index_structure():
    ah = LM.AttentionHead(layer=1, head=7, token_indices=[3])
    idx = ah.index_component("dummy")
    assert idx == [[[7]], [[3]]]


# --------------------------------------------------------------------------- #
#  3. Feature-index bounds behaviour                                           #
# --------------------------------------------------------------------------- #
def test_attentionhead_feature_bounds_violation():
    big_feat   = F.SubspaceFeaturizer(shape=(4, 4), trainable=False)   # 4 features
    small_feat = F.SubspaceFeaturizer(shape=(4, 2), trainable=False)   # 2 features

    ah = LM.AttentionHead(
        layer=0,
        head=0,
        token_indices=[0],
        featurizer=big_feat,
        feature_indices=[3],   # valid in 4-dim space
    )

    with pytest.raises(ValueError):
        ah.set_featurizer(small_feat)


# --------------------------------------------------------------------------- #
#  4. get_substring_token_ids tests                                           #
# --------------------------------------------------------------------------- #
class MockTokenizer:
    """Mock tokenizer for testing get_substring_token_ids."""

    def __init__(self, tokens, reconstructed_text=None):
        """
        Parameters
        ----------
        tokens : list of str
            List of token strings that the text will be split into.
        reconstructed_text : str, optional
            What the text looks like when decoded. If None, uses "".join(tokens).
        """
        self.tokens = tokens
        self.reconstructed_text = reconstructed_text or "".join(tokens)
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"

    def convert_tokens_to_ids(self, token):
        return 0

    def __call__(self, texts, **kwargs):
        # Return mock token IDs (just indices)
        return {"input_ids": torch.tensor([[i for i in range(len(self.tokens))]])}

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, list):
            # Decode subset of tokens
            if len(token_ids) == 0:
                return ""
            # Reconstruct progressively for the given range
            result = ""
            for idx, token_id in enumerate(token_ids):
                if token_id < len(self.tokens):
                    result += self.tokens[token_id]
            return result
        return self.reconstructed_text


class MockPipeline:
    """Mock pipeline for testing."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load(self, input_dict, add_special_tokens=False):
        text = input_dict["raw_input"]
        # Just return token indices
        return {"input_ids": torch.tensor([[i for i in range(len(self.tokenizer.tokens))]])}


def test_get_substring_token_ids_single_token():
    """Test substring that aligns exactly with a single token."""
    tokenizer = MockTokenizer(["The", " ", "quick", " ", "fox"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("The quick fox", "quick", pipeline)
    assert result == [2], f"Expected [2], got {result}"


def test_get_substring_token_ids_multiple_tokens():
    """Test substring spanning multiple complete tokens."""
    tokenizer = MockTokenizer(["The", " ", "quick", " ", "brown", " ", "fox"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("The quick brown fox", "quick brown", pipeline)
    assert result == [2, 3, 4], f"Expected [2, 3, 4], got {result}"


def test_get_substring_token_ids_partial_start():
    """Test substring starting in the middle of a token."""
    tokenizer = MockTokenizer(["hello", "world"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("helloworld", "loworld", pipeline)
    # Should include both tokens since substring overlaps both
    assert result == [0, 1], f"Expected [0, 1], got {result}"


def test_get_substring_token_ids_partial_end():
    """Test substring ending in the middle of a token."""
    tokenizer = MockTokenizer(["hello", "world"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("helloworld", "hellowor", pipeline)
    # Should include both tokens
    assert result == [0, 1], f"Expected [0, 1], got {result}"


def test_get_substring_token_ids_partial_both():
    """Test substring both starting and ending mid-token."""
    tokenizer = MockTokenizer(["abc", "def", "ghi"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("abcdefghi", "cdefg", pipeline)
    # Should include all three tokens
    assert result == [0, 1, 2], f"Expected [0, 1, 2], got {result}"


def test_get_substring_token_ids_single_char():
    """Test single character substring."""
    tokenizer = MockTokenizer(["The", " ", "cat"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("The cat", "c", pipeline)
    assert result == [2], f"Expected [2], got {result}"


def test_get_substring_token_ids_with_spaces():
    """Test substring with leading/trailing spaces."""
    tokenizer = MockTokenizer(["The", " ", "quick", " ", "fox"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("The quick fox", " quick ", pipeline)
    assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"


def test_get_substring_token_ids_empty_substring():
    """Test that empty substring raises ValueError."""
    tokenizer = MockTokenizer(["hello"])
    pipeline = MockPipeline(tokenizer)

    with pytest.raises(ValueError, match="Substring cannot be empty"):
        LM.get_substring_token_ids("hello", "", pipeline)


def test_get_substring_token_ids_empty_text():
    """Test that empty text raises ValueError."""
    tokenizer = MockTokenizer([])
    pipeline = MockPipeline(tokenizer)

    with pytest.raises(ValueError, match="Text cannot be empty"):
        LM.get_substring_token_ids("", "test", pipeline)


def test_get_substring_token_ids_substring_not_found():
    """Test that substring not in text raises ValueError."""
    tokenizer = MockTokenizer(["hello", "world"])
    pipeline = MockPipeline(tokenizer)

    with pytest.raises(ValueError, match="not found in text"):
        LM.get_substring_token_ids("helloworld", "xyz", pipeline)


def test_get_substring_token_ids_full_text():
    """Test substring that is the entire text."""
    tokenizer = MockTokenizer(["hello", "world"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("helloworld", "helloworld", pipeline)
    assert result == [0, 1], f"Expected [0, 1], got {result}"


def test_get_substring_token_ids_first_token_only():
    """Test substring matching only the first token."""
    tokenizer = MockTokenizer(["The", " ", "end"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("The end", "The", pipeline)
    assert result == [0], f"Expected [0], got {result}"


def test_get_substring_token_ids_last_token_only():
    """Test substring matching only the last token."""
    tokenizer = MockTokenizer(["The", " ", "end"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("The end", "end", pipeline)
    assert result == [2], f"Expected [2], got {result}"


def test_get_substring_token_ids_multiple_occurrences():
    """Test substring that appears multiple times (should match first)."""
    tokenizer = MockTokenizer(["cat", " ", "and", " ", "cat"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("cat and cat", "cat", pipeline)
    # Should match the first occurrence at token 0
    assert result == [0], f"Expected [0], got {result}"


def test_get_substring_token_ids_multiple_spaces():
    """Test substring with multiple consecutive spaces."""
    tokenizer = MockTokenizer(["The", "  ", "cat"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("The  cat", "  cat", pipeline)
    assert result == [1, 2], f"Expected [1, 2], got {result}"


def test_get_substring_token_ids_punctuation():
    """Test substring with punctuation."""
    tokenizer = MockTokenizer(["Hello", ",", " ", "world", "!"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("Hello, world!", ", world!", pipeline)
    assert result == [1, 2, 3, 4], f"Expected [1, 2, 3, 4], got {result}"


def test_get_substring_token_ids_single_char_from_multichar_token():
    """Test single character from middle of a multi-character token."""
    tokenizer = MockTokenizer(["hello"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("hello", "l", pipeline)
    # Should include the whole token even though it's just one char
    assert result == [0], f"Expected [0], got {result}"


def test_get_substring_token_ids_all_but_one_char():
    """Test substring that's all but one character of a token."""
    tokenizer = MockTokenizer(["hello", "world"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("helloworld", "elloworld", pipeline)
    # Should include both tokens since substring overlaps both
    assert result == [0, 1], f"Expected [0, 1], got {result}"


def test_get_substring_token_ids_whitespace_mismatch():
    """Test tokenizer that normalizes whitespace differently."""
    # Tokenizer removes extra spaces when decoding
    tokenizer = MockTokenizer(["The", " ", "cat"], reconstructed_text="The cat")
    pipeline = MockPipeline(tokenizer)

    # Original text has different spacing than reconstructed
    # This should still work with whitespace normalization fallback
    result = LM.get_substring_token_ids("The cat", "cat", pipeline)
    assert result == [2], f"Expected [2], got {result}"


def test_get_substring_token_ids_adjacent_identical_tokens():
    """Test with adjacent identical tokens."""
    tokenizer = MockTokenizer(["la", "la", "la"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("lalala", "lala", pipeline)
    # Should match first two tokens
    assert result == [0, 1], f"Expected [0, 1], got {result}"


def test_get_substring_token_ids_substring_longer_than_original():
    """Test substring that's supposedly longer than the text."""
    tokenizer = MockTokenizer(["hi"])
    pipeline = MockPipeline(tokenizer)

    with pytest.raises(ValueError, match="not found in text"):
        LM.get_substring_token_ids("hi", "hi there", pipeline)


def test_get_substring_token_ids_very_short_tokens():
    """Test with many single-character tokens."""
    tokenizer = MockTokenizer(["a", "b", "c", "d", "e"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("abcde", "bcd", pipeline)
    assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"


def test_get_substring_token_ids_token_split_mid_word():
    """Test where word is split in unusual way by tokenizer."""
    # Example: "reading" split as "read", "ing"
    tokenizer = MockTokenizer(["read", "ing"], reconstructed_text="reading")
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("reading", "reading", pipeline)
    # Should include both tokens
    assert result == [0, 1], f"Expected [0, 1], got {result}"

    # Test substring that matches exactly one token
    result2 = LM.get_substring_token_ids("reading", "read", pipeline)
    assert result2 == [0], f"Expected [0], got {result2}"

    # Test substring that matches the other token
    result3 = LM.get_substring_token_ids("reading", "ing", pipeline)
    assert result3 == [1], f"Expected [1], got {result3}"


def test_get_substring_token_ids_overlapping_boundary():
    """Test substring that barely touches two tokens."""
    tokenizer = MockTokenizer(["abc", "def"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("abcdef", "cd", pipeline)
    # Should include both tokens since 'c' is in first, 'd' is in second
    assert result == [0, 1], f"Expected [0, 1], got {result}"


def test_get_substring_token_ids_newline_handling():
    """Test with newlines in text."""
    tokenizer = MockTokenizer(["Hello", "\n", "World"])
    pipeline = MockPipeline(tokenizer)

    result = LM.get_substring_token_ids("Hello\nWorld", "\nWorld", pipeline)
    assert result == [1, 2], f"Expected [1, 2], got {result}"


# --------------------------------------------------------------------------- #
#  5. Integration tests with real tokenizers (marked as slow)                #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestGetSubstringTokenIdsRealTokenizers:
    """Integration tests using actual HuggingFace tokenizers."""

    @pytest.fixture(scope="class")
    def gpt2_pipeline(self):
        """Load a real GPT-2 pipeline for testing."""
        from neural.pipeline import LMPipeline
        try:
            pipeline = LMPipeline("gpt2", max_new_tokens=1)
            return pipeline
        except Exception as e:
            pytest.skip(f"Could not load GPT-2 model: {e}")

    @pytest.fixture(scope="class")
    def olmo_pipeline(self):
        """Load a real OLMo pipeline for testing."""
        from neural.pipeline import LMPipeline
        try:
            pipeline = LMPipeline("allenai/OLMo-1B-hf", max_new_tokens=1)
            return pipeline
        except Exception as e:
            pytest.skip(f"Could not load OLMo model: {e}")

    def test_gpt2_simple_text(self, gpt2_pipeline):
        """Test with GPT-2 tokenizer on simple text."""
        text = "The quick brown fox"
        substring = "quick"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        # Verify result is a list of integers
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        assert len(result) > 0

        # Verify the tokens actually contain the substring
        token_ids = gpt2_pipeline.load({"raw_input": text})["input_ids"][0]
        selected_tokens = [token_ids[i].item() for i in result]
        decoded = gpt2_pipeline.tokenizer.decode(selected_tokens)

        # The decoded tokens should contain or be part of "quick"
        assert "quick" in decoded.lower() or decoded.lower() in "quick"

    def test_gpt2_with_spaces(self, gpt2_pipeline):
        """Test GPT-2 tokenizer with leading spaces (Ġ encoding)."""
        text = "Hello world from GPT"
        substring = " world"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        assert isinstance(result, list)
        assert len(result) > 0

        # Verify the selection includes the space
        token_ids = gpt2_pipeline.load({"raw_input": text})["input_ids"][0]
        selected_tokens = [token_ids[i].item() for i in result]
        decoded = gpt2_pipeline.tokenizer.decode(selected_tokens)

        # Should contain "world" with the space
        assert "world" in decoded.lower()

    def test_gpt2_partial_token_overlap(self, gpt2_pipeline):
        """Test with substring that spans partial tokens."""
        text = "unbelievable"
        # Pick a substring that's likely to span multiple BPE tokens
        substring = "liev"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        assert isinstance(result, list)
        assert len(result) > 0

        # Verify tokens include the substring
        token_ids = gpt2_pipeline.load({"raw_input": text})["input_ids"][0]
        selected_tokens = [token_ids[i].item() for i in result]
        decoded = gpt2_pipeline.tokenizer.decode(selected_tokens)

        assert "liev" in decoded.lower()

    def test_gpt2_multi_word_substring(self, gpt2_pipeline):
        """Test with multi-word substring."""
        text = "The quick brown fox jumps over the lazy dog"
        substring = "fox jumps over"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        assert isinstance(result, list)
        assert len(result) >= 3  # At least 3 tokens for these words

        # Verify all words are covered
        token_ids = gpt2_pipeline.load({"raw_input": text})["input_ids"][0]
        selected_tokens = [token_ids[i].item() for i in result]
        decoded = gpt2_pipeline.tokenizer.decode(selected_tokens)

        assert "fox" in decoded.lower()
        assert "jump" in decoded.lower()  # Could be "jumps" or "jump"
        assert "over" in decoded.lower()

    def test_gpt2_punctuation(self, gpt2_pipeline):
        """Test with punctuation."""
        text = "Hello, world! How are you?"
        substring = ", world!"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        assert isinstance(result, list)
        assert len(result) > 0

        token_ids = gpt2_pipeline.load({"raw_input": text})["input_ids"][0]
        selected_tokens = [token_ids[i].item() for i in result]
        decoded = gpt2_pipeline.tokenizer.decode(selected_tokens)

        assert "world" in decoded.lower()

    def test_gpt2_single_character(self, gpt2_pipeline):
        """Test with single character substring."""
        text = "The cat sat"
        substring = "c"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_olmo_simple_text(self, olmo_pipeline):
        """Test with OLMo tokenizer on simple text."""
        text = "The quick brown fox"
        substring = "quick"

        result = LM.get_substring_token_ids(text, substring, olmo_pipeline)

        # Verify result is a list of integers
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        assert len(result) > 0

        # Verify the tokens actually contain the substring
        token_ids = olmo_pipeline.load({"raw_input": text})["input_ids"][0]
        selected_tokens = [token_ids[i].item() for i in result]
        decoded = olmo_pipeline.tokenizer.decode(selected_tokens)

        # The decoded tokens should contain or be part of "quick"
        assert "quick" in decoded.lower() or decoded.lower() in "quick"

    def test_olmo_with_spaces(self, olmo_pipeline):
        """Test OLMo tokenizer with spaces."""
        text = "Hello world from OLMo"
        substring = " world"

        result = LM.get_substring_token_ids(text, substring, olmo_pipeline)

        assert isinstance(result, list)
        assert len(result) > 0

        # Verify the selection includes the space
        token_ids = olmo_pipeline.load({"raw_input": text})["input_ids"][0]
        selected_tokens = [token_ids[i].item() for i in result]
        decoded = olmo_pipeline.tokenizer.decode(selected_tokens)

        # Should contain "world" with the space
        assert "world" in decoded.lower()

    def test_multiple_tokenizers_consistency(self, gpt2_pipeline, olmo_pipeline):
        """Test that the function works consistently across different tokenizers."""
        text = "The quick brown fox"
        substring = "brown"

        # Both should return valid results
        gpt2_result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)
        olmo_result = LM.get_substring_token_ids(text, substring, olmo_pipeline)

        # Both should be non-empty lists
        assert len(gpt2_result) > 0
        assert len(olmo_result) > 0

        # Verify both selections decode to contain the substring
        gpt2_tokens = gpt2_pipeline.load({"raw_input": text})["input_ids"][0]
        gpt2_selected = [gpt2_tokens[i].item() for i in gpt2_result]
        gpt2_decoded = gpt2_pipeline.tokenizer.decode(gpt2_selected)

        olmo_tokens = olmo_pipeline.load({"raw_input": text})["input_ids"][0]
        olmo_selected = [olmo_tokens[i].item() for i in olmo_result]
        olmo_decoded = olmo_pipeline.tokenizer.decode(olmo_selected)

        assert "brown" in gpt2_decoded.lower()
        assert "brown" in olmo_decoded.lower()

    def test_gpt2_edge_case_first_token(self, gpt2_pipeline):
        """Test selecting substring at the very start."""
        text = "The quick brown fox"
        substring = "The"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        assert isinstance(result, list)
        assert len(result) > 0
        assert 0 in result  # Should include the first token

    def test_gpt2_edge_case_last_token(self, gpt2_pipeline):
        """Test selecting substring at the very end."""
        text = "The quick brown fox"
        substring = "fox"

        result = LM.get_substring_token_ids(text, substring, gpt2_pipeline)

        assert isinstance(result, list)
        assert len(result) > 0

        # Should include the last token
        token_ids = gpt2_pipeline.load({"raw_input": text})["input_ids"][0]
        assert (len(token_ids) - 1) in result
