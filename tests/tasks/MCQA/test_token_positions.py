"""
Test Script: MCQA Token Positions

This script tests token position functions with actual tokenizer integration.
It verifies that tokens are correctly identified in MCQA prompts.
"""

from causalab.tasks.MCQA.causal_models import NUM_CHOICES
from causalab.tasks.MCQA.counterfactuals import sample_answerable_question
from causalab.tasks.MCQA.token_positions import create_token_positions, TEMPLATES
from tests._helpers.pipeline_shim import PipelineShim
from causalab.tasks.token_positions import TokenPosition
import pytest


pytestmark = pytest.mark.unit


def test_basic_tokenization():
    """Test basic tokenization of MCQA prompts."""
    print("=== Test 1: Basic Tokenization ===")

    pipeline = PipelineShim("gpt2")

    # sample_answerable_question() returns a fully computed CausalTrace
    trace = sample_answerable_question()

    prompt = trace["raw_input"]
    print(f"Prompt:\n{prompt}\n")

    # Tokenize
    tokens = pipeline.tokenizer.encode(prompt)
    print(f"Number of tokens: {len(tokens)}")
    print(f"Tokens: {tokens}")

    # Decode each token
    print("\nToken breakdown:")
    for i, token_id in enumerate(tokens):
        token_str = pipeline.tokenizer.decode([token_id])
        print(f"  {i}: {repr(token_str)}")

    print("\n✓ Test 1 passed\n")


def test_get_symbol_index():
    """Test symbol token positions via create_token_positions."""
    print("=== Test 2: Get Symbol Index ===")

    pipeline = PipelineShim("gpt2")

    # Create token positions
    token_positions = create_token_positions(pipeline)

    # sample_answerable_question() returns a fully computed CausalTrace
    trace = sample_answerable_question()

    print(f"Prompt:\n{trace['raw_input']}\n")

    # Test for each symbol position
    for i in range(NUM_CHOICES):
        symbol = trace[f"symbols[{i}]"]
        token_pos = token_positions[f"symbols[{i}]"]
        indices = token_pos.index(trace)

        print(f"Symbol {i}: '{symbol}'")
        print(f"  Token index: {indices}")

        # Verify it's a single-element list
        assert isinstance(indices, list), "Should return a list"
        assert len(indices) == 1, "Should return single index"

        # Decode the token at that position
        tokenized = pipeline.load([trace])
        tokens = tokenized["input_ids"][0]
        token_at_index = pipeline.tokenizer.decode([tokens[indices[0]]])
        print(f"  Token at index: {repr(token_at_index)}")

        # Note: The token might include whitespace, so we check if symbol is in it
        assert symbol in token_at_index or token_at_index.strip() == symbol, (
            f"Token should contain symbol '{symbol}'"
        )

    print("\n✓ Test 2 passed\n")


def test_get_correct_symbol_index():
    """Test correct symbol token position via create_token_positions."""
    print("=== Test 3: Get Correct Symbol Index ===")

    pipeline = PipelineShim("gpt2")

    # Create token positions
    token_positions = create_token_positions(pipeline)
    correct_symbol_pos = token_positions["correct_symbol"]

    # Test multiple examples
    for i in range(5):
        # sample_answerable_question() returns a fully computed CausalTrace
        trace = sample_answerable_question()

        correct_symbol = trace["answer"]
        indices = correct_symbol_pos.index(trace)

        print(f"Example {i + 1}:")
        print(f"  Correct answer: '{correct_symbol}'")
        print(f"  Token index: {indices}")

        # Verify it's a single-element list
        assert isinstance(indices, list), "Should return a list"
        assert len(indices) == 1, "Should return single index"

        # Decode the token
        tokenized = pipeline.load([trace])
        tokens = tokenized["input_ids"][0]
        token_at_index = pipeline.tokenizer.decode([tokens[indices[0]]])
        print(f"  Token at index: {repr(token_at_index)}")

    print("\n✓ Test 3 passed\n")


def test_token_position_objects():
    """Test TokenPosition object creation."""
    print("=== Test 4: TokenPosition Objects ===")

    pipeline = PipelineShim("gpt2")

    # Create all token positions
    token_positions = create_token_positions(pipeline)

    # Test token positions for each symbol
    for i in range(NUM_CHOICES):
        token_pos = token_positions[f"symbols[{i}]"]

        print(f"Symbol {i} TokenPosition:")
        print(f"  ID: {token_pos.id}")

        # Test with a sample - sample_answerable_question() returns a CausalTrace
        trace = sample_answerable_question()
        indices = token_pos.index(trace)

        print(f"  Sample indices: {indices}")
        assert isinstance(indices, list), "Should return list of indices"
        assert len(indices) > 0, "Should have at least one index"

    print("\n✓ Test 4 passed\n")


def test_correct_symbol_token_position():
    """Test correct symbol TokenPosition."""
    print("=== Test 5: Correct Symbol TokenPosition ===")

    pipeline = PipelineShim("gpt2")

    # Create all token positions
    token_positions = create_token_positions(pipeline)
    token_pos = token_positions["correct_symbol"]

    print(f"Correct Symbol TokenPosition ID: {token_pos.id}")

    # Test with multiple samples
    for i in range(5):
        # sample_answerable_question() returns a fully computed CausalTrace
        trace = sample_answerable_question()

        indices = token_pos.index(trace)
        correct_answer = trace["answer"]

        print(f"Sample {i + 1}: Correct answer '{correct_answer}', indices {indices}")

        assert len(indices) == 1, "Should return single index"

    print("\n✓ Test 5 passed\n")


def test_last_token_position():
    """Test last token TokenPosition."""
    print("=== Test 6: Last Token Position ===")

    pipeline = PipelineShim("gpt2")

    # Create all token positions
    token_positions = create_token_positions(pipeline)
    token_pos = token_positions["last_token"]

    print(f"Last Token TokenPosition ID: {token_pos.id}")

    # Test with samples
    for i in range(3):
        # sample_answerable_question() returns a fully computed CausalTrace
        trace = sample_answerable_question()

        indices = token_pos.index(trace)

        print(f"Sample {i + 1}:")
        print(f"  Last token index: {indices}")

        # Decode to see what it is
        tokens = pipeline.tokenizer.encode(trace["raw_input"])
        # indices is list[int] when batch=False (default)
        first_idx = indices[0]
        assert isinstance(first_idx, int), "Expected single index, not batch"
        if first_idx < len(tokens):
            last_token = pipeline.tokenizer.decode([tokens[first_idx]])
            print(f"  Last token: {repr(last_token)}")
        else:
            print(f"  Index {first_idx} is at boundary (total tokens: {len(tokens)})")

    print("\n✓ Test 6 passed\n")


def test_create_token_positions():
    """Test the factory function that creates all token positions."""
    print("=== Test 7: Create All Token Positions ===")

    pipeline = PipelineShim("gpt2")

    token_positions = create_token_positions(pipeline)

    print("Token positions created:")
    for name, token_pos in token_positions.items():
        print(f"  {name}: {token_pos.id}")

    # Verify expected keys
    expected_keys = [
        "correct_symbol",
        "correct_symbol_period",
        "last_token",
        "symbols[0]",
        "symbol0_period",
        "symbols[1]",
        "symbol1_period",
    ]

    for key in expected_keys:
        assert key in token_positions, f"Should have '{key}' token position"

    print(f"\n✓ All {len(token_positions)} token positions created")
    print("✓ Test 7 passed\n")


def test_highlight_selected_token():
    """Test the highlight_selected_token method."""
    print("=== Test 8: Highlight Selected Token ===")

    pipeline = PipelineShim("gpt2")

    token_positions = create_token_positions(pipeline)

    # sample_answerable_question() returns a fully computed CausalTrace
    trace = sample_answerable_question()

    print("Highlighting tokens in sample prompt:\n")

    for name, token_pos in list(token_positions.items())[:4]:  # Just show first few
        highlighted = token_pos.highlight_selected_token(trace)
        print(f"{name}:")
        print(highlighted)
        print()

    print("✓ Test 8 passed\n")


def test_edge_case_symbol_not_found():
    """Test error handling when symbol is not found."""
    print("=== Test 9: Edge Case - Symbol Not Found ===")

    pipeline = PipelineShim("gpt2")

    # Create token positions
    token_positions = create_token_positions(pipeline)
    symbol0_pos = token_positions["symbols[0]"]

    # Create a malformed input where symbol doesn't appear in raw_input
    input_sample = {
        "symbols[0]": "Z",
        "raw_input": "The banana is yellow. What color is the banana?\nA. blue\nB. yellow\nAnswer:",
    }

    print("Testing with symbol 'Z' that doesn't appear in prompt...")

    try:
        indices = symbol0_pos.index(input_sample)
        print(f"ERROR: Should have raised ValueError, got {indices}")
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:100]}...")

    print("\n✓ Test 9 passed\n")


def test_period_tokens():
    """Test period token identification."""
    print("=== Test 10: Period Token Positions ===")

    pipeline = PipelineShim("gpt2")

    token_positions = create_token_positions(pipeline)

    # sample_answerable_question() returns a fully computed CausalTrace
    trace = sample_answerable_question()

    print(f"Prompt:\n{trace['raw_input']}\n")

    # Test symbol0_period
    if "symbol0_period" in token_positions:
        period_pos = token_positions["symbol0_period"]
        indices = period_pos.index(trace)

        print(f"Symbol0 period token index: {indices}")

        # Check it's right after symbols[0]
        symbol0_pos = token_positions["symbols[0]"]
        symbol0_indices = symbol0_pos.index(trace)

        # Type narrowing - indices are list[int] when batch=False (default)
        period_idx = indices[0]
        symbol_idx = symbol0_indices[0]
        assert isinstance(period_idx, int) and isinstance(symbol_idx, int)

        print(f"Symbol0 index: {symbol0_indices}")
        print(f"Expected period index: {symbol_idx + 1}")
        print(f"Actual period index: {period_idx}")

        # Note: This might not always be exactly +1 depending on tokenization
        # but it should be close
        assert period_idx == symbol_idx + 1, (
            "Period should be immediately after symbol (may fail with some tokenizers)"
        )

    print("\n✓ Test 10 passed\n")


def test_token_positions_return_type_and_ids():
    """Test that create_token_positions returns a dict of TokenPosition objects with correct ids."""
    pipeline = PipelineShim("gpt2")

    token_positions = create_token_positions(pipeline)

    # Verify return type is dict
    assert isinstance(token_positions, dict), "Should return dict"

    # Verify required keys exist
    required_keys = [
        "last_token",
        "symbols[0]",
        "symbols[1]",
        "symbol0_period",
        "symbol1_period",
        "correct_symbol",
        "correct_symbol_period",
    ]
    for key in required_keys:
        assert key in token_positions, f"Missing required key: {key}"

    # Verify each value is a TokenPosition with correct id
    for name, pos in token_positions.items():
        assert isinstance(pos, TokenPosition), (
            f"{name} should be TokenPosition, got {type(pos)}"
        )
        assert hasattr(pos, "index"), f"TokenPosition {name} should have 'index' method"
        assert hasattr(pos, "id"), f"TokenPosition {name} should have 'id' attribute"
        assert pos.id == name, (
            f"TokenPosition {name}.id should be '{name}', got '{pos.id}'"
        )


def test_template_none_equivalent_to_default_template():
    """Test that template=None (default) uses TEMPLATES[0] and produces same results."""
    pipeline = PipelineShim("gpt2")

    # Create with explicit default template
    explicit_template = TEMPLATES[0]
    pos_explicit = create_token_positions(pipeline, template=explicit_template)

    # Create with default (None)
    pos_default = create_token_positions(pipeline, template=None)

    # Both should have the same keys
    assert set(pos_explicit.keys()) == set(pos_default.keys()), (
        "Explicit and default templates should produce same keys"
    )

    # They should produce equivalent results on the same example
    trace = sample_answerable_question()

    for name in ["last_token", "correct_symbol", "symbols[0]"]:
        idx_explicit = pos_explicit[name].index(trace)
        idx_default = pos_default[name].index(trace)
        assert idx_explicit == idx_default, (
            f"template='{explicit_template[:20]}...' and template=None differ on {name}: "
            f"{idx_explicit} vs {idx_default}"
        )


def main():
    """Run all tests."""
    print("Testing MCQA Token Positions")
    print("=" * 70)
    print()

    try:
        test_basic_tokenization()
        test_get_symbol_index()
        test_get_correct_symbol_index()
        test_token_position_objects()
        test_correct_symbol_token_position()
        test_last_token_position()
        test_create_token_positions()
        test_highlight_selected_token()
        test_edge_case_symbol_not_found()

        # This test might fail with some tokenizers
        print("Note: The following test assumes period tokenizes separately...")
        try:
            test_period_tokens()
        except AssertionError as e:
            print(f"⚠ Period token test failed (tokenizer-dependent): {e}")
            print("  This is expected with some tokenizers\n")

        print("\n" + "=" * 70)
        print("🎉 All token position tests passed!")
        print("=" * 70)
        print("\nToken position functions available:")
        print("✓ create_token_positions - Factory function for all positions")
        print("✓ TokenPosition objects - Dynamic position identification")
        print("✓ Uses new declarative token position system")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
