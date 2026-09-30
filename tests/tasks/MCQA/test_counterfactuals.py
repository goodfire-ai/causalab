"""
Test Script: MCQA Counterfactual Generation

This script tests the counterfactual generation functions for the MCQA task.
It verifies that counterfactuals are generated correctly and can distinguish
between causal variables.
"""

import pytest

from causalab.causal.model_comparison import can_distinguish_with_dataset
from causalab.tasks.MCQA.causal_models import NUM_CHOICES, positional_causal_model
from causalab.tasks.MCQA.counterfactuals import (
    different_symbol,
    random_counterfactual,
    same_symbol_different_position,
    sample_answerable_question,
)

pytestmark = pytest.mark.unit


def test_sample_answerable_question():
    """Test that sample_answerable_question generates valid questions."""
    print("=== Test 1: Sample Answerable Question ===")

    print("Generating 10 answerable questions:\n")

    for i in range(10):
        # sample_answerable_question() returns a CausalTrace with all computed values
        trace = sample_answerable_question()

        print(f"Sample {i + 1}:")
        print(f"  Object-Color: {trace['object']}, {trace['color']}")
        print(f"  Choices: {trace['choices[0]']}, {trace['choices[1]']}")
        print(f"  Symbols: {trace['symbols[0]']}, {trace['symbols[1]']}")
        print(f"  Answer position: {trace['answer_position']}")
        print(f"  Answer: {trace['answer']}")

        # Verify it's answerable (color is in choices)
        color = trace["color"]
        choices = [trace[f"choices[{j}]"] for j in range(NUM_CHOICES)]
        assert color in choices, f"Color {color} should be in choices {choices}"

        # Verify answer is not None
        assert trace["answer_position"] is not None, "Should have valid answer position"
        assert trace["answer"] is not None, "Should have valid answer"

        # Verify symbols are unique
        symbols = [trace[f"symbols[{j}]"] for j in range(NUM_CHOICES)]
        assert len(symbols) == len(set(symbols)), "Symbols should be unique"

        # Verify choices are unique
        assert len(choices) == len(set(choices)), "Choices should be unique"

    print("\n✓ All questions are answerable and well-formed")
    print("✓ Test 1 passed\n")


def test_same_symbol_different_position():
    """Test same_symbol_different_position counterfactual generation."""
    print("=== Test 2: Same Symbol Different Position ===")

    print("Generating 5 counterfactual pairs:\n")

    for i in range(5):
        example = same_symbol_different_position()
        # example["input"] and example["counterfactual_inputs"][0] are already CausalTraces
        input_trace = example["input"]
        counter_trace = example["counterfactual_inputs"][0]

        print(f"Pair {i + 1}:")
        print(
            f"  Input symbols: {input_trace['symbols[0]']}, {input_trace['symbols[1]']}"
        )
        print(
            f"  Counter symbols: {counter_trace['symbols[0]']}, {counter_trace['symbols[1]']}"
        )
        print(
            f"  Input answer position: {input_trace['answer_position']} -> {input_trace['answer']}"
        )
        print(
            f"  Counter answer position: {counter_trace['answer_position']} -> {counter_trace['answer']}"
        )

        # Verify symbols SET is the same (they're swapped in position)
        input_symbols = {input_trace[f"symbols[{j}]"] for j in range(NUM_CHOICES)}
        counter_symbols = {counter_trace[f"symbols[{j}]"] for j in range(NUM_CHOICES)}
        assert input_symbols == counter_symbols, (
            "Same symbols should be used (but swapped)"
        )

        # Verify answer positions are different
        assert input_trace["answer_position"] != counter_trace["answer_position"], (
            "Answer positions should differ"
        )

        # Verify symbols and choices were swapped together
        pos = input_trace["answer_position"]
        new_pos = counter_trace["answer_position"]
        assert input_trace[f"choices[{pos}]"] == counter_trace[f"choices[{new_pos}]"], (
            "Choices should be swapped"
        )
        assert input_trace[f"symbols[{pos}]"] == counter_trace[f"symbols[{new_pos}]"], (
            "Symbols should be swapped"
        )

        print("  ✓ Symbols and choices swapped together, positions differ")

    print("\n✓ All counterfactuals generated correctly")
    print("✓ Test 2 passed\n")


def test_different_symbol():
    """Test different_symbol counterfactual generation."""
    print("=== Test 3: Different Symbol ===")

    print("Generating 5 counterfactual pairs:\n")

    for i in range(5):
        example = different_symbol()
        # example["input"] and example["counterfactual_inputs"][0] are already CausalTraces
        input_trace = example["input"]
        counter_trace = example["counterfactual_inputs"][0]

        print(f"Pair {i + 1}:")
        print(
            f"  Input symbols: {input_trace['symbols[0]']}, {input_trace['symbols[1]']}"
        )
        print(
            f"  Counter symbols: {counter_trace['symbols[0]']}, {counter_trace['symbols[1]']}"
        )
        print(f"  Input answer: {input_trace['answer']}")
        print(f"  Counter answer: {counter_trace['answer']}")

        # Verify all symbols are different
        input_symbols = {input_trace[f"symbols[{j}]"] for j in range(NUM_CHOICES)}
        counter_symbols = {counter_trace[f"symbols[{j}]"] for j in range(NUM_CHOICES)}
        assert input_symbols.isdisjoint(counter_symbols), (
            "All symbols should be different"
        )

        # Verify choices are the same
        for j in range(NUM_CHOICES):
            assert input_trace[f"choices[{j}]"] == counter_trace[f"choices[{j}]"], (
                f"Choice {j} should be same in both"
            )

        # Verify answer position is the same
        assert input_trace["answer_position"] == counter_trace["answer_position"], (
            "Answer position should be same"
        )

        # Verify answer symbols are different
        assert input_trace["answer"] != counter_trace["answer"], (
            "Answer symbols should differ"
        )

        print("  ✓ Symbols different, choices and position same")

    print("\n✓ All counterfactuals generated correctly")
    print("✓ Test 3 passed\n")


def test_random_counterfactual():
    """Test random_counterfactual generation."""
    print("=== Test 4: Random Counterfactual ===")

    print("Generating 5 random counterfactual pairs:\n")

    for i in range(5):
        example = random_counterfactual()
        # example["input"] and example["counterfactual_inputs"][0] are already CausalTraces
        input_trace = example["input"]
        counter_trace = example["counterfactual_inputs"][0]

        print(f"Pair {i + 1}:")
        print(f"  Input: {input_trace['object']} is {input_trace['color']}")
        print(f"  Counter: {counter_trace['object']} is {counter_trace['color']}")
        print(f"  Input answer: {input_trace['answer']}")
        print(f"  Counter answer: {counter_trace['answer']}")

        # Verify both are valid (have computed values)
        assert "raw_input" in input_trace
        assert "raw_output" in input_trace
        assert "raw_input" in counter_trace
        assert "raw_output" in counter_trace

        # They should likely be different (but not guaranteed)
        print(
            f"  Same prompt? {input_trace['raw_input'] == counter_trace['raw_input']}"
        )

    print("\n✓ All random counterfactuals generated successfully")
    print("✓ Test 4 passed\n")


def test_counterfactual_structure():
    """Test that counterfactuals have the correct structure."""
    print("=== Test 5: Counterfactual Structure ===")

    # Test all three types
    examples = [
        ("same_symbol_different_position", same_symbol_different_position()),
        ("different_symbol", different_symbol()),
        ("random_counterfactual", random_counterfactual()),
    ]

    for name, example in examples:
        print(f"Testing {name}:")

        # Check structure
        assert "input" in example, f"{name} should have 'input' key"
        assert "counterfactual_inputs" in example, (
            f"{name} should have 'counterfactual_inputs' key"
        )
        assert len(example["counterfactual_inputs"]) == 1, (
            f"{name} should have 1 counterfactual"
        )

        input_sample = example["input"]
        counterfactual = example["counterfactual_inputs"][0]

        # Check both have raw_input
        assert "raw_input" in input_sample, "Input should have raw_input"
        assert "raw_input" in counterfactual, "Counterfactual should have raw_input"

        print("  ✓ Structure correct")

    print("\n✓ Test 5 passed\n")


def test_distinguishability():
    """Test that counterfactuals can distinguish between variables."""
    print("=== Test 6: Counterfactual Distinguishability ===")

    model = positional_causal_model

    # Generate small datasets
    print("Generating 10 examples of each counterfactual type...\n")

    diff_symbol_examples = [different_symbol() for _ in range(10)]
    same_symbol_examples = [same_symbol_different_position() for _ in range(10)]

    # Test different_symbol distinguishes "answer" from "answer_position"
    print("Testing different_symbol dataset:")
    result = can_distinguish_with_dataset(
        diff_symbol_examples,
        model,
        ["answer"],
        causal_model2=model,
        target_variables2=["answer_position"],
    )
    print(
        f"  Can distinguish 'answer' from 'answer_position': {result['proportion']:.2f}"
    )
    print(f"  Count: {result['count']}/{len(diff_symbol_examples)}")
    assert result["proportion"] > 0.5, "Should distinguish well"

    # Test different_symbol distinguishes "answer" from no intervention
    result2 = can_distinguish_with_dataset(
        diff_symbol_examples, model, ["answer"], target_variables2=None
    )
    print(
        f"  Can distinguish 'answer' from no intervention: {result2['proportion']:.2f}"
    )
    print(f"  Count: {result2['count']}/{len(diff_symbol_examples)}")
    assert result2["proportion"] > 0.5, "Should distinguish well"

    # Test same_symbol_different_position confounds with no intervention
    print("\nTesting same_symbol_different_position dataset:")
    result3 = can_distinguish_with_dataset(
        same_symbol_examples, model, ["answer_position"], target_variables2=None
    )
    print(
        f"  Can distinguish 'answer_position' from no intervention: {result3['proportion']:.2f}"
    )
    print(f"  Count: {result3['count']}/{len(same_symbol_examples)}")
    # This should be low because symbols are the same

    print("\n✓ Distinguishability tests completed")
    print("✓ Test 6 passed\n")


def test_edge_case_sampling():
    """Test that sampling doesn't get stuck in infinite loops."""
    print("=== Test 7: Edge Case - Sampling Performance ===")

    import time

    # Time how long it takes to generate 100 samples
    start = time.time()
    for _ in range(100):
        sample_answerable_question()
    elapsed = time.time() - start

    print(f"Generated 100 answerable questions in {elapsed:.2f} seconds")
    print(f"Average: {elapsed / 100 * 1000:.2f} ms per sample")

    assert elapsed < 10, "Should complete in reasonable time"

    print("✓ Sampling performance acceptable")
    print("✓ Test 7 passed\n")


def test_counterfactual_validity():
    """Test that all counterfactuals produce valid prompts."""
    print("=== Test 8: Counterfactual Validity ===")

    # Test each type
    types = [
        ("same_symbol_different_position", same_symbol_different_position),
        ("different_symbol", different_symbol),
        ("random_counterfactual", random_counterfactual),
    ]

    for name, generator in types:
        print(f"Testing {name}:")

        for _ in range(5):
            example = generator()
            # example["input"] and example["counterfactual_inputs"][0] are already CausalTraces
            input_trace = example["input"]
            counter_trace = example["counterfactual_inputs"][0]

            # Check prompts are well-formed
            assert len(input_trace["raw_input"]) > 0, "Input prompt should not be empty"
            assert len(counter_trace["raw_input"]) > 0, (
                "Counter prompt should not be empty"
            )

            # Check they contain expected elements
            assert "What color" in input_trace["raw_input"], "Should contain question"
            assert "Answer:" in input_trace["raw_input"], "Should contain Answer:"

            assert "What color" in counter_trace["raw_input"], "Should contain question"
            assert "Answer:" in counter_trace["raw_input"], "Should contain Answer:"

        print(f"  ✓ All {name} examples valid")

    print("\n✓ All counterfactuals produce valid prompts")
    print("✓ Test 8 passed\n")


def main():
    """Run all tests."""
    print("Testing MCQA Counterfactual Generation")
    print("=" * 70)
    print()

    try:
        test_sample_answerable_question()
        test_same_symbol_different_position()
        test_different_symbol()
        test_random_counterfactual()
        test_counterfactual_structure()
        test_distinguishability()
        test_edge_case_sampling()
        test_counterfactual_validity()

        print("\n" + "=" * 70)
        print("🎉 All counterfactual tests passed!")
        print("=" * 70)
        print("\nCounterfactual types available:")
        print("✓ sample_answerable_question - Valid questions with answer in choices")
        print("✓ same_symbol_different_position - Swap positions, keep symbols")
        print("✓ different_symbol - Change symbols, keep position")
        print("✓ random_counterfactual - Independent random samples")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
