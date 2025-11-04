"""
Test Script: MCQA Counterfactual Generation

This script tests the counterfactual generation functions for the MCQA task.
It verifies that counterfactuals are generated correctly and can distinguish
between causal variables.
"""

from tasks.MCQA.causal_models import positional_causal_model, NUM_CHOICES, COLORS, ALPHABET
from tasks.MCQA.counterfactuals import (
    sample_answerable_question,
    same_symbol_different_position,
    different_symbol,
    random_counterfactual
)


def test_sample_answerable_question():
    """Test that sample_answerable_question generates valid questions."""
    print("=== Test 1: Sample Answerable Question ===")

    model = positional_causal_model

    print("Generating 10 answerable questions:\n")

    for i in range(10):
        input_sample = sample_answerable_question()

        # Run forward to get answer
        output = model.run_forward(input_sample)

        print(f"Sample {i+1}:")
        print(f"  Object-Color: {input_sample['object_color']}")
        print(f"  Choices: {input_sample['choice0']}, {input_sample['choice1']}")
        print(f"  Symbols: {input_sample['symbol0']}, {input_sample['symbol1']}")
        print(f"  Answer position: {output['answer_position']}")
        print(f"  Answer: {output['answer']}")

        # Verify it's answerable (color is in choices)
        color = input_sample['object_color'][1]
        choices = [input_sample[f'choice{i}'] for i in range(NUM_CHOICES)]
        assert color in choices, f"Color {color} should be in choices {choices}"

        # Verify answer is not None
        assert output['answer_position'] is not None, "Should have valid answer position"
        assert output['answer'] is not None, "Should have valid answer"

        # Verify symbols are unique
        symbols = [input_sample[f'symbol{i}'] for i in range(NUM_CHOICES)]
        assert len(symbols) == len(set(symbols)), "Symbols should be unique"

        # Verify choices are unique
        assert len(choices) == len(set(choices)), "Choices should be unique"

    print("\n✓ All questions are answerable and well-formed")
    print("✓ Test 1 passed\n")


def test_same_symbol_different_position():
    """Test same_symbol_different_position counterfactual generation."""
    print("=== Test 2: Same Symbol Different Position ===")

    model = positional_causal_model

    print("Generating 5 counterfactual pairs:\n")

    for i in range(5):
        example = same_symbol_different_position()
        input_sample = example["input"]
        counterfactual = example["counterfactual_inputs"][0]

        # Run forward on both
        input_output = model.run_forward(input_sample)
        counter_output = model.run_forward(counterfactual)

        print(f"Pair {i+1}:")
        print(f"  Input symbols: {input_sample['symbol0']}, {input_sample['symbol1']}")
        print(f"  Counter symbols: {counterfactual['symbol0']}, {counterfactual['symbol1']}")
        print(f"  Input answer position: {input_output['answer_position']} -> {input_output['answer']}")
        print(f"  Counter answer position: {counter_output['answer_position']} -> {counter_output['answer']}")

        # Verify symbols SET is the same (they're swapped in position)
        input_symbols = {input_sample[f'symbol{j}'] for j in range(NUM_CHOICES)}
        counter_symbols = {counterfactual[f'symbol{j}'] for j in range(NUM_CHOICES)}
        assert input_symbols == counter_symbols, \
            "Same symbols should be used (but swapped)"

        # Verify answer positions are different
        assert input_output['answer_position'] != counter_output['answer_position'], \
            "Answer positions should differ"

        # Verify symbols and choices were swapped together
        pos = input_output['answer_position']
        new_pos = counter_output['answer_position']
        assert input_sample[f'choice{pos}'] == counterfactual[f'choice{new_pos}'], \
            "Choices should be swapped"
        assert input_sample[f'symbol{pos}'] == counterfactual[f'symbol{new_pos}'], \
            "Symbols should be swapped"

        print(f"  ✓ Symbols and choices swapped together, positions differ")

    print("\n✓ All counterfactuals generated correctly")
    print("✓ Test 2 passed\n")


def test_different_symbol():
    """Test different_symbol counterfactual generation."""
    print("=== Test 3: Different Symbol ===")

    model = positional_causal_model

    print("Generating 5 counterfactual pairs:\n")

    for i in range(5):
        example = different_symbol()
        input_sample = example["input"]
        counterfactual = example["counterfactual_inputs"][0]

        # Run forward on both
        input_output = model.run_forward(input_sample)
        counter_output = model.run_forward(counterfactual)

        print(f"Pair {i+1}:")
        print(f"  Input symbols: {input_sample['symbol0']}, {input_sample['symbol1']}")
        print(f"  Counter symbols: {counterfactual['symbol0']}, {counterfactual['symbol1']}")
        print(f"  Input answer: {input_output['answer']}")
        print(f"  Counter answer: {counter_output['answer']}")

        # Verify all symbols are different
        input_symbols = {input_sample[f'symbol{j}'] for j in range(NUM_CHOICES)}
        counter_symbols = {counterfactual[f'symbol{j}'] for j in range(NUM_CHOICES)}
        assert input_symbols.isdisjoint(counter_symbols), \
            "All symbols should be different"

        # Verify choices are the same
        for j in range(NUM_CHOICES):
            assert input_sample[f'choice{j}'] == counterfactual[f'choice{j}'], \
                f"Choice {j} should be same in both"

        # Verify answer position is the same
        assert input_output['answer_position'] == counter_output['answer_position'], \
            "Answer position should be same"

        # Verify answer symbols are different
        assert input_output['answer'] != counter_output['answer'], \
            "Answer symbols should differ"

        print(f"  ✓ Symbols different, choices and position same")

    print("\n✓ All counterfactuals generated correctly")
    print("✓ Test 3 passed\n")


def test_random_counterfactual():
    """Test random_counterfactual generation."""
    print("=== Test 4: Random Counterfactual ===")

    model = positional_causal_model

    print("Generating 5 random counterfactual pairs:\n")

    for i in range(5):
        example = random_counterfactual()
        input_sample = example["input"]
        counterfactual = example["counterfactual_inputs"][0]

        # Run forward on both
        input_output = model.run_forward(input_sample)
        counter_output = model.run_forward(counterfactual)

        print(f"Pair {i+1}:")
        print(f"  Input: {input_sample['object_color'][0]} is {input_sample['object_color'][1]}")
        print(f"  Counter: {counterfactual['object_color'][0]} is {counterfactual['object_color'][1]}")
        print(f"  Input answer: {input_output['answer']}")
        print(f"  Counter answer: {counter_output['answer']}")

        # Verify both are valid
        assert 'raw_input' in input_sample
        assert 'raw_output' in input_output
        assert 'raw_input' in counterfactual
        assert 'raw_output' in counter_output

        # They should likely be different (but not guaranteed)
        print(f"  Same prompt? {input_sample['raw_input'] == counterfactual['raw_input']}")

    print("\n✓ All random counterfactuals generated successfully")
    print("✓ Test 4 passed\n")


def test_counterfactual_structure():
    """Test that counterfactuals have the correct structure."""
    print("=== Test 5: Counterfactual Structure ===")

    # Test all three types
    examples = [
        ("same_symbol_different_position", same_symbol_different_position()),
        ("different_symbol", different_symbol()),
        ("random_counterfactual", random_counterfactual())
    ]

    for name, example in examples:
        print(f"Testing {name}:")

        # Check structure
        assert "input" in example, f"{name} should have 'input' key"
        assert "counterfactual_inputs" in example, f"{name} should have 'counterfactual_inputs' key"
        assert len(example["counterfactual_inputs"]) == 1, f"{name} should have 1 counterfactual"

        input_sample = example["input"]
        counterfactual = example["counterfactual_inputs"][0]

        # Check both have raw_input
        assert "raw_input" in input_sample, "Input should have raw_input"
        assert "raw_input" in counterfactual, "Counterfactual should have raw_input"

        print(f"  ✓ Structure correct")

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
    result = model.can_distinguish_with_dataset(diff_symbol_examples, ["answer"], ["answer_position"])
    print(f"  Can distinguish 'answer' from 'answer_position': {result['proportion']:.2f}")
    print(f"  Count: {result['count']}/{len(diff_symbol_examples)}")
    assert result['proportion'] > 0.5, "Should distinguish well"

    # Test different_symbol distinguishes "answer" from no intervention
    result2 = model.can_distinguish_with_dataset(diff_symbol_examples, ["answer"], None)
    print(f"  Can distinguish 'answer' from no intervention: {result2['proportion']:.2f}")
    print(f"  Count: {result2['count']}/{len(diff_symbol_examples)}")
    assert result2['proportion'] > 0.5, "Should distinguish well"

    # Test same_symbol_different_position confounds with no intervention
    print("\nTesting same_symbol_different_position dataset:")
    result3 = model.can_distinguish_with_dataset(same_symbol_examples, ["answer_position"], None)
    print(f"  Can distinguish 'answer_position' from no intervention: {result3['proportion']:.2f}")
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
    print(f"Average: {elapsed/100*1000:.2f} ms per sample")

    assert elapsed < 10, "Should complete in reasonable time"

    print("✓ Sampling performance acceptable")
    print("✓ Test 7 passed\n")


def test_counterfactual_validity():
    """Test that all counterfactuals produce valid prompts."""
    print("=== Test 8: Counterfactual Validity ===")

    model = positional_causal_model

    # Test each type
    types = [
        ("same_symbol_different_position", same_symbol_different_position),
        ("different_symbol", different_symbol),
        ("random_counterfactual", random_counterfactual)
    ]

    for name, generator in types:
        print(f"Testing {name}:")

        for i in range(5):
            example = generator()
            input_sample = example["input"]
            counterfactual = example["counterfactual_inputs"][0]

            # Run forward
            input_output = model.run_forward(input_sample)
            counter_output = model.run_forward(counterfactual)

            # Check prompts are well-formed
            assert len(input_output['raw_input']) > 0, "Input prompt should not be empty"
            assert len(counter_output['raw_input']) > 0, "Counter prompt should not be empty"

            # Check they contain expected elements
            assert "What color" in input_output['raw_input'], "Should contain question"
            assert "Answer:" in input_output['raw_input'], "Should contain Answer:"

            assert "What color" in counter_output['raw_input'], "Should contain question"
            assert "Answer:" in counter_output['raw_input'], "Should contain Answer:"

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

        print("\n" + "="*70)
        print("🎉 All counterfactual tests passed!")
        print("="*70)
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
