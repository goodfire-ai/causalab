"""
Test Script: MCQA Causal Models

This script tests the MCQA causal model structure, mechanisms, and forward execution.
"""

from tasks.MCQA.causal_models import (
    positional_causal_model,
    fill_template,
    get_answer_position,
    get_answer,
    OBJECT_COLORS,
    COLORS,
    ALPHABET,
    NUM_CHOICES,
    TEMPLATES
)


def test_causal_model_structure():
    """Test that the causal model has the correct structure."""
    print("=== Test 1: Causal Model Structure ===")

    model = positional_causal_model

    # Check variables
    expected_vars = [
        "template", "object_color", "raw_input",
        "symbol0", "symbol1", "choice0", "choice1",
        "answer_position", "answer", "raw_output"
    ]

    for var in expected_vars:
        assert var in model.variables, f"Variable '{var}' should be in model"
        print(f"✓ Variable '{var}' present")

    # Check parent relationships
    assert model.parents["raw_input"] == ["template", "object_color", "symbol0", "symbol1", "choice0", "choice1"]
    assert model.parents["answer_position"] == ["object_color", "choice0", "choice1"]
    assert model.parents["answer"] == ["answer_position", "symbol0", "symbol1"]

    print("✓ Parent relationships correct")
    print("✓ Test 1 passed\n")


def test_sample_input():
    """Test sampling valid inputs."""
    print("=== Test 2: Sample Input ===")

    model = positional_causal_model

    # Sample 5 inputs
    for i in range(5):
        input_sample = model.sample_input()

        # Check all input variables are present
        assert 'template' in input_sample
        assert 'object_color' in input_sample
        assert 'symbol0' in input_sample
        assert 'symbol1' in input_sample
        assert 'choice0' in input_sample
        assert 'choice1' in input_sample

        print(f"Sample {i+1}: {input_sample['object_color']} with choices {input_sample['choice0']}, {input_sample['choice1']}")

    print("✓ Test 2 passed\n")


def test_fill_template_mechanism():
    """Test the template filling mechanism."""
    print("=== Test 3: Fill Template Mechanism ===")

    template = TEMPLATES[0]
    object_color = ("banana", "yellow")
    symbol0 = "A"
    symbol1 = "B"
    choice0 = "blue"
    choice1 = "yellow"

    result = fill_template(template, object_color, symbol0, symbol1, choice0, choice1)

    print(f"Template: {template}")
    print(f"\nFilled result:\n{result}")

    # Check substitutions
    assert "banana" in result, "Object name should be in result"
    assert "yellow" in result, "Color should be in result"
    assert "A" in result, "Symbol0 should be in result"
    assert "B" in result, "Symbol1 should be in result"
    assert "blue" in result, "Choice0 should be in result"

    # Check no placeholders remain
    assert "<object>" not in result, "No placeholder should remain"
    assert "<color>" not in result, "No placeholder should remain"
    assert "<symbol0>" not in result, "No placeholder should remain"
    assert "<symbol1>" not in result, "No placeholder should remain"
    assert "<choice0>" not in result, "No placeholder should remain"
    assert "<choice1>" not in result, "No placeholder should remain"

    print("✓ All substitutions correct")
    print("✓ Test 3 passed\n")


def test_answer_position_mechanism():
    """Test the answer position mechanism."""
    print("=== Test 4: Answer Position Mechanism ===")

    # Test case 1: Answer in position 0
    object_color = ("banana", "yellow")
    choice0 = "yellow"
    choice1 = "blue"

    pos = get_answer_position(object_color, choice0, choice1)
    print(f"Test 1: Color 'yellow' in choices ['yellow', 'blue'] -> position {pos}")
    assert pos == 0, "Should find yellow at position 0"

    # Test case 2: Answer in position 1
    object_color = ("grass", "green")
    choice0 = "blue"
    choice1 = "green"

    pos = get_answer_position(object_color, choice0, choice1)
    print(f"Test 2: Color 'green' in choices ['blue', 'green'] -> position {pos}")
    assert pos == 1, "Should find green at position 1"

    # Test case 3: Answer not in choices (edge case)
    object_color = ("banana", "yellow")
    choice0 = "blue"
    choice1 = "red"

    pos = get_answer_position(object_color, choice0, choice1)
    print(f"Test 3: Color 'yellow' in choices ['blue', 'red'] -> position {pos}")
    assert pos is None, "Should return None when color not in choices"

    # Test case 4: Duplicate colors (edge case - returns first match)
    object_color = ("snow", "white")
    choice0 = "white"
    choice1 = "white"

    pos = get_answer_position(object_color, choice0, choice1)
    print(f"Test 4: Color 'white' in choices ['white', 'white'] -> position {pos}")
    assert pos == 0, "Should return first match when duplicate"

    print("✓ Test 4 passed\n")


def test_answer_retrieval_mechanism():
    """Test the answer retrieval mechanism."""
    print("=== Test 5: Answer Retrieval Mechanism ===")

    symbol0 = "A"
    symbol1 = "B"

    # Test position 0
    answer = get_answer(0, symbol0, symbol1)
    print(f"Position 0 with symbols ['A', 'B'] -> answer '{answer}'")
    assert answer == "A", "Position 0 should return first symbol"

    # Test position 1
    answer = get_answer(1, symbol0, symbol1)
    print(f"Position 1 with symbols ['A', 'B'] -> answer '{answer}'")
    assert answer == "B", "Position 1 should return second symbol"

    # Test None position (edge case)
    answer = get_answer(None, symbol0, symbol1)
    print(f"Position None with symbols ['A', 'B'] -> answer '{answer}'")
    assert answer is None, "None position should return None"

    print("✓ Test 5 passed\n")


def test_forward_execution():
    """Test running the causal model forward."""
    print("=== Test 6: Forward Execution ===")

    model = positional_causal_model

    # Create specific input
    input_sample = {
        'template': TEMPLATES[0],
        'object_color': ('banana', 'yellow'),
        'symbol0': 'A',
        'symbol1': 'B',
        'choice0': 'blue',
        'choice1': 'yellow',
    }

    output = model.run_forward(input_sample)

    print(f"Prompt:\n{output['raw_input']}\n")
    print(f"Answer position: {output['answer_position']}")
    print(f"Answer: {output['answer']}")
    print(f"Raw output: {output['raw_output']}")

    # Verify output
    assert output['answer_position'] == 1, "Yellow should be at position 1"
    assert output['answer'] == 'B', "Answer should be B"
    assert output['raw_output'] == 'B', "Raw output should match answer"
    assert 'banana' in output['raw_input']
    assert 'yellow' in output['raw_input']

    print("✓ Forward execution correct")
    print("✓ Test 6 passed\n")


def test_multiple_forward_executions():
    """Test multiple forward executions for consistency."""
    print("=== Test 7: Multiple Forward Executions ===")

    model = positional_causal_model

    print("Testing 10 random samples:\n")

    for i in range(10):
        input_sample = model.sample_input()
        output = model.run_forward(input_sample)

        # Verify consistency
        assert 'raw_input' in output
        assert 'raw_output' in output
        assert 'answer_position' in output
        assert 'answer' in output

        # Verify answer matches answer_position (if position is valid)
        if output['answer_position'] is not None:
            expected_symbol = input_sample[f"symbol{output['answer_position']}"]
            assert output['answer'] == expected_symbol, f"Answer should match symbol at answer_position"
        else:
            assert output['answer'] is None, "Answer should be None when position is None"

        print(f"Sample {i+1}: {input_sample['object_color'][0]} is {input_sample['object_color'][1]}")
        print(f"  Choices: {input_sample['choice0']}, {input_sample['choice1']}")
        print(f"  Answer at position {output['answer_position']}: {output['answer']}")

    print("\n✓ All executions consistent")
    print("✓ Test 7 passed\n")


def test_edge_case_duplicate_colors():
    """Test edge case where object color appears in multiple choice positions."""
    print("=== Test 8: Edge Case - Duplicate Colors ===")

    model = positional_causal_model

    input_sample = {
        'template': TEMPLATES[0],
        'object_color': ('snow', 'white'),
        'symbol0': 'X',
        'symbol1': 'Y',
        'choice0': 'white',
        'choice1': 'white',
    }

    output = model.run_forward(input_sample)

    print(f"Object: snow is white")
    print(f"Choices: white, white")
    print(f"Answer position: {output['answer_position']}")
    print(f"Answer: {output['answer']}")

    # Should return first match
    assert output['answer_position'] == 0, "Should return first match"
    assert output['answer'] == 'X', "Should return symbol at position 0"

    print("✓ Handles duplicate colors correctly")
    print("✓ Test 8 passed\n")


def test_constants():
    """Test that constants are correctly defined."""
    print("=== Test 9: Constants ===")

    print(f"Number of object-color pairs: {len(OBJECT_COLORS)}")
    print(f"Number of unique colors: {len(COLORS)}")
    print(f"Number of choices: {NUM_CHOICES}")
    print(f"Alphabet length: {len(ALPHABET)}")
    print(f"Number of templates: {len(TEMPLATES)}")

    # Verify we have enough colors for choices
    assert len(COLORS) >= NUM_CHOICES, "Should have at least NUM_CHOICES colors"

    # Verify we have enough symbols
    assert len(ALPHABET) >= NUM_CHOICES, "Should have at least NUM_CHOICES symbols"

    # Verify color extraction worked
    assert 'yellow' in COLORS
    assert 'green' in COLORS
    assert 'blue' in COLORS

    print("✓ Constants properly defined")
    print("✓ Test 9 passed\n")


def main():
    """Run all tests."""
    print("Testing MCQA Causal Models")
    print("=" * 70)
    print()

    try:
        test_causal_model_structure()
        test_sample_input()
        test_fill_template_mechanism()
        test_answer_position_mechanism()
        test_answer_retrieval_mechanism()
        test_forward_execution()
        test_multiple_forward_executions()
        test_edge_case_duplicate_colors()
        test_constants()

        print("\n" + "="*70)
        print("🎉 All causal model tests passed!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
