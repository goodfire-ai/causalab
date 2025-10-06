# tests/test_pyvene_core/conftest.py

import pytest
import torch
import random
import numpy as np
from collections import defaultdict

from causal.causal_model import CausalModel
from causal.counterfactual_dataset import CounterfactualDataset
from neural.pipeline import LMPipeline
from neural.LM_units import TokenPosition, ResidualStream
from neural.model_units import AtomicModelUnit


@pytest.fixture(scope="session")
def seed_everything():
    """Set random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session")
def mcqa_causal_model():
    """
    Create a simple MCQA causal model fixture.
    
    This model represents a simplified version of the multiple-choice question answering task
    where questions have 4 choices and 1 correct answer.
    """
    # Define model variables
    NUM_CHOICES = 4
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Define object/color pairs for the questions
    COLOR_OBJECTS = [
        ("red", "apple"), ("yellow", "banana"), ("green", "leaf"),
        ("blue", "sky"), ("brown", "chocolate"), ("white", "snow"),
        ("black", "coal"), ("purple", "grape"), ("orange", "carrot"),
        ("pink", "flamingo"), ("gray", "elephant"), ("gold", "coin")
    ]
    
    COLORS = [item[0] for item in COLOR_OBJECTS]
    
    # Define the causal model variables and values
    variables = ["question"] + [f"symbol{x}" for x in range(NUM_CHOICES)] + \
                [f"choice{x}" for x in range(NUM_CHOICES)] + ["answer_pointer", "answer"]
    
    values = {f"choice{x}": COLORS for x in range(NUM_CHOICES)}
    values.update({f"symbol{x}": list(ALPHABET) for x in range(NUM_CHOICES)})
    values.update({"answer_pointer": list(range(NUM_CHOICES)), "answer": list(ALPHABET)})
    values.update({"question": COLOR_OBJECTS})
    
    # Define parent relationships
    parents = {"answer": ["answer_pointer"] + [f"symbol{x}" for x in range(NUM_CHOICES)], 
               "answer_pointer": ["question"] + [f"choice{x}" for x in range(NUM_CHOICES)],
               "question": []}
    parents.update({f"choice{x}": [] for x in range(NUM_CHOICES)})
    parents.update({f"symbol{x}": [] for x in range(NUM_CHOICES)})
    
    # Define causal mechanisms
    def get_question():
        return random.choice(COLOR_OBJECTS)
    
    def get_symbol():
        return random.choice(list(ALPHABET))
    
    def get_choice():
        return random.choice(COLORS)
    
    def get_answer_pointer(question, *choices):
        for i, choice in enumerate(choices):
            if choice == question[0]:  # question[0] is the color
                return i
        # If no match, return random (shouldn't happen in well-formed questions)
        return random.randint(0, NUM_CHOICES - 1)
    
    def get_answer(answer_pointer, *symbols):
        return " " + symbols[answer_pointer]
    
    mechanisms = {
        "question": get_question,
        **{f"symbol{i}": get_symbol for i in range(NUM_CHOICES)},
        **{f"choice{i}": get_choice for i in range(NUM_CHOICES)},
        "answer_pointer": get_answer_pointer,
        "answer": get_answer
    }
    
    # Define input and output format functions
    def input_dumper(input_data):
        output = f"Question: The {input_data['question'][1]} is {input_data['question'][0]}. What color is the {input_data['question'][1]}?"
        for i in range(NUM_CHOICES):
            output += f"\n{input_data[f'symbol{i}']}. {input_data[f'choice{i}']}"
        output += f"\nAnswer:"
        return output
    
    def output_dumper(setting):
        return setting["answer"]
    
    # Helper to parse inputs
    def input_parser(text):
        # Simple parsing of format: "Question: The object is color. What color is the object?"
        try:
            question_part = text.split("Question: ")[1].split(". What color")[0]
            object_color = question_part.split(" is ")
            object_name = object_color[0].replace("The ", "").strip()
            color = object_color[1].strip()
            
            # Extract choices
            lines = text.strip().split('\n')
            choices = {}
            for i, line in enumerate(lines[1:]):
                if line.startswith("Answer:"):
                    break
                if "." in line:
                    symbol, choice_text = line.split(". ", 1)
                    choices[f"symbol{i}"] = symbol
                    choices[f"choice{i}"] = choice_text
            
            return {
                'question': (color, object_name),
                **choices
            }
        except:
            # Return empty dictionary if parsing fails
            return {}
    
    # Create and return the model
    return CausalModel(
        variables, values, parents, mechanisms, 
        input_dumper=input_dumper, 
        output_dumper=output_dumper, 
        input_loader=input_parser,
        id="4_answer_MCQA_test"
    )


@pytest.fixture(scope="session")
def mcqa_counterfactual_datasets(mcqa_causal_model, seed_everything):
    """
    Generate test counterfactual datasets for the MCQA task.
    
    Returns a dictionary with 3 types of counterfactual datasets:
    1. random_letter - Swapping the letter symbols while keeping choices same
    2. random_position - Moving the correct answer to a different position
    3. combined - Both letter and position changes
    
    Each type has small train and test sets.
    """
    model = mcqa_causal_model
    NUM_CHOICES = 4
    
    # Helper to check if inputs are well-formed
    def is_input_valid(x):
        # Check that the question color appears in the choices
        question_color = x["question"][0]
        choice_colors = [x[f"choice{i}"] for i in range(NUM_CHOICES)]
        symbols = [x[f"symbol{i}"] for i in range(NUM_CHOICES)]
        
        # The color must be in the choices and symbols must be unique
        return question_color in choice_colors and len(symbols) == len(set(symbols))
    
    # Counterfactual generator functions
    def random_letter_counterfactual():
        """Generate counterfactual with new random letters."""
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy
        
        # Get current symbols to avoid duplicates
        used_symbols = [input_setting[f"symbol{i}"] for i in range(NUM_CHOICES)]
        
        # Generate new set of symbols
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        available_symbols = [s for s in alphabet if s not in used_symbols]
        new_symbols = random.sample(available_symbols, NUM_CHOICES)
        
        # Update symbols in counterfactual
        for i in range(NUM_CHOICES):
            counterfactual[f"symbol{i}"] = new_symbols[i]
        
        return {
            "input": input_setting,
            "counterfactual_inputs": [counterfactual]
        }
    
    def random_position_counterfactual():
        """Generate counterfactual with answer moved to new position."""
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy
        
        # Get current answer position
        answer_position = model.run_forward(input_setting)["answer_pointer"]
        
        # Choose a different position
        available_positions = [i for i in range(NUM_CHOICES) if i != answer_position]
        new_position = random.choice(available_positions)
        
        # Swap choices to move correct answer
        correct_color = counterfactual[f"choice{answer_position}"]
        counterfactual[f"choice{answer_position}"] = counterfactual[f"choice{new_position}"]
        counterfactual[f"choice{new_position}"] = correct_color
        
        return {
            "input": input_setting,
            "counterfactual_inputs": [counterfactual]
        }
    
    def combined_counterfactual():
        """Generate counterfactual with both letter and position changes."""
        letter_cf = random_letter_counterfactual()
        
        # Start with the letter-changed counterfactual
        input_setting = letter_cf["input"]
        counterfactual = letter_cf["counterfactual_inputs"][0]
        
        # Now change position too
        answer_position = model.run_forward(input_setting)["answer_pointer"]
        available_positions = [i for i in range(NUM_CHOICES) if i != answer_position]
        new_position = random.choice(available_positions)
        
        # Swap choices
        correct_color = counterfactual[f"choice{answer_position}"]
        counterfactual[f"choice{answer_position}"] = counterfactual[f"choice{new_position}"]
        counterfactual[f"choice{new_position}"] = correct_color
        
        return {
            "input": input_setting,
            "counterfactual_inputs": [counterfactual]
        }
    
    # Generate the datasets
    datasets = {}
    
    # Small size for tests
    train_size = 10
    test_size = 5
    
    # Generate datasets for each counterfactual type
    for name, generator in [
        ("random_letter", random_letter_counterfactual),
        ("random_position", random_position_counterfactual),
        ("combined", combined_counterfactual)
    ]:
        # Train dataset
        train_data = {"input": [], "counterfactual_inputs": []}
        for _ in range(train_size):
            sample = generator()
            train_data["input"].append(sample["input"])
            train_data["counterfactual_inputs"].append(sample["counterfactual_inputs"])
        
        # Test dataset
        test_data = {"input": [], "counterfactual_inputs": []}
        for _ in range(test_size):
            sample = generator()
            test_data["input"].append(sample["input"])
            test_data["counterfactual_inputs"].append(sample["counterfactual_inputs"])
        
        # Create CounterfactualDataset objects
        datasets[f"{name}_train"] = CounterfactualDataset.from_dict(train_data, id=f"{name}_train")
        datasets[f"{name}_test"] = CounterfactualDataset.from_dict(test_data, id=f"{name}_test")
    
    return datasets


@pytest.fixture(scope="function")
def mock_tiny_lm():
    """
    Create a minimal mock implementation of a language model for testing.
    
    This returns a mock LMPipeline with simple generation capabilities that's
    suitable for testing without requiring full model weights.
    """
    class MockTokenizer:
        def __init__(self):
            self.pad_token = "<pad>"
            self.pad_token_id = 0
            self.eos_token = "</s>"
            self.eos_token_id = 1
            self.padding_side = "right"
        
        def __call__(self, texts, padding=None, max_length=None, truncation=None, 
                    return_tensors=None, add_special_tokens=None):
            # Very simple tokenization - just use character codes
            batch = []
            for text in texts:
                # Tokenize by character ordinals for simplicity
                tokens = [ord(c) % 100 + 2 for c in text]  # +2 to avoid pad/eos IDs
                batch.append(tokens)
            
            # Apply padding if needed
            if padding or max_length:
                max_len = max_length if max_length else max(len(seq) for seq in batch)
                batch = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in batch]
            
            # Convert to tensors
            input_ids = torch.tensor(batch, dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        def decode(self, token_ids, skip_special_tokens=False):
            # Simple decoding - convert back to characters
            if skip_special_tokens:
                token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.eos_token_id]]
            return "".join(chr((t - 2) + 97) if t >= 2 else "_" for t in token_ids)
        
        def batch_decode(self, sequences, skip_special_tokens=False):
            return [self.decode(seq, skip_special_tokens) for seq in sequences]
    
    class MockConfig:
        def __init__(self):
            self.name_or_path = "mock_model"
            self.num_hidden_layers = 4
            self.hidden_size = 32
            self.n_head = 4
    
    class MockModel:
        def __init__(self):
            self.config = MockConfig()
            self.device = "cpu"
            self.dtype = torch.float32
        
        def to(self, device=None, dtype=None):
            if device:
                self.device = device
            if dtype:
                self.dtype = dtype
            return self
        
        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            # Mock forward pass - return random logits and final hidden states
            batch_size, seq_len = input_ids.shape
            hidden_size = self.config.hidden_size
            
            # Create mock hidden states and logits
            hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=self.device, dtype=self.dtype)
            vocab_size = 100  # Small vocab for testing
            logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device, dtype=self.dtype)
            
            return type('obj', (object,), {
                'hidden_states': hidden_states,
                'logits': logits,
                'last_hidden_state': hidden_states
            })
        
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, 
                    return_dict_in_generate=False, output_scores=False, **kwargs):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            # Generate a simple continuation - just increment token IDs
            new_tokens = torch.randint(2, 99, (batch_size, max_new_tokens), device=self.device)
            sequences = torch.cat([input_ids, new_tokens], dim=1)
            
            if return_dict_in_generate:
                if output_scores:
                    # Mock scores - random probabilities
                    vocab_size = 100
                    scores = [torch.randn(batch_size, vocab_size, device=self.device) for _ in range(max_new_tokens)]
                    return type('GenerationOutput', (object,), {'sequences': sequences, 'scores': scores})
                return type('GenerationOutput', (object,), {'sequences': sequences, 'scores': None})
            
            return sequences
    
    # Create the mock pipeline
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()
    
    # Create a LMPipeline with the mock components
    pipeline = LMPipeline(model_or_name="mock_model", max_new_tokens=3)
    
    # Replace with our mocks
    pipeline.model = mock_model
    pipeline.tokenizer = mock_tokenizer
    
    return pipeline


@pytest.fixture(scope="function")
def token_positions(mock_tiny_lm, mcqa_causal_model):
    """Create token position identifiers for the MCQA task."""
    # Define a function to get the last token
    def get_last_token(prompt):
        token_ids = mock_tiny_lm.load(prompt)["input_ids"][0]
        return [len(token_ids) - 1]
    
    # Define a function to get the position of the answer symbol
    def get_answer_symbol_position(prompt):
        # Parse the prompt to get the causal input
        causal_input = mcqa_causal_model.input_loader(prompt)
        
        # Get the answer pointer
        output = mcqa_causal_model.run_forward(causal_input)
        answer_position = output["answer_pointer"]
        
        # Get the symbol at that position
        answer_symbol = causal_input[f"symbol{answer_position}"]
        
        # Find the token position of this symbol
        tokens = mock_tiny_lm.load(prompt)["input_ids"][0]
        text = prompt.split("\n")
        
        for i, line in enumerate(text[1:]):  # Skip the question line
            if line.startswith(answer_symbol):
                # Found the line with the answer
                # Count tokens up to this point
                substring = "\n".join(text[:i+1]) + "\n" + answer_symbol
                position_tokens = mock_tiny_lm.load(substring)["input_ids"][0]
                return [len(position_tokens) - 1]  # Return the last token position (the symbol)
        
        # Fallback to last token if the symbol isn't found
        return get_last_token(prompt)
    
    # Create TokenPosition objects
    return [
        TokenPosition(get_last_token, mock_tiny_lm, id="last_token"),
        TokenPosition(get_answer_symbol_position, mock_tiny_lm, id="answer_symbol")
    ]


@pytest.fixture(scope="function")
def model_units_list(mock_tiny_lm, token_positions):
    """Create model units list for testing."""
    # Create a list of ResidualStream units for different layers and positions
    units = []
    
    # Use a subset of layers for testing
    layers = [0, 2]  
    
    for layer in layers:
        for token_position in token_positions:
            # Create a ResidualStream unit
            unit = ResidualStream(
                layer=layer,
                token_indices=token_position,
                shape=(mock_tiny_lm.model.config.hidden_size,),
                target_output=True  # Use block output
            )
            units.append([unit])  # Each unit is wrapped in its own list
    
    return units


@pytest.fixture(scope="function")
def mock_intervenable_model(mock_tiny_lm, model_units_list):
    """Mock intervenable model for testing."""
    class MockIntervenableModel:
        def __init__(self):
            self.model = mock_tiny_lm.model
            self.interventions = {}
            
            # Create mock interventions
            for i, units in enumerate(model_units_list):
                key = f"intervention_{i}"
                self.interventions[key] = type('MockIntervention', (object,), {
                    'forward': lambda x, y, **kwargs: torch.randn_like(x),
                    'parameters': lambda: [torch.nn.Parameter(torch.randn(10))],
                    'state_dict': lambda: {"weight": torch.randn(10)},
                    'load_state_dict': lambda x: None,
                    'get_sparsity_loss': lambda: torch.tensor(0.1),
                    'set_temperature': lambda x: None,
                    'mask': torch.nn.Parameter(torch.randn(10))
                })
        
        def disable_model_gradients(self):
            pass
            
        def eval(self):
            pass
            
        def set_device(self, device, set_model=True):
            pass
            
        def set_zero_grad(self):
            pass
            
        def count_parameters(self):
            return 100
            
        def __call__(self, inputs, unit_locations=None, **kwargs):
            # Mock forward pass
            batch_size = inputs["input_ids"].shape[0]
            hidden_size = self.model.config.hidden_size
            
            # Create mock activations
            activations = [torch.randn(1, hidden_size) for _ in range(10)]
            
            # Create mock outputs
            outputs = type('obj', (object,), {
                'hidden_states': torch.randn(batch_size, 10, hidden_size),
                'logits': torch.randn(batch_size, 10, 100)
            })
            
            return [(outputs, activations), None]
            
        def generate(self, inputs, sources=None, unit_locations=None, subspaces=None, **kwargs):
            # Mock generation with interventions
            batch_size = inputs["input_ids"].shape[0]
            seq_len = 5  # Fixed output sequence length
            
            if kwargs.get("output_scores", False):
                # Return mock scores
                scores = [torch.randn(batch_size, 100) for _ in range(seq_len)]
                return [type('GenerationOutput', (object,), {'sequences': None, 'scores': scores})]
            
            # Return mock sequences
            sequences = torch.randint(2, 99, (batch_size, seq_len))
            return [type('GenerationOutput', (object,), {'sequences': sequences, 'scores': None})]
    
    return MockIntervenableModel()