import os
import sys
import random
import torch
import re
import pyvene as pv
import json
import pandas as pd
from typing import Dict, List, Tuple

def setup_cuda():
    print("Built with CUDA:", torch.version.cuda)
    available = torch.cuda.is_available()
    print("cuda.is_available():", available)
    if available:
        print("Device name:", torch.cuda.get_device_name(0))
    return available

def add_project_root(root_path="/data/jason_lim/CausalAbstraction"):
    if root_path not in sys.path:
        sys.path.append(root_path)
    os.path.join(root_path)
    os.getcwd()
    return root_path

add_project_root()

# Internal module imports (same as MCQA)
from causal.causal_model import CausalModel, CounterfactualDataset
from experiments.scone_filter_experiment import SconeFilterExperiment # SCONE-specific filter experiment
from lm_units.fixed_pipeline import LMPipeline
from lm_units.LM_units import TokenPosition, get_last_token_index
from experiments.residual_stream_experiment import PatchResidualStream

# SCONE-specific data loading
def load_scone_datasets(data_dir="data", half=True):
    """Load and combine SCONE datasets."""
    if half:
        dataset_names = ["no_negation", "one_scoped", "one_not_scoped"]
    else:
        dataset_names = ["no_negation", "one_scoped", "one_not_scoped",
                        "one_scoped_one_not_scoped", "two_scoped", "two_not_scoped"]
    
    combined_datasets = {}
    for name in dataset_names:
        train_df = pd.read_csv(f"{data_dir}/train/{name}.csv")
        test_df = pd.read_csv(f"{data_dir}/test/{name}.csv")
        combined_datasets[name] = pd.concat([train_df, test_df], ignore_index=True)
    
    return combined_datasets

def load_entailment_data(file_path="data/lookup/entailment_lookup_filtered.jsonl"):
    """Load entailment lookup data."""
    entailment_data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                entailment_data.append(entry)
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line}")
    return entailment_data


ENTAILMENT_DATA = load_entailment_data()

# Input/output handling (SCONE-specific)
def input_dumper(example):
    """Convert example to prompt format."""
    if isinstance(example, str):
        return example
    
    premise = example.get("PremiseSentence", "")
    hypothesis = example.get("HypothesisSentence", "")
    
    prompt = (
        f"Sentence 1: {premise}\n"
        f"Sentence 2: {hypothesis}\n"
        f"Does Sentence 1 entail Sentence 2? Please respond only with either 'entailment' or 'neutral'.\n"
        "Answer: "
    )
    
    return prompt

def output_dumper(setting):
    """Extract final relation from setting."""
    return setting['FinalRelation']

def checker(neural_output, causal_output, debug=False):
    """Enhanced checker that handles both old and new formats."""
    
    # Handle the case where causal_output is a string (old format)
    if isinstance(causal_output, str):
        expected_label = causal_output
    # Handle case where it's a dict with raw_output
    elif isinstance(causal_output, dict) and "raw_output" in causal_output:
        expected_label = causal_output["raw_output"]
    else:
        expected_label = str(causal_output)
    
    # Simple containment check
    return expected_label.lower().strip() in str(neural_output).lower().strip()

# def checker(neural_output, causal_output, debug=False):
#     """Enhanced checker with debugging."""
#     import os
#     import json
#     from datetime import datetime
    
#     if debug:
#         # Create debug entry
#         debug_entry = {
#             "timestamp": datetime.now().isoformat(),
#             "neural_output": repr(neural_output),
#             "neural_output_type": str(type(neural_output)),
#             "causal_output": repr(causal_output),
#             "causal_output_type": str(type(causal_output))
#         }
        
#         # Extract expected label
#         if isinstance(causal_output, dict):
#             if "raw_output" in causal_output:
#                 expected_label = causal_output["raw_output"]
#                 debug_entry["expected_label_from_setting"] = expected_label
#             else:
#                 debug_entry["setting_keys"] = list(causal_output.keys())
#                 expected_label = str(causal_output)
#         else:
#             expected_label = str(causal_output)
            
#         debug_entry["expected_label"] = expected_label
        
#         # Compute match
#         result = expected_label.lower().strip() in str(neural_output).lower().strip()
#         debug_entry["match_result"] = result
        
#         # Save to debug file (append mode)
#         os.makedirs("debug", exist_ok=True)
#         debug_file = "debug/checker_calls.jsonl"
        
#         with open(debug_file, 'a') as f:
#             f.write(json.dumps(debug_entry) + '\n')
        
#         return result
    
#     # Original checker logic for non-debug
#     return neural_output in causal_output


# def checker(llm_output: str, expected_label: str, debug=True) -> bool:
#     """Check if LLM output matches expected label."""
#     if isinstance(llm_output, list):
#         if llm_output:
#             llm_output = llm_output[0]
#         else:
#             if debug:
#                 print("[CHECKER] Received empty list as output")
#             return False
    
#     normalized_output = llm_output.strip().lower()
    
#     if isinstance(expected_label, str):
#         normalized_label = expected_label.strip().lower()
#     else:
#         normalized_label = output_dumper(expected_label).strip().lower()
    
#     if debug:
#         print("="*50)
#         print(f"[CHECKER] Raw LLM output: '{llm_output}'")
#         print(f"[CHECKER] Normalized LLM output: '{normalized_output}'")
#         print(f"[CHECKER] Expected label: '{normalized_label}'")
#         print(f"[CHECKER] Contains check: {normalized_label in normalized_output}")
#         print("="*50)
    
#     return normalized_label in normalized_output


# "google/gemma-2-9b-it"
# Pipeline and experiment setup (reuse from MCQA pattern)
def init_pipeline(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = LMPipeline(model_name, max_new_tokens=10, device=device, dtype=torch.float16)  # Temporarily increase
    pipe.tokenizer.padding_side = "right"
    print("Loaded on", pipe.model.device)
    return pipe

def demo_examples(counterfactual_datasets, scone_model, pipeline):
    """Demo examples from the datasets."""
    sample = next(iter(counterfactual_datasets.values()))[0]
    print("INPUT:", sample["input"])
    # Use raw_input key for demo
    print("RAW INPUT:", sample["input"]["raw_input"])
    print("EXPECTED OUTPUT:", sample["input"]["raw_output"])  # Use raw_output
    output = pipeline.generate(sample["input"]["raw_input"])  # Generate from raw_input
    print("MODEL PREDICTION:", pipeline.dump(output))

def test_raw_generation(pipeline, prompt):
    """Test what the model actually generates for a given prompt."""
    print(f"\n=== TESTING RAW GENERATION ===")
    print(f"Prompt: {prompt}")
    
    try:
        # Generate with more tokens to see full output
        temp_output = pipeline.generate([prompt])
        dumped = pipeline.dump(temp_output)
        
        print(f"Raw generation: {repr(dumped[0])}")
        print(f"Full text: '{dumped[0]}'")
        print(f"Length: {len(dumped[0])}")
        
        # Also try to see what tokens were generated
        tokenized = pipeline.load([prompt])
        print(f"Input tokens: {len(tokenized['input_ids'][0])}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Updated run_filter_experiment function
def run_filter_experiment(pipeline, scone_model, datasets):
    """Filter datasets based on model performance using custom SCONE filter."""
    print("\nFiltering datasets based on model performance...")
    exp = SconeFilterExperiment(pipeline, scone_model, checker, debug=True)  # Enable debug
    return exp.filter(datasets, verbose=True, batch_size=1)

# def run_filter_experiment(pipeline, scone_model, datasets):
#     """Filter datasets based on model performance using custom SCONE filter."""
#     print("\nFiltering datasets based on model performance...")
#     exp = SconeFilterExperiment(pipeline, scone_model, checker)  # Use custom class
#     return exp.filter(datasets, verbose=True, batch_size=1)

# ------------------------------------------# SCONE-specific imports
# Token position functions (SCONE-specific)
def get_predicate_premise(prompt, pipeline, debug=False):
    """Return token indices for the premise predicate."""
    try:
        tokens = pipeline.tokenizer.tokenize(prompt)
        
        # Look for "Sentence 1:" and find predicate after it
        for i in range(len(tokens) - 1):
            if 'sentence' in tokens[i].lower() and '1' in tokens[i+1]:
                # Search in reasonable range after sentence 1 marker
                search_start = i + 2
                search_end = min(search_start + 20, len(tokens))
                
                # Look for content words (longer tokens)
                for j in range(search_start, search_end):
                    clean_token = tokens[j].lower().replace('▁', '')
                    if len(clean_token) > 3 and clean_token not in ["that", "with", "from", "this"]:
                        return [j]
                
                return [search_start + 3]  # Fallback position
        
        return [5]  # Default fallback
    except Exception as e:
        if debug:
            print(f"Error in get_predicate_premise: {e}")
        return [5]

def get_predicate_hypothesis(prompt, pipeline, debug=False):
    """Return token indices for the hypothesis predicate."""
    try:
        tokens = pipeline.tokenizer.tokenize(prompt)
        
        # Look for "Sentence 2:" and find predicate after it
        for i in range(len(tokens) - 1):
            if 'sentence' in tokens[i].lower() and '2' in tokens[i+1]:
                search_start = i + 2
                search_end = min(search_start + 20, len(tokens))
                
                # Look for content words
                for j in range(search_start, search_end):
                    clean_token = tokens[j].lower().replace('▁', '')
                    if len(clean_token) > 3 and clean_token not in ["that", "with", "from", "this"]:
                        return [j]
                
                return [search_start + 3]
        
        return [15]  # Default fallback
    except Exception as e:
        if debug:
            print(f"Error in get_predicate_hypothesis: {e}")
        return [15]

def get_negation_index(prompt, pipeline, debug=False):
    """Return token indices for negation terms."""
    try:
        tokens = pipeline.tokenizer.tokenize(prompt)
        
        # Look for negation terms
        negation_terms = ["not", "n't", "no", "never", "none"]
        for i, token in enumerate(tokens):
            clean_token = token.lower().replace('▁', '')
            for term in negation_terms:
                if term in clean_token:
                    return [i]
        
        return [3]  # Fallback if no negation found
    except Exception as e:
        if debug:
            print(f"Error in get_negation_index: {e}")
        return [3]


# # SCONE-specific causal model

class SconeCausalModel(CausalModel):
    def __init__(self, datasets_dict, entailment_data, variables, values, parents, mechanisms):
        super().__init__(variables, values, parents, mechanisms, id="SCONE_entailment")
        self.datasets_dict = datasets_dict
        self.entailment_data = entailment_data
        self.current_sentence_pair = None
        self.current_gold_label = None
        
        # Create entailment lookup from entailment_data
        self.entailment_lookup = {}
        for entry in entailment_data:
            if isinstance(entry, dict) and entry.get('label') == 1:
                self.entailment_lookup[(entry['hyponym'], entry['hypernym'])] = "entailment"
    
    def run_forward(self, intervention=None, debug=False):
        # Reset cached data at start of forward pass
        self.current_sentence_pair = None
        self.current_gold_label = None
        result = super().run_forward(intervention=intervention)
        return result
    
    
    def label_counterfactual_data(self, dataset, target_variables):
        """
        Proper intervention: Take original input, change only target variables,
        then recompute through causal model.
        """
        import os
        import json
        from datetime import datetime
        
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "target_variables": target_variables,
            "dataset_size": len(dataset),
            "examples": [],
            "label_summary": {}
        }
        
        inputs = []
        counterfactual_inputs_list = []
        labels = []
        settings = []
        
        for i, example in enumerate(dataset):
            input_data = example["input"] 
            counterfactual_inputs = example["counterfactual_inputs"]
            
            if isinstance(counterfactual_inputs, list):
                counterfactual_input = counterfactual_inputs[0]
            else:
                counterfactual_input = counterfactual_inputs
            
            # PROPER INTERVENTION: Start with original, change only target variables
            intervention_input = {}

            # Always keep the exogenous variables (predicates)
            intervention_input["Predicate1"] = input_data["Predicate1"] 
            intervention_input["Predicate2"] = input_data["Predicate2"]
            intervention_input["Scope"] = input_data["Scope"]

            # Add the target variable intervention
            for var in target_variables:
                if var in counterfactual_input:
                    intervention_input[var] = counterfactual_input[var]
                                
            # Recompute through causal model with intervention
            intervention_result = self.run_forward(intervention=intervention_input)
            expected_output = intervention_result["raw_output"]
            
            print(f"Intervention complete. Expected output: {expected_output}")
            print(f"[DEBUG] Example {i}: Original input: {input_data}")
            print(f"[DEBUG] Example {i}: Counterfactual input: {counterfactual_input}")
            print(f"[DEBUG] Example {i}: Intervention input: {intervention_input}")
            
            
            # Debug info for first 5 examples
            if i < 5:
                example_debug = {
                    "example_id": i,
                    "original_input": {k: input_data.get(k, "N/A") for k in ["Predicate1", "Predicate2", "Scope", "BaseEntailment", "FinalRelation", "raw_output"]},
                    "counterfactual_input": {k: counterfactual_input.get(k, "N/A") for k in ["Predicate1", "Predicate2", "Scope", "BaseEntailment", "FinalRelation", "raw_output"]},
                    "intervention_input": {k: intervention_input.get(k, "N/A") for k in ["Predicate1", "Predicate2", "Scope", "BaseEntailment", "FinalRelation"]},
                    "expected_output_old": counterfactual_input["raw_output"],  # Old wrong way
                    "expected_output_new": expected_output,  # New correct way
                    "target_variables": target_variables
                }
                debug_info["examples"].append(example_debug)
            
            # Create setting with intervention values
            setting = {}
            for var in target_variables:
                if var in intervention_input:
                    setting[var] = intervention_input[var]
            setting["raw_output"] = expected_output
            
            inputs.append(input_data)
            counterfactual_inputs_list.append(example["counterfactual_inputs"])
            labels.append(expected_output)
            settings.append(setting)
        
        # Save debug info
        debug_info["label_summary"] = {
            "total_examples": len(labels),
            "label_distribution": {label: labels.count(label) for label in set(labels)},
            "first_10_labels": labels[:10],
            "unique_labels": list(set(labels))
        }
        
        os.makedirs("debug", exist_ok=True)
        target_vars_str = "_".join(target_variables)
        dataset_id = getattr(dataset, 'id', 'unknown')
        
        with open(f"debug/label_counterfactual_data_{dataset_id}_{target_vars_str}_fixed.json", 'w') as f:
            json.dump(debug_info, f, indent=2)
        
        data_dict = {
            "input": inputs,
            "counterfactual_inputs": counterfactual_inputs_list,
            "label": labels,
            "setting": settings
        }
        
        return CounterfactualDataset.from_dict(data_dict, id=f"{dataset_id}_{target_vars_str}_labeled")
    
    def get_sentence_pair_and_label(self, pred1: str, pred2: str, scope: str) -> Tuple[str, str, str]:
        """Get sentence pair and gold label from datasets based on predicates and scope."""
        if self.current_sentence_pair is None:
            if scope not in self.datasets_dict:
                return None, None, None
                
            df = self.datasets_dict[scope]
            
            # Determine column names based on dataset
            if scope == "one_scoped":
                # one_scoped.csv doesn't have edited columns
                sent1_col = 'sentence1'
                sent2_col = 'sentence2'
                label_col = 'gold_label'
            else:
                # Other datasets have edited versions
                sent1_col = 'sentence1_edited' if 'sentence1_edited' in df.columns else 'sentence1'
                sent2_col = 'sentence2_edited' if 'sentence2_edited' in df.columns else 'sentence2'
                label_col = 'gold_label_edited' if 'gold_label_edited' in df.columns else 'gold_label'
            
            # Find matching row
            matches = df[(df['sentence1_lex'] == pred1) & (df['sentence2_lex'] == pred2)]
            
            if matches.empty:
                return None, None, None
                
            chosen_row = matches.sample(n=1).iloc[0]
            self.current_sentence_pair = (chosen_row[sent1_col], chosen_row[sent2_col])
            self.current_gold_label = chosen_row[label_col]
        
        return self.current_sentence_pair[0], self.current_sentence_pair[1], self.current_gold_label

    def get_sentence_pair(self, pred1: str, pred2: str, scope: str) -> Tuple[str, str]:
        """Get sentence pair (for backward compatibility)."""
        sent1, sent2, _ = self.get_sentence_pair_and_label(pred1, pred2, scope)
        return (sent1, sent2) if sent1 and sent2 else (None, None)
    
    def get_gold_label(self, pred1: str, pred2: str, scope: str) -> str:
        """Get gold label from dataset."""
        _, _, label = self.get_sentence_pair_and_label(pred1, pred2, scope)
        return label if label else "neutral"

# ------------------------------------------
# SCONE-specific counterfactual generation


def create_base_entailment_counterfactual(datasets_dict, scone_model, debug=False):
    """
    Create base entailment counterfactual by finding pairs with opposite BaseEntailment values.
    Adapted from toggle_base_entailment_counterfactuals.
    """
    allowed_scopes = [s for s in ['no_negation', 'one_scoped', 'one_not_scoped'] 
                     if s in datasets_dict]
    if not allowed_scopes:
        raise ValueError("None of the allowed scopes are available in datasets_dict.")
    
    # Sample original input
    orig_scope = random.choice(allowed_scopes)
    df_orig = datasets_dict[orig_scope]
    
    # Sample a random row
    idx = random.randint(0, len(df_orig) - 1)
    row_orig = df_orig.iloc[idx]
    
    # Extract predicate pair
    pred1 = row_orig.get("sentence1_lex", None)
    pred2 = row_orig.get("sentence2_lex", None)
    
    # Build original input
    original_input = {
        "Scope": orig_scope,
        "Predicate1": pred1,
        "Predicate2": pred2
    }
    
    # Run through causal model to get BaseEntailment
    original_result = scone_model.run_forward(intervention=original_input)
    original_base_entailment = original_result["BaseEntailment"]
    
    # Determine target BaseEntailment (opposite of original)
    target_base_entailment = "neutral" if original_base_entailment == "entailment" else "entailment"
    
    if debug:
        print(f"[DEBUG] Original BaseEntailment: {original_base_entailment}")
        print(f"[DEBUG] Target BaseEntailment: {target_base_entailment}")
    
    # Search for a counterfactual with opposite BaseEntailment
    max_attempts = 50
    counterfactual_found = False
    
    for _ in range(max_attempts):
        # Sample random input from any allowed scope
        random_scope = random.choice(allowed_scopes)
        df_random = datasets_dict[random_scope]
        random_idx = random.randint(0, len(df_random) - 1)
        row_random = df_random.iloc[random_idx]
        
        # Get predicates
        random_pred1 = row_random.get("sentence1_lex", None)
        random_pred2 = row_random.get("sentence2_lex", None)
        
        # Build candidate counterfactual
        candidate_input = {
            "Scope": random_scope,
            "Predicate1": random_pred1,
            "Predicate2": random_pred2
        }
        
        # Check BaseEntailment using causal model
        candidate_result = scone_model.run_forward(intervention=candidate_input)
        candidate_base_entailment = candidate_result["BaseEntailment"]
        
        if candidate_base_entailment == target_base_entailment:
            # Found suitable counterfactual
            counterfactual_input = candidate_result
            counterfactual_found = True
            break
    
    if not counterfactual_found:
        # Fallback: find any different entry
        if debug:
            print(f"[DEBUG] Failed to find natural counterfactual, using any different entry")
        
        retry = 0
        while retry < 10:
            cf_scope = random.choice(allowed_scopes)
            df_cf = datasets_dict[cf_scope]
            cf_idx = random.randint(0, len(df_cf) - 1)
            
            # Skip if same as original
            if cf_scope == orig_scope and cf_idx == idx:
                retry += 1
                continue
                
            row_cf = df_cf.iloc[cf_idx]
            cf_pred1 = row_cf.get("sentence1_lex", None)
            cf_pred2 = row_cf.get("sentence2_lex", None)
            
            # Build fallback counterfactual
            fallback_input = {
                "Scope": cf_scope,
                "Predicate1": cf_pred1,
                "Predicate2": cf_pred2
            }
            
            counterfactual_input = scone_model.run_forward(intervention=fallback_input)
            break
    
    if debug:
        print(f"[DEBUG] Original: {original_result}")
        print(f"[DEBUG] Counterfactual: {counterfactual_input}")
    
    return {"input": original_result, "counterfactual_inputs": [counterfactual_input]}


def create_random_counterfactual_pair(datasets_dict, scone_model, debug=False):
    """
    Create completely independent random counterfactual pairs.
    Adapted from random_counterfactual_pair.
    """
    allowed_scopes = [s for s in ['no_negation', 'one_scoped', 'one_not_scoped']
                     if s in datasets_dict]
    if not allowed_scopes:
        raise ValueError("None of the allowed scopes are available in datasets_dict.")
    
    # Sample scope for original and counterfactual
    orig_scope = random.choice(allowed_scopes)
    cf_scope = random.choice(allowed_scopes)
    
    df_orig = datasets_dict[orig_scope]
    df_cf = datasets_dict[cf_scope]
    
    # Sample random rows
    orig_idx = random.randint(0, len(df_orig) - 1)
    row_orig = df_orig.iloc[orig_idx]
    
    cf_idx = random.randint(0, len(df_cf) - 1)
    row_cf = df_cf.iloc[cf_idx]
    
    # Ensure they're different entries
    attempts = 0
    while (orig_idx == cf_idx and orig_scope == cf_scope) and attempts < 5:
        cf_idx = random.randint(0, len(df_cf) - 1)
        row_cf = df_cf.iloc[cf_idx]
        attempts += 1
    
    # Extract predicate pairs
    pred1 = row_orig.get("sentence1_lex", None)
    pred2 = row_orig.get("sentence2_lex", None)
    
    cf_pred1 = row_cf.get("sentence1_lex", None)
    cf_pred2 = row_cf.get("sentence2_lex", None)
    
    # Build original input
    original_input = {
        "Scope": orig_scope,
        "Predicate1": pred1,
        "Predicate2": pred2
    }
    
    # Build counterfactual input
    counterfactual_input = {
        "Scope": cf_scope,
        "Predicate1": cf_pred1,
        "Predicate2": cf_pred2
    }
    
    # Run both through causal model
    original_result = scone_model.run_forward(intervention=original_input)
    counterfactual_result = scone_model.run_forward(intervention=counterfactual_input)
    
    if debug:
        print(f"[DEBUG] Original: {original_result}")
        print(f"[DEBUG] Counterfactual: {counterfactual_result}")
    
    return {"input": original_result, "counterfactual_inputs": [counterfactual_result]}


def create_predicate_tracing_counterfactual(datasets_dict, scone_model, debug=False):
    """
    Create predicate tracing counterfactual by swapping predicates and sentences.
    Adapted from predicate_tracing_counterfactuals.
    """
    allowed_scopes = [s for s in ['no_negation', 'one_scoped', 'one_not_scoped'] 
                     if s in datasets_dict]
    if not allowed_scopes:
        raise ValueError("None of the allowed scopes are available in datasets_dict.")
    
    # Choose original scope
    orig_scope = random.choice(allowed_scopes)
    df_orig = datasets_dict[orig_scope]
    
    # Sample a random row
    idx = random.randint(0, len(df_orig) - 1)
    row_orig = df_orig.iloc[idx]
    
    # Extract predicate pair
    pred1 = row_orig.get("sentence1_lex", None)
    pred2 = row_orig.get("sentence2_lex", None)
    
    # Build original input
    original_input = {
        "Scope": orig_scope,
        "Predicate1": pred1,
        "Predicate2": pred2
    }
    
    # Create counterfactual by swapping predicates
    # This will effectively swap the premise and hypothesis sentences too
    counterfactual_input = {
        "Scope": orig_scope,  # Keep same scope
        "Predicate1": pred2,  # Swap predicates
        "Predicate2": pred1   # Swap predicates
    }
    
    # Run both through causal model
    original_result = scone_model.run_forward(intervention=original_input)
    counterfactual_result = scone_model.run_forward(intervention=counterfactual_input)
    
    if debug:
        print(f"[DEBUG] Original: {original_result}")
        print(f"[DEBUG] Counterfactual: {counterfactual_result}")
    
    return {"input": original_result, "counterfactual_inputs": [counterfactual_result]}

def create_scope_counterfactual(datasets_dict, scone_model, debug=False):
    """
    Create scope counterfactual by toggling between even/odd scopes with randomized predicates.
    Based on toggle_scope_counterfactual_pair function.
    """
    allowed_scopes = [s for s in ['no_negation', 'one_scoped', 'one_not_scoped'] 
                     if s in datasets_dict]
    if not allowed_scopes:
        raise ValueError("None of the allowed scopes are available in datasets_dict.")
    
    # Choose an original scope at random
    orig_scope = random.choice(allowed_scopes)
    df_orig = datasets_dict[orig_scope]
    
    # Sample a random row from the original dataset
    orig_idx = random.randint(0, len(df_orig) - 1)
    row_orig = df_orig.iloc[orig_idx]
    
    # Extract original predicates
    orig_pred1 = row_orig.get("sentence1_lex", None)
    orig_pred2 = row_orig.get("sentence2_lex", None)
    
    # Build the original input dictionary
    original_input = {
        "Scope": orig_scope,
        "Predicate1": orig_pred1,
        "Predicate2": orig_pred2
    }
    
    # Define allowed scope groups for toggling
    even_scopes = [s for s in ['no_negation', 'one_not_scoped'] if s in allowed_scopes]
    odd_scopes = [s for s in ['one_scoped'] if s in allowed_scopes]
    
    # Toggle scope based on parity (even/odd)
    if orig_scope in even_scopes and odd_scopes:
        # If original is even and odd scopes are available, choose an odd scope
        new_scope = odd_scopes[0]
        if debug:
            print(f"[DEBUG] Even scope '{orig_scope}' -> odd scope: '{new_scope}'")
    elif orig_scope in odd_scopes and even_scopes:
        # If original is odd and even scopes are available, choose an even scope
        new_scope = random.choice(even_scopes)
        if debug:
            print(f"[DEBUG] Odd scope '{orig_scope}' -> even scope: '{new_scope}'")
    else:
        # Fallback if we can't toggle parity: choose any different scope
        available_scopes = [s for s in allowed_scopes if s != orig_scope]
        if available_scopes:
            new_scope = random.choice(available_scopes)
            if debug:
                print(f"[DEBUG] Cannot toggle parity, using different scope: '{new_scope}'")
        else:
            # Last resort: use the same scope but will ensure different predicates
            new_scope = orig_scope
            if debug:
                print(f"[DEBUG] No alternative scopes available, using same scope: '{new_scope}'")
    
    # Get the target dataset
    df_new = datasets_dict[new_scope]
    
    # Randomly sample a different row from the new dataset
    # This ensures we get different predicates (randomized)
    new_idx = random.randint(0, len(df_new) - 1)
    
    # Make sure it's a different row if we're in the same dataset
    if new_scope == orig_scope:
        attempts = 0
        while new_idx == orig_idx and len(df_new) > 1 and attempts < 10:
            new_idx = random.randint(0, len(df_new) - 1)
            attempts += 1
    
    row_new = df_new.iloc[new_idx]
    
    # Extract new predicates (which will be different/randomized)
    new_pred1 = row_new.get("sentence1_lex", None)
    new_pred2 = row_new.get("sentence2_lex", None)
    
    # Build the counterfactual input dictionary with randomized predicates
    counterfactual_input = {
        "Scope": new_scope,        # CONTROLLED: Different scope
        "Predicate1": new_pred1,   # RANDOMIZED: Different predicates
        "Predicate2": new_pred2    # RANDOMIZED: Different predicates
    }
    
    # Run both through causal model
    original_result = scone_model.run_forward(intervention=original_input)
    counterfactual_result = scone_model.run_forward(intervention=counterfactual_input)
    
    if debug:
        print(f"[DEBUG] Original: {original_result}")
        print(f"[DEBUG] Counterfactual: {counterfactual_result}")
    
    return {"input": original_result, "counterfactual_inputs": [counterfactual_result]}

def build_counterfactual_datasets(datasets_dict, scone_model, size=100): 
    """Build counterfactual datasets for SCONE with all four types."""
    counterfactual_dataset = {
        # "scope_counterfactuals": CounterfactualDataset.from_sampler(
        #     size, lambda: create_scope_counterfactual(datasets_dict, scone_model)
        # )#,
        "base_entailment_counterfactuals": CounterfactualDataset.from_sampler(
            size, lambda: create_base_entailment_counterfactual(datasets_dict, scone_model)
        )#,
        # "random_counterfactuals": CounterfactualDataset.from_sampler(
        #     size, lambda: create_random_counterfactual_pair(datasets_dict, scone_model)
        # )#,
        # "predicate_tracing_counterfactuals": CounterfactualDataset.from_sampler(
        #     size, lambda: create_predicate_tracing_counterfactual(datasets_dict, scone_model)
        # )
        }
    
    return counterfactual_dataset

def save_filtered_datasets(filtered_datasets):
    # Save filtered datasets to data/counterfactual_filtered folder
    save_dir = "data/counterfactuals_filtered"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name, dataset in filtered_datasets.items():
        dataset_path = os.path.join(save_dir, f"{dataset_name}.jsonl")
        
        with open(dataset_path, 'w') as f:
            for example in dataset:
                # Convert to JSON and write each example on a new line
                json.dump(example, f)
                f.write('\n')
        
        print(f"Saved {len(dataset)} examples to {dataset_path}")

# Mechanism functions
def create_raw_input(premise_sentence, hypothesis_sentence):
    """Create the raw input prompt from premise and hypothesis."""
    if premise_sentence is None or hypothesis_sentence is None:
        return None
    
    prompt = (
        f"Sentence 1: {premise_sentence}\n"
        f"Sentence 2: {hypothesis_sentence}\n"
        f"Does Sentence 1 entail Sentence 2? Answer only 'entailment' or 'neutral'. \n "
        "Answer: "
    )
    return prompt

def create_raw_output(final_relation):
    """Create the raw output from final relation."""
    return final_relation

def compute_base_entailment(pred1, pred2):
    """Compute base entailment between predicates using entailment lookup."""
    entailment_lookup = {}
    for entry in ENTAILMENT_DATA:
        if isinstance(entry, dict) and entry.get('label') == 1:
            entailment_lookup[(entry['hyponym'], entry['hypernym'])] = "entailment"
    return entailment_lookup.get((pred1, pred2), "neutral")

def compute_final_relation(base_entailment, scope):
    """Pure function: compute final relation from base entailment and scope"""
    if scope in ["no_negation", "one_not_scoped"]:  # Even scopes
        return base_entailment  # Pass through
    elif scope == "one_scoped":  # Odd scope  
        return "neutral" if base_entailment == "entailment" else "entailment"  # Flip
    else:
        return base_entailment  # Default fallback

def get_premise_sentence(pred1, pred2, scope, datasets_dict):
    """Get premise sentence based on predicates and scope."""
    if scope not in datasets_dict:
        return None
        
    df = datasets_dict[scope]
    
    # Determine column names based on dataset
    if scope == "one_scoped":
        sent1_col = 'sentence1'
    else:
        sent1_col = 'sentence1_edited' if 'sentence1_edited' in df.columns else 'sentence1'
    
    # Find matching row
    matches = df[(df['sentence1_lex'] == pred1) & (df['sentence2_lex'] == pred2)]
    
    if matches.empty:
        return None
        
    chosen_row = matches.sample(n=1).iloc[0]
    return chosen_row[sent1_col]

def get_hypothesis_sentence(pred1, pred2, scope, datasets_dict):
    """Get hypothesis sentence based on predicates and scope."""
    if scope not in datasets_dict:
        return None
        
    df = datasets_dict[scope]
    
    # Determine column names based on dataset
    if scope == "one_scoped":
        sent2_col = 'sentence2'
    else:
        sent2_col = 'sentence2_edited' if 'sentence2_edited' in df.columns else 'sentence2'
    
    # Find matching row
    matches = df[(df['sentence1_lex'] == pred1) & (df['sentence2_lex'] == pred2)]
    
    if matches.empty:
        return None
        
    chosen_row = matches.sample(n=1).iloc[0]
    return chosen_row[sent2_col]


## debug code

def debug_pipeline_generation(pipeline, prompt):
    """Debug the pipeline generation process step by step."""
    print(f"\n=== PIPELINE GENERATION DEBUG ===")
    
    # Check model state
    print(f"Model device: {pipeline.model.device}")
    print(f"Model dtype: {pipeline.model.dtype}")
    
    # Check tokenization
    inputs = pipeline.load([prompt])
    print(f"Tokenized input shape: {inputs['input_ids'].shape}")
    print(f"Input length: {inputs['input_ids'].shape[1]}")
    
    # Check generation parameters
    print(f"Pipeline max_new_tokens: {getattr(pipeline, 'max_new_tokens', 'Not set')}")
    
    # Try raw model generation
    try:
        with torch.no_grad():
            # Generate with explicit parameters
            outputs = pipeline.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=20,  # Force this parameter
                do_sample=False,
                pad_token_id=pipeline.tokenizer.pad_token_id,
                eos_token_id=pipeline.tokenizer.eos_token_id,
            )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            decoded = pipeline.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            print(f"Raw model output: '{decoded}'")
            print(f"Raw model output repr: {repr(decoded)}")
            print(f"New tokens generated: {len(new_tokens)}")
            print(f"New token IDs: {new_tokens.tolist()}")
            
    except Exception as e:
        print(f"Error in raw generation: {e}")
        import traceback
        traceback.print_exc()


# Add debugging to see what's actually being fed to the model
def debug_pipeline_input(pipeline, raw_input_text):
    """Debug what the pipeline is actually receiving and processing."""
    print(f"\n=== PIPELINE DEBUG ===")
    print(f"Raw input text type: {type(raw_input_text)}")
    print(f"Raw input text: {repr(raw_input_text)}")
    print(f"Raw input length: {len(raw_input_text) if raw_input_text else 0}")
    
    # Check what load() produces
    try:
        loaded = pipeline.load(raw_input_text)
        print(f"Loaded keys: {loaded.keys()}")
        print(f"Input IDs shape: {loaded['input_ids'].shape}")
        print(f"Input IDs: {loaded['input_ids'][0][:20]}...")  # First 20 tokens
        
        # Decode back to text to verify
        decoded = pipeline.tokenizer.decode(loaded['input_ids'][0])
        print(f"Decoded text: {repr(decoded[:200])}...")  # First 200 chars
        
    except Exception as e:
        print(f"ERROR in pipeline.load(): {e}")
    
    print("=" * 30)

def test_direct_model_call(pipeline, prompt):
    """Test direct model generation without pipeline wrapper."""
    print(f"\n=== DIRECT MODEL TEST ===")
    
    # Tokenize manually
    inputs = pipeline.tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        add_special_tokens=True
    ).to(pipeline.device)
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input length: {inputs['input_ids'].shape[1]}")
    
    # Generate directly with model
    with torch.no_grad():
        outputs = pipeline.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=pipeline.tokenizer.pad_token_id,
            eos_token_id=pipeline.tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
        )
    
    print(f"Output shape: {outputs.shape}")
    print(f"Full output: {outputs[0].tolist()}")
    
    # Extract only new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    print(f"New tokens: {new_tokens.tolist()}")
    
    # Decode
    decoded = pipeline.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"Decoded: '{decoded}'")
    
    return decoded


# Updated demo function with pipeline debugging
def demo_examples(counterfactual_datasets, scone_model, pipeline, num_examples=3):
    """Demo examples from the datasets."""
    dataset = next(iter(counterfactual_datasets.values()))
    
    print(f"\n=== Showing {num_examples} examples ===")
    for i in range(min(num_examples, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Example {i+1} ---")
        print("CAUSAL MODEL INPUT:", {k: v for k, v in sample["input"].items() if k in ["Predicate1", "Predicate2", "Scope"]})
        
        raw_input_text = sample["input"]["raw_input"]
        print("RAW PROMPT:")
        print(repr(raw_input_text))  # Use repr to see exact string including newlines
        print("EXPECTED OUTPUT:", sample["input"]["raw_output"])
        
        # DEBUG: Check pipeline input processing
        debug_pipeline_input(pipeline, raw_input_text)
        
        # Generate LLM output
        try:
            output = pipeline.generate(raw_input_text)
            dumped_output = pipeline.dump(output)
            print("LLM RAW OUTPUT:", repr(dumped_output))
            print("LLM PROCESSED:", dumped_output)
            
            # Test the checker
            is_correct = checker(dumped_output, sample["input"]["raw_output"], debug=False)
            print("CHECKER RESULT:", is_correct)
        except Exception as e:
            print(f"ERROR during generation: {e}")
        
        print("-" * 40)

# Also check what create_raw_input is actually producing
def debug_raw_input_creation():
    """Debug the raw input creation process."""
    print("\n=== RAW INPUT CREATION DEBUG ===")
    
    # Test with sample data
    premise = "There is a man wearing a hat."
    hypothesis = "There is a man wearing a sunhat."
    
    raw_input = create_raw_input(premise, hypothesis)
    print(f"Premise: {repr(premise)}")
    print(f"Hypothesis: {repr(hypothesis)}")
    print(f"Created raw_input: {repr(raw_input)}")
    print(f"Raw input type: {type(raw_input)}")
    print(f"Raw input length: {len(raw_input) if raw_input else 0}")
    
    # Test with None values
    raw_input_none = create_raw_input(None, hypothesis)
    print(f"With None premise: {repr(raw_input_none)}")
    
    raw_input_none2 = create_raw_input(premise, None)
    print(f"With None hypothesis: {repr(raw_input_none2)}")

# Check if the issue is in the mechanism itself
def debug_mechanism_outputs(scone_model, datasets_dict):
    """Debug what each mechanism is producing."""
    print("\n=== MECHANISM DEBUG ===")
    
    # Test a known good example
    scope = "no_negation"
    df = datasets_dict[scope]
    row = df.iloc[0]  # Take first row
    
    pred1 = row["sentence1_lex"]
    pred2 = row["sentence2_lex"]
    
    print(f"Testing with pred1={pred1}, pred2={pred2}, scope={scope}")
    
    # Test each mechanism individually
    try:
        premise = get_premise_sentence(pred1, pred2, scope, datasets_dict)
        print(f"Premise sentence: {repr(premise)}")
    except Exception as e:
        print(f"ERROR in get_premise_sentence: {e}")
    
    try:
        hypothesis = get_hypothesis_sentence(pred1, pred2, scope, datasets_dict)
        print(f"Hypothesis sentence: {repr(hypothesis)}")
    except Exception as e:
        print(f"ERROR in get_hypothesis_sentence: {e}")
    
    try:
        raw_input = create_raw_input(premise, hypothesis)
        print(f"Raw input: {repr(raw_input)}")
    except Exception as e:
        print(f"ERROR in create_raw_input: {e}")


def verify_counterfactual_control(datasets_dict, scone_model, num_samples=2):
    """
    Generate samples from each counterfactual type to verify they only control
    for the variables mentioned in their names.
    """
    print("=== COUNTERFACTUAL VERIFICATION ===\n")
    
    # Build all counterfactual types
    counterfactual_datasets = build_counterfactual_datasets(datasets_dict, scone_model, size=10)
    
    for cf_name, cf_dataset in counterfactual_datasets.items():
        print(f"🔍 TESTING: {cf_name}")
        print("=" * 60)
        
        for i in range(num_samples):
            sample = cf_dataset[i]
            original = sample["input"]
            counterfactual = sample["counterfactual_inputs"][0]
            
            print(f"\n--- Sample {i+1} ---")
            print("ORIGINAL:")
            print(f"  Predicate1: {original.get('Predicate1', 'N/A')}")
            print(f"  Predicate2: {original.get('Predicate2', 'N/A')}")
            print(f"  Scope: {original.get('Scope', 'N/A')}")
            print(f"  BaseEntailment: {original.get('BaseEntailment', 'N/A')}")
            print(f"  FinalRelation: {original.get('FinalRelation', 'N/A')}")
            
            print("COUNTERFACTUAL:")
            print(f"  Predicate1: {counterfactual.get('Predicate1', 'N/A')}")
            print(f"  Predicate2: {counterfactual.get('Predicate2', 'N/A')}")
            print(f"  Scope: {counterfactual.get('Scope', 'N/A')}")
            print(f"  BaseEntailment: {counterfactual.get('BaseEntailment', 'N/A')}")
            print(f"  FinalRelation: {counterfactual.get('FinalRelation', 'N/A')}")
            
            print("CHANGES:")
            changes = []
            for var in ['Predicate1', 'Predicate2', 'Scope', 'BaseEntailment', 'FinalRelation']:
                orig_val = original.get(var, 'N/A')
                cf_val = counterfactual.get(var, 'N/A')
                if orig_val != cf_val:
                    changes.append(f"  {var}: {orig_val} → {cf_val}")
            
            if changes:
                print("\n".join(changes))
            else:
                print("  No changes detected!")
            
            print("-" * 40)
        
        print("\n" + "=" * 60 + "\n")

# Updated main function
def main(debug=False):
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    add_project_root()
    cuda_ok = setup_cuda()
    
    # Load SCONE data
    datasets_dict = load_scone_datasets(half=True)
    entailment_data = load_entailment_data()
    
    # Create valid pairs
    valid_pairs = []
    for entry in entailment_data:
        if entry['label'] == 1:
            valid_pairs.append((entry['hyponym'], entry['hypernym']))
    
    # Define causal model components (MUST include raw_input and raw_output)
    variables = ["Predicate1", "Predicate2", "Scope", "BaseEntailment", 
                "FinalRelation", "PremiseSentence", "HypothesisSentence", 
                "raw_input", "raw_output"]
    
    scope_types = ["no_negation", "one_scoped", "one_not_scoped"]
    
    values = {
        "Predicate1": [pair[0] for pair in valid_pairs],
        "Predicate2": [pair[1] for pair in valid_pairs], 
        "Scope": scope_types,
        "BaseEntailment": ["entailment", "neutral"],
        "FinalRelation": ["entailment", "neutral"],
        "PremiseSentence": None,  # Dynamic
        "HypothesisSentence": None,  # Dynamic
        "raw_input": None,  # Dynamic
        "raw_output": ["entailment", "neutral"]
    }
    
    parents = {
        "Predicate1": [],
        "Predicate2": [],
        "Scope": [],
        "BaseEntailment": ["Predicate1", "Predicate2"],
        "FinalRelation": ["BaseEntailment", "Scope"],  # Keep the new causal structure
        "PremiseSentence": ["Predicate1", "Predicate2", "Scope"],
        "HypothesisSentence": ["Predicate1", "Predicate2", "Scope"],
        "raw_input": ["PremiseSentence", "HypothesisSentence"],
        "raw_output": ["FinalRelation"]
    }

    # Define mechanisms (including required raw_input and raw_output)
    mechanisms = {
        # Keep these as you requested
        "Predicate1": lambda: random.choice([pair[0] for pair in valid_pairs]),
        "Predicate2": lambda: random.choice([pair[1] for pair in valid_pairs]),
        "Scope": lambda: random.choice(scope_types),
        
        # True causal functions that recompute when inputs change
        "BaseEntailment": compute_base_entailment,  # Takes (p1, p2)
        "FinalRelation": compute_final_relation,    # Takes (base_entailment, scope)
        
        # Sentence generation from semantic content
        "PremiseSentence": lambda p1, p2, scope: get_premise_sentence(p1, p2, scope, datasets_dict),
        "HypothesisSentence": lambda p1, p2, scope: get_hypothesis_sentence(p1, p2, scope, datasets_dict),
        
        # I/O functions
        "raw_input": create_raw_input,
        "raw_output": create_raw_output
    }
    
    # Create SCONE causal model with all components
    scone_model = SconeCausalModel(datasets_dict, entailment_data, variables, values, parents, mechanisms)
    
    
    if debug:
        debug_raw_input_creation()
    
        # Create SCONE causal model
        scone_model = SconeCausalModel(datasets_dict, entailment_data, variables, values, parents, mechanisms)
        
        # DEBUG: Test mechanisms
        debug_mechanism_outputs(scone_model, datasets_dict)
        
        # Create counterfactual datasets (small size for debugging)
        counterfactual_datasets = build_counterfactual_datasets(datasets_dict, scone_model, size=100)
        
        # Initialize pipeline
        pipeline = init_pipeline(model_name=model_name)

        # # Demo examples with detailed debugging
        # demo_examples(counterfactual_datasets, scone_model, pipeline, num_examples=5)

    #debugging counterfactual dataset
    # verify_counterfactual_control(datasets_dict, scone_model, num_samples=3)
    
    # Check if filtered datasets already exist
    filtered_datasets_path = "data/counterfactuals_filtered"
    if not os.path.exists(filtered_datasets_path) or not os.listdir(filtered_datasets_path):
        print("No filtered datasets found - generating new counterfactual datasets...")
        # Create counterfactual datasets (for filtering)
        counterfactual_datasets = build_counterfactual_datasets(datasets_dict, scone_model, size=200)
        
        #Initialize pipeline
        pipeline = init_pipeline()
        
        #Filter datasets
        filtered_datasets = run_filter_experiment(pipeline, scone_model, counterfactual_datasets)
        save_filtered_datasets(filtered_datasets)
    else:
        print("Loading existing filtered datasets...")
        filtered_datasets = {}
        for filename in os.listdir(filtered_datasets_path):
            dataset_name = filename.replace('.jsonl', '')
            dataset_path = os.path.join(filtered_datasets_path, filename)
            
            examples = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            filtered_datasets[dataset_name] = examples
            print(f"Loaded {len(examples)} examples from {filename}")
            
        pipeline = init_pipeline()
    
    token_positions = [
        TokenPosition(lambda x: get_predicate_premise(x, pipeline, debug=debug), pipeline, id="premise_predicate"),
        TokenPosition(lambda x: get_predicate_hypothesis(x, pipeline, debug=debug), pipeline, id="hypothesis_predicate"),
        TokenPosition(lambda x: get_negation_index(x, pipeline, debug=debug), pipeline, id="negation"),
        TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token")
    ]
    
    start = 22
    # end = 24
    # for full layers experiment
    end = pipeline.get_num_layers()
    target_variables_list = [["FinalRelation"], ["BaseEntailment"], ["Scope"]]
    results_dir = "SCONE_demo_results"
    
    config = {
        "batch_size": 16, 
        "evaluation_batch_size": 128, 
        "training_epoch": 32, 
        "n_features": 16, 
    }
    
    experiment = PatchResidualStream(pipeline, scone_model, list(range(start, end)), token_positions, checker, config=config)
    experiment.train_interventions(filtered_datasets, ["Scope"], method="DAS", verbose=False)
    raw_results = experiment.perform_interventions(filtered_datasets, verbose=True, target_variables_list=target_variables_list, save_dir=results_dir)
    
    # experiment.plot_heatmaps(raw_results, ["Scope"])
    
if __name__ == "__main__":
    main(debug=False)