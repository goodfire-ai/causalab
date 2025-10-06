"""Test script to reproduce NaN loss issue with Qwen in DBM training"""

from tasks.MCQA.mcqa import MCQA_task
import torch
from neural.pipeline import LMPipeline
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.attention_head_experiment import PatchAttentionHeads
from neural.LM_units import get_all_tokens, TokenPosition

device = "cuda" if torch.cuda.is_available() else "cpu"

# Test with Qwen
print("="*80)
print("Testing with Qwen/Qwen2.5-0.5B")
print("="*80)

model_name = "Qwen/Qwen2.5-0.5B"
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float16, max_length=32)
pipeline.tokenizer.padding_side = "left"

causal_model = MCQA_task.causal_models["positional"]

def checker(neural_output, causal_output):
    return causal_output in neural_output["string"] or neural_output["string"] in causal_output

# Create small dataset for testing
size = 64
counterfactual_datasets = MCQA_task.create_datasets(size)

# Filter the datasets
print("\nFiltering datasets...")
exp = FilterExperiment(pipeline, causal_model, checker)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=128)

# Setup DBM experiment with minimal regularization
config = {
    "batch_size": 32,
    "evaluation_batch_size": 128,
    "training_epoch": 3,  # Just 3 epochs for testing
    "masking":{
        "regularization_coefficient": 0,
    },
    "featurizer_kwargs": {
        "tie_masks": True
    }
}

num_heads = pipeline.get_num_attention_heads()
end = pipeline.get_num_layers()
all_tokens = TokenPosition(lambda x: get_all_tokens(x, pipeline), pipeline, id="all_tokens")

heads_masking = [[(layer, head) for layer in range(0, end) for head in range(num_heads)]]

print(f"\nModel info:")
print(f"  Num layers: {end}")
print(f"  Num heads: {num_heads}")
print(f"  Hidden size: {pipeline.model.config.hidden_size}")

experiment = PatchAttentionHeads(pipeline, causal_model, heads_masking, all_tokens, checker, config=config)

print("\nStarting DBM training...")
try:
    experiment.train_interventions(filtered_datasets["random_counterfactual"], ["answer"], method="DBM", verbose=True)
    print("\n✓ Training completed successfully!")
except Exception as e:
    print(f"\n✗ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
