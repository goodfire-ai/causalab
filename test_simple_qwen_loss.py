"""Simple test to see if the loss function produces NaN with Qwen"""

from tasks.MCQA.mcqa import MCQA_task
import torch
from neural.pipeline import LMPipeline
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.attention_head_experiment import PatchAttentionHeads
from neural.LM_units import get_all_tokens, TokenPosition

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Testing with Qwen")
model_name = "Qwen/Qwen2.5-0.5B"
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float16, max_length=32)
pipeline.tokenizer.padding_side = "left"

causal_model = MCQA_task.causal_models["positional"]

def checker(neural_output, causal_output):
    return causal_output in neural_output["string"] or neural_output["string"] in causal_output

# Create small dataset
size = 64
counterfactual_datasets = MCQA_task.create_datasets(size)

exp = FilterExperiment(pipeline, causal_model, checker)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=False, batch_size=128)

print(f"Dataset sizes: {[(k, len(v.dataset)) for k, v in filtered_datasets.items()]}")

# Setup experiment with masking
config = {
    "batch_size": 16,
    "evaluation_batch_size": 16,
    "training_epoch": 1,
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

# Use a few heads
heads_masking = [[(layer, head) for layer in range(0, end) for head in range(num_heads)]]

print(f"Model: {num_heads} heads, {end} layers")

# Patch the loss function to add debugging
import experiments.LM_experiments.LM_utils as lm_utils

original_loss_fn = lm_utils.compute_cross_entropy_loss

def debug_loss_fn(eval_preds, eval_labels, pad_token_id):
    """Instrumented version of compute_cross_entropy_loss"""
    print(f"\n[DEBUG] Loss computation:")
    print(f"  eval_preds shape: {eval_preds.shape}")
    print(f"  eval_preds dtype: {eval_preds.dtype}")
    print(f"  eval_labels shape: {eval_labels.shape}")

    # Check for NaN/Inf in predictions
    nan_count = torch.isnan(eval_preds).sum().item()
    inf_count = torch.isinf(eval_preds).sum().item()
    print(f"  NaN in preds: {nan_count}, Inf in preds: {inf_count}")

    if nan_count == 0 and inf_count == 0:
        print(f"  Preds min: {eval_preds.min().item():.4f}, max: {eval_preds.max().item():.4f}")
        print(f"  Preds mean: {eval_preds.mean().item():.4f}, std: {eval_preds.std().item():.4f}")
    else:
        print(f"  Preds have NaN/Inf!")
        print(f"  Finite values - min: {eval_preds[torch.isfinite(eval_preds)].min().item():.4f}")
        print(f"  Finite values - max: {eval_preds[torch.isfinite(eval_preds)].max().item():.4f}")

    # Now call original
    loss = original_loss_fn(eval_preds, eval_labels, pad_token_id)
    print(f"  Loss: {loss.item()}")
    return loss

lm_utils.compute_cross_entropy_loss = debug_loss_fn

experiment = PatchAttentionHeads(pipeline, causal_model, heads_masking, all_tokens, checker, config=config)

print("\nStarting training...")
# Pick dataset with most examples
dataset_name = max(filtered_datasets.keys(), key=lambda k: len(filtered_datasets[k].dataset))
print(f"Using dataset: {dataset_name}")

experiment.train_interventions(filtered_datasets[dataset_name], ["answer"], method="DBM", verbose=True)
