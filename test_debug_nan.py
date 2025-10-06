"""Debug script to identify source of NaN loss with Qwen"""

from tasks.MCQA.mcqa import MCQA_task
import torch
from neural.pipeline import LMPipeline
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.attention_head_experiment import PatchAttentionHeads
from neural.LM_units import get_all_tokens, TokenPosition
from experiments.pyvene_core import _prepare_intervenable_inputs
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

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
size = 8  # Even smaller for debugging
counterfactual_datasets = MCQA_task.create_datasets(size)

# Filter the datasets
print("\nFiltering datasets...")
exp = FilterExperiment(pipeline, causal_model, checker)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=128)

print(f"\nAvailable datasets: {list(filtered_datasets.keys())}")
print(f"Dataset sizes: {[(k, len(filtered_datasets[k].dataset)) for k in filtered_datasets.keys()]}")

# Pick the first available dataset with data
dataset_name = list(filtered_datasets.keys())[0]
dataset = filtered_datasets[dataset_name]

print(f"\nUsing dataset: {dataset_name}")

# Get one batch to debug
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset.dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: {key: [item[key] for item in x] for key in x[0].keys()}
)

batch = next(iter(dataloader))

print(f"\nBatch info:")
print(f"  Keys: {batch.keys()}")
print(f"  Num examples: {len(batch['input'])}")
if 'input' in batch:
    print(f"  Base strings: {batch['input'][:2]}")

# Setup DBM experiment
config = {
    "batch_size": 4,
    "evaluation_batch_size": 4,
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

# Just use a few heads for debugging
heads_masking = [[(layer, head) for layer in range(0, 2) for head in range(0, 2)]]

print(f"\nModel info:")
print(f"  Num layers: {end}")
print(f"  Num heads: {num_heads}")
print(f"  Hidden size: {pipeline.model.config.hidden_size}")
print(f"  Head dim: {pipeline.model.config.hidden_size // num_heads}")

experiment = PatchAttentionHeads(pipeline, causal_model, heads_masking, all_tokens, checker, config=config)

# Now let's manually call the loss function to see what's happening
print("\n" + "="*80)
print("Debugging loss computation")
print("="*80)

from experiments.LM_experiments.LM_utils import LM_loss_and_metric_fn

# Get the intervenable model
model_units_list = experiment.model_units_lists[0]
intervenable_model = experiment._intervenable_models[str(model_units_list)]

# Use the same dataset for training
print(f"\nUsing dataset: {dataset_name} with {len(dataset.dataset)} examples")

print("\nRunning forward pass...")
try:
    # Prepare inputs
    batched_base, batched_counterfactuals, inv_locations, feature_indices = _prepare_intervenable_inputs(
        pipeline, batch, model_units_list)

    print(f"Base input shape: {batched_base['input_ids'].shape}")
    print(f"Num counterfactuals: {len(batched_counterfactuals)}")

    # Get labels
    batched_inv_label = batch['label']
    batched_inv_label = pipeline.load(
        batched_inv_label, max_length=pipeline.max_new_tokens, padding_side='right', add_special_tokens=False)

    print(f"Label shape: {batched_inv_label['input_ids'].shape}")

    # Concatenate
    for k in batched_base:
        if isinstance(batched_base[k], torch.Tensor):
            batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)

    print(f"Concatenated base input shape: {batched_base['input_ids'].shape}")

    # Run model
    _, counterfactual_logits = intervenable_model(
        batched_base, batched_counterfactuals, unit_locations=inv_locations, subspaces=feature_indices)

    print(f"Logits shape: {counterfactual_logits.logits.shape}")

    # Extract relevant portions
    labels = batched_inv_label['input_ids']
    logits = counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1]

    print(f"Extracted logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # Check for NaN in logits
    nan_count = torch.isnan(logits).sum().item()
    inf_count = torch.isinf(logits).sum().item()
    print(f"\nNaN count in logits: {nan_count}")
    print(f"Inf count in logits: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print(f"Logits min: {torch.nanmin(logits).item()}")
        print(f"Logits max: {torch.nanmax(logits).item()}")
        print(f"Logits mean: {torch.nanmean(logits).item()}")
    else:
        print(f"Logits min: {logits.min().item()}")
        print(f"Logits max: {logits.max().item()}")
        print(f"Logits mean: {logits.mean().item()}")

    # Check softmax
    print("\nChecking softmax...")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    nan_count_probs = torch.isnan(probs).sum().item()
    print(f"NaN count in probs: {nan_count_probs}")

    # Compute loss
    from experiments.LM_experiments.LM_utils import compute_cross_entropy_loss

    print("\nComputing loss...")
    loss = compute_cross_entropy_loss(logits, labels, pipeline.tokenizer.pad_token_id)
    print(f"Loss: {loss.item()}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
