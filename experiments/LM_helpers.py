from experiments.pyvene_core import _prepare_intervenable_inputs
import torch
import logging


def LM_loss_and_metric_fn(pipeline, intervenable_model, batch, model_units_list):
    """
    Calculate loss and evaluation metrics for language model interventions.
    
    This function evaluates intervention effects by:
    
    1. Preparing intervenable inputs from the batch
    2. Concatenating ground truth label tokens to the base inputs
       (e.g., if input has length 10 and labels length 3, creates sequence of length 13)
    3. Running the intervenable model's forward pass with these concatenated inputs
       and applying interventions at specified locations
    4. Extracting logits corresponding only to the positions where labels were appended
       (e.g., positions 9-11 in the example above)
    5. Computing accuracy and loss by comparing predicted continuations against ground truth
    
    This approach allows measuring how interventions affect the model's ability
    to predict the correct continuation, even for multi-token responses.
    
    Args:
        pipeline: The language model pipeline handling tokenization and generation
        intervenable_model: The model with intervention capabilities
        batch: Batch of data containing inputs and counterfactual inputs
        model_units_list: List of model units to intervene on
        
    Returns:
        tuple: (loss, eval_metrics, logging_info)
    """
    # Prepare intervenable inputs
    batched_base, batched_counterfactuals, inv_locations, feature_indices = _prepare_intervenable_inputs(
        pipeline, batch, model_units_list)

    # Get ground truth labels
    batched_inv_label = batch['label']
    batched_inv_label = pipeline.load(
        batched_inv_label, max_length=pipeline.max_new_tokens, padding_side='right', add_special_tokens=False)
    
    # Concatenate labels to base inputs for evaluation
    for k in batched_base:
        if isinstance(batched_base[k], torch.Tensor):
            batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)
    
    # Run the intervenable model with interventions
    _, counterfactual_logits = intervenable_model(
        batched_base, batched_counterfactuals, unit_locations=inv_locations, subspaces=feature_indices)
    
    # Extract relevant portions of logits and labels for evaluation
    labels = batched_inv_label['input_ids']
    logits = counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1]
    pred_ids = torch.argmax(logits, dim=-1)
    
    # Compute metrics and loss
    eval_metrics = compute_metrics(pred_ids, labels, pipeline.tokenizer.pad_token_id)
    loss = compute_cross_entropy_loss(logits, labels, pipeline.tokenizer.pad_token_id)
    
    # Collect detailed information for logging
    logging_info = {
        "preds": pipeline.dump(pred_ids), 
        "labels": pipeline.dump(labels),
        "base_ids": batched_base["input_ids"][0],
        "base_masks": batched_base["attention_mask"][0],
        "counterfactual_masks": [c["attention_mask"][0] for c in batched_counterfactuals],
        "counterfactual_ids": [c["input_ids"][0] for c in batched_counterfactuals],
        "base_inputs": pipeline.dump(batched_base["input_ids"][0]),
        "counterfactual_inputs": [pipeline.dump(c["input_ids"][0]) for c in batched_counterfactuals],
        "inv_locations": inv_locations,
        "feature_indices": feature_indices
    }
    
    return loss, eval_metrics, logging_info

def compute_metrics(predicted_token_ids, eval_labels, pad_token_id):
    """
    Compute sequence-level and token-level accuracy metrics.
    
    Args:
        predicted_token_ids (torch.Tensor): Predicted token IDs from the model
        eval_labels (torch.Tensor): Ground truth token IDs 
        pad_token_id (int): ID of the padding token to be ignored in evaluation
    
    Returns:
        dict: Dictionary containing accuracy metrics:
            - accuracy: Proportion of sequences where all tokens match
            - token_accuracy: Proportion of individual tokens that match
    """
    # Create mask to ignore pad tokens in labels
    mask = (eval_labels != pad_token_id)

    # Calculate token-level accuracy (only for non-pad tokens)
    correct_tokens = (predicted_token_ids == eval_labels) & mask
    token_accuracy = correct_tokens.sum().float() / mask.sum() if mask.sum() > 0 else torch.tensor(1.0)

    # Calculate sequence-level accuracy (sequence correct if all non-pad tokens correct)
    sequence_correct = torch.stack([torch.all(correct_tokens[i, mask[i]]) for i in range(eval_labels.shape[0])])
    sequence_accuracy = sequence_correct.float().mean() if len(sequence_correct) > 0 else torch.tensor(1.0)

    return {
        "accuracy": float(sequence_accuracy.item()),
        "token_accuracy": float(token_accuracy.item())
    }

def compute_cross_entropy_loss(eval_preds, eval_labels, pad_token_id):
    """
    Compute cross-entropy loss over non-padding tokens.
    
    Args:
        eval_preds (torch.Tensor): Model predictions of shape (batch_size, seq_length, vocab_size)
        eval_labels (torch.Tensor): Ground truth labels of shape (batch_size, seq_length)
        pad_token_id (int): ID of the padding token to be ignored in loss calculation
    
    Returns:
        torch.Tensor: The computed cross-entropy loss
    """
    # Reshape predictions to (batch_size * sequence_length, vocab_size)
    batch_size, seq_length, vocab_size = eval_preds.shape
    preds_flat = eval_preds.reshape(-1, vocab_size)

    # Reshape labels to (batch_size * sequence_length)
    labels_flat = eval_labels.reshape(-1)

    # Create mask for non-pad tokens
    mask = labels_flat != pad_token_id

    # Only compute loss on non-pad tokens by filtering predictions and labels
    active_preds = preds_flat[mask]
    active_labels = labels_flat[mask]

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(active_preds, active_labels)

    return loss