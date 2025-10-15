from experiments.pyvene_core import _prepare_intervenable_inputs
import torch
import logging
from typing import Callable, Any, Optional


def default_checker(neural_output, causal_output):
    """
    Default checker that replicates the original exact token matching behavior.
    This maintains backward compatibility with existing experiments.

    Args:
        neural_output: Dict with 'string' key containing predicted text
        causal_output: Dict with 'string' key containing expected text

    Returns:
        bool: True if strings match exactly (after stripping), False otherwise
    """
    pred_str = neural_output["string"].strip()
    expected_str = causal_output["string"].strip()
    return pred_str == expected_str


def LM_loss_and_metric_fn(pipeline, intervenable_model, batch, model_units_list, checker=None):
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
        checker: Function for metric evaluation. If None, uses default_checker (exact matching).
                Should have signature: checker(neural_output, causal_output) -> bool/float

    Returns:
        tuple: (loss, eval_metrics, logging_info)
    """
    # Use default checker if none provided
    if checker is None:
        checker = default_checker
    # Prepare intervenable inputs
    batched_base, batched_counterfactuals, inv_locations, feature_indices = _prepare_intervenable_inputs(
        pipeline, batch, model_units_list)

    # Get ground truth labels
    batched_inv_label = batch['label']
    if isinstance(batched_inv_label[0], dict):
        batched_inv_label = [item['string'] for item in batched_inv_label]
    batched_inv_label = pipeline.load(
        batched_inv_label,
        max_length=pipeline.max_new_tokens,
        padding_side='right',
        add_special_tokens=False,
        use_chat_template=False)

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

    # Compute metrics using checker function
    scores = []
    for i in range(pred_ids.shape[0]):
        # Decode predictions and labels to strings
        pred_str = pipeline.dump(pred_ids[i:i+1])

        # Create output dicts in same format as perform_interventions
        neural_output = {"string": pred_str}

        # Apply checker function
        score = checker(neural_output, batch["label"][i])
        if isinstance(score, torch.Tensor):
            score = score.item()
        scores.append(float(score))

    accuracy = sum(scores) / len(scores) if scores else 1.0
    eval_metrics = {
        "accuracy": accuracy,
        "token_accuracy": accuracy
    }

    # Compute loss
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

def LM_logit_loss_and_metric_fn(pipeline, intervenable_model, batch, model_units_list):
    """
    Calculate loss and evaluation metrics for language model interventions using logit distributions.

    This function is designed for labels that are dictionaries mapping token strings to expected logit values.
    For example: {"Jeremy": 2.0, "Kelly": 0.0, "Alice": None, ...}

    The function:
    1. Extracts tokens with non-None logit values from the label dictionaries
    2. Tokenizes each token string and uses the first token ID
    3. Runs the model with interventions to get predicted logits
    4. Extracts model logits for the relevant token positions
    5. Compares predicted vs expected logit distributions using loss and accuracy metrics

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

    # Run the intervenable model with interventions
    _, counterfactual_logits = intervenable_model(
        batched_base, batched_counterfactuals, unit_locations=inv_locations, subspaces=feature_indices)

    # Get the logits at the last position (where we predict the next token)
    # Shape: (batch_size, vocab_size)
    pred_logits = counterfactual_logits.logits[:, -1, :]

    # Process labels to extract token IDs and expected logits
    batched_inv_label = batch['label']
    batch_size = len(batched_inv_label)

    # Extract relevant token IDs and their expected logit values for each example
    token_ids_list = []
    expected_logits_list = []

    for example_label in batched_inv_label:
        # Get tokens with non-None values
        relevant_tokens = {k: v for k, v in example_label['logits'].items() if v is not None}

        # Tokenize each token string and get the first token ID
        token_ids = []
        expected_logits = []
        for token_str, expected_logit in relevant_tokens.items():
            # Tokenize the token string (may produce multiple tokens)
            tokenized = pipeline.tokenizer.encode(token_str, add_special_tokens=False)
            if len(tokenized) > 0:
                # Use the first token ID
                token_ids.append(tokenized[0])
                expected_logits.append(expected_logit)

        token_ids_list.append(token_ids)
        expected_logits_list.append(expected_logits)

    # Compute loss and metrics
    loss = compute_logit_distribution_loss(pred_logits, token_ids_list, expected_logits_list)
    eval_metrics = compute_logit_distribution_metrics(pred_logits, token_ids_list, expected_logits_list)

    # Collect detailed information for logging
    predicted_token_ids = torch.argmax(pred_logits, dim=-1)
    logging_info = {
        "preds": pipeline.dump(predicted_token_ids),
        "labels": [example['token'] for example in batched_inv_label],
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


def compute_logit_distribution_loss(pred_logits, token_ids_list, expected_logits_list):
    """
    Compute loss based on expected logit distributions.

    This computes MSE loss between the predicted logits and expected logits
    for the relevant tokens in each example.

    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (batch_size, vocab_size)
        token_ids_list (list): List of lists of token IDs for each example
        expected_logits_list (list): List of lists of expected logit values for each example

    Returns:
        torch.Tensor: The computed loss
    """
    total_loss = 0.0
    count = 0

    for i, (token_ids, expected_logits) in enumerate(zip(token_ids_list, expected_logits_list)):
        if len(token_ids) == 0:
            continue

        # Extract predicted logits for the relevant tokens
        token_ids_tensor = torch.tensor(token_ids, device=pred_logits.device)
        pred_logits_subset = pred_logits[i, token_ids_tensor]

        # Expected logits as tensor
        expected_logits_tensor = torch.tensor(expected_logits, device=pred_logits.device, dtype=pred_logits.dtype)

        # MSE loss between predicted and expected logits
        loss = torch.nn.functional.mse_loss(pred_logits_subset, expected_logits_tensor)
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0, device=pred_logits.device)


def compute_logit_distribution_metrics(pred_logits, token_ids_list, expected_logits_list):
    """
    Compute accuracy metrics based on expected logit distributions.

    Accuracy is 1.0 if the token with the highest expected logit matches
    the token with the highest predicted logit.

    Args:
        pred_logits (torch.Tensor): Predicted logits of shape (batch_size, vocab_size)
        token_ids_list (list): List of lists of token IDs for each example
        expected_logits_list (list): List of lists of expected logit values for each example

    Returns:
        dict: Dictionary containing accuracy metric
    """
    correct = 0
    total = 0

    for i, (token_ids, expected_logits) in enumerate(zip(token_ids_list, expected_logits_list)):
        if len(token_ids) == 0:
            continue

        # Find the token with highest expected logit
        expected_best_idx = expected_logits.index(max(expected_logits))
        expected_best_token_id = token_ids[expected_best_idx]

        # Find the token with highest predicted logit (among relevant tokens)
        token_ids_tensor = torch.tensor(token_ids, device=pred_logits.device)
        pred_logits_subset = pred_logits[i, token_ids_tensor]
        pred_best_idx = torch.argmax(pred_logits_subset).item()
        pred_best_token_id = token_ids[pred_best_idx]

        # Check if they match
        if expected_best_token_id == pred_best_token_id:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 1.0

    return {
        "accuracy": accuracy,
        "token_accuracy": accuracy  # Same as accuracy for this case
    }

# ========== SAE (Sparse Autoencoder) Registry ==========

SAE_REGISTRY = {
    "google/gemma-2-2b": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical"
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "release": "llama_scope_lxr_8x",
        "sae_id_template": "l{layer}r_8x"
    }
}


def get_sae_loader(model_path: str) -> Optional[Callable[[int], Any]]:
    """
    Get a SAE loader function for a specific model.

    This function returns a closure that can load Sparse Autoencoders (SAEs)
    for different layers of a specific model. The SAEs are used for feature
    extraction and interpretation in mechanistic interpretability experiments.

    Args:
        model_path: HuggingFace model path or name (e.g., "google/gemma-2-2b")

    Returns:
        A function that takes a layer index and returns an SAE instance,
        or None if no SAE is available for the model.

    Example:
        >>> loader = get_sae_loader("google/gemma-2-2b")
        >>> if loader:
        ...     sae = loader(5)  # Load SAE for layer 5

    Note:
        Requires sae_lens package to be installed for SAE loading.
    """
    if model_path not in SAE_REGISTRY:
        return None

    config = SAE_REGISTRY[model_path]

    def sae_loader(layer: int) -> Any:
        """
        Load a SAE for a specific layer.

        Args:
            layer: Layer index to load SAE for

        Returns:
            SAE instance for the specified layer

        Raises:
            ImportError: If sae_lens is not installed
            Exception: If SAE loading fails
        """
        try:
            from sae_lens import SAE
        except ImportError:
            raise ImportError(
                "sae_lens package is required for SAE loading. "
                "Install it with: pip install sae_lens"
            )

        sae_id = config["sae_id_template"].format(layer=layer)
        sae, _, _ = SAE.from_pretrained(
            release=config["release"],
            sae_id=sae_id,
            device="cpu"
        )
        return sae

    return sae_loader