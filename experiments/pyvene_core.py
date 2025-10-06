"""
pyvene_core.py
==============
Core utilities for running intervention experiments.

This module provides functions for creating, managing, and running interventions
on the pyvene library. Key components include model preparation, data handling,
intervention execution, and training functions.
"""

import gc
import collections
import logging
from typing import List, Dict, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
import pyvene as pv
import numpy as np
from tqdm import *

from causal.counterfactual_dataset import CounterfactualDataset
from neural.pipeline import Pipeline
from neural.model_units import AtomicModelUnit

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set to INFO by default, debug disabled

def shallow_collate_fn(batch):
    """Only batch at dictionary level, preserve nested structures"""
    return {key: [item[key] for item in batch] for key in batch[0].keys()}

def _delete_intervenable_model(intervenable_model):
    """
    Delete the intervenable model and clear CUDA memory.
    
    This function properly cleans up an intervenable model by moving it to CPU first,
    then deleting it and clearing all CUDA caches to prevent memory leaks.
    
    Args:
        intervenable_model: The pyvene intervenable model to be deleted
    """
    intervenable_model.set_device("cpu", set_model=False)
    del intervenable_model
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return


def _prepare_intervenable_model(pipeline: Pipeline, model_units_list: List[List[AtomicModelUnit]], intervention_type: str = "interchange"):
    """
    Prepare an intervenable model for specified model units and intervention type.
    
    Creates a pyvene IntervenableModel configured for the specified intervention type
    and model units. Handles both static and dynamic index configurations. The intervention
    configs are linked across inner lists meaning those components share a counterfactual input.
    
    Args:
        pipeline (Pipeline): The pipeline containing the base model
        model_unit_lists (List[List[AtomicModelUnit]]): A list of lists of model units to be intervened on.
            The inner lists contain model components that are intervened on together with one counterfactual input.
        intervention_type (str): The type of intervention to use ("interchange", "collect", or "mask")
    
    Returns:
        intervenable_model: The prepared intervenable model on the pipeline's device
    """
    # Check if all model units have static indices
    # If all indices are static, we can use a more efficient model
    static = True
    for model_units in model_units_list:
        for model_unit in model_units:
            if not model_unit.is_static():
                static = False

    # Create intervention configs for all model units
    configs = []
    for i, model_units in enumerate(model_units_list): 
        for model_unit in model_units:
            config = model_unit.create_intervention_config(i, intervention_type)
            configs.append(config)

    # Create the intervenable model with the collected configs
    intervention_config = pv.IntervenableConfig(configs)
    intervenable_model = pv.IntervenableModel(intervention_config, model=pipeline.model, use_fast=static)
    intervenable_model.set_device(pipeline.model.device)
    
    return intervenable_model

def _prepare_intervenable_inputs(pipeline, batch, model_units_list):
    """
    Prepare the inputs for the intervenable model.
    This function loads the base and counterfactual inputs, and prepares the indices
    for the model units.

    Args:
        pipeline (Pipeline): The pipeline containing the model
        batch (dict): The batch of data containing the base and counterfactual
            inputs. The batch should contain "input" and "counterfactual_inputs" keys.
            The "counterfactual_inputs" key should contain a list of lists with shape,
            (batch_size, num_counterfactuals).
        model_units_list (List[List[AtomicModelUnit]]): A list of lists of model units to be intervened on
            The inner lists contain model components that are intervened on together with one counterfactual input.
            The outer dimension should be num_counterfactuals.
    Returns:
        batched_base: The loaded base input
        batched_counterfactuals: The loaded counterfactual inputs
        inv_locations: A dictionary containing the counterfactual and base indices
        feature_indices: A list of feature indices for each model unit

    """
    batched_base = batch["input"]
    # Change the shape of the counterfactual inputs from (batch_size, num_counterfactuals) to (num_counterfactuals, batch_size)
    batched_counterfactuals = list(zip(*batch["counterfactual_inputs"]))


    #shape: (num_model_units, batch_size, num_component_indices)
    base_indices = [
        model_unit.index_component(batched_base, batch=True)
        for model_units in model_units_list 
        for model_unit in model_units
    ]

    #shape: (num_model_units, batch_size, num_component_indices)
    counterfactual_indices = [
        model_unit.index_component(batched_counterfactual, batch=True)
        for model_units, batched_counterfactual in zip(model_units_list, batched_counterfactuals)
        for model_unit in model_units
    ]

    #shape: (num_model_units, batch_size, num_feature_indices)
    feature_indices= [
        [model_unit.get_feature_indices() for _ in range(len(batched_base))]
        for model_units in model_units_list
        for model_unit in model_units
    ]

    batched_base = pipeline.load(batched_base)
    batched_counterfactuals = [pipeline.load(batched_counterfactual) for batched_counterfactual in batched_counterfactuals]

    inv_locations = {"sources->base": (counterfactual_indices, base_indices)}
    logger.debug("base %s", base_indices)
    logger.debug("counterfactual %s", counterfactual_indices)
    for model_units in model_units_list:
        for model_unit in model_units:
            logger.debug("model_unit indices_func id: %s", model_unit.component._indices_func.id)
    # visualize_intervention_tokens(pipeline, batched_base, batched_counterfactuals, inv_locations)
    return batched_base, batched_counterfactuals, inv_locations, feature_indices

def visualize_intervention_tokens(pipeline, batched_base, batched_counterfactuals, inv_locations, max_tokens=10):
    """
    Visualizes intervention tokens by showing the specific tokens at intervention positions
    for both base and counterfactual inputs.
    
    Args:
        pipeline: Pipeline object with tokenizer for decoding tokens
        batched_base: Base inputs from prepare_interventable_inputs
        batched_counterfactuals: Counterfactual inputs from prepare_interventable_inputs
        inv_locations: Intervention locations from prepare_interventable_inputs
        max_tokens: Maximum number of tokens to display for long sequences
    """
    # Extract the first example from the base batch
    base_ids = batched_base["input_ids"][0].cpu()
    
    # Decode the full base prompt
    base_text = pipeline.tokenizer.decode(base_ids, skip_special_tokens=False)
    print("Base Prompt:")
    print(f"\"{base_text}\"")
    print()
    
    # Process each counterfactual
    counterfactual_ids_list = []
    for i, counterfactual in enumerate(batched_counterfactuals):
        cf_ids = counterfactual["input_ids"][0].cpu()
        counterfactual_ids_list.append(cf_ids)
        
        cf_text = pipeline.tokenizer.decode(cf_ids, skip_special_tokens=False)
        print(f"Counterfactual {i+1} Prompt:")
        print(f"\"{cf_text}\"")
        print()
    
    # Get the intervention locations
    if "sources->base" in inv_locations:
        source_indices, base_indices = inv_locations["sources->base"]
        
        print("Intervention Tokens:")
        # For each model unit
        for i, (source_idx_batch, base_idx_batch) in enumerate(zip(source_indices, base_indices)):
            print(f"\nModel Unit {i+1}:")
            
            # Get indices for the first example in the batch
            source_indices_for_example = source_idx_batch[0]  # First example in batch
            base_indices_for_example = base_idx_batch[0]  # First example in batch
            
            # Make sure indices are in list form
            if not isinstance(source_indices_for_example, list):
                source_indices_for_example = [source_indices_for_example]
            if not isinstance(base_indices_for_example, list):
                base_indices_for_example = [base_indices_for_example]
            
            # Limit display if too many tokens
            if len(base_indices_for_example) > max_tokens:
                base_indices_to_show = base_indices_for_example[:max_tokens]
                truncated_base = True
            else:
                base_indices_to_show = base_indices_for_example
                truncated_base = False
                
            if len(source_indices_for_example) > max_tokens:
                source_indices_to_show = source_indices_for_example[:max_tokens]
                truncated_source = True
            else:
                source_indices_to_show = source_indices_for_example
                truncated_source = False
            
            # Display base tokens
            print("  Base Token Indices:")
            for idx in base_indices_to_show:
                if isinstance(idx, list):  # Skip nested structures (like attention head indices)
                    continue
                    
                if idx < len(base_ids):
                    token = pipeline.tokenizer.decode(base_ids[idx:idx+1], skip_special_tokens=False)
                    print(f"    Position {idx}: '{token}'")
            
            if truncated_base:
                print(f"    ... and {len(base_indices_for_example) - max_tokens} more tokens")
            
            # Display counterfactual tokens
            if i < len(counterfactual_ids_list):  # Make sure we have a corresponding counterfactual
                cf_ids = counterfactual_ids_list[i]
                print("  Counterfactual Token Indices:")
                for idx in source_indices_to_show:
                    if isinstance(idx, list):  # Skip nested structures (like attention head indices)
                        continue
                        
                    if idx < len(cf_ids):
                        token = pipeline.tokenizer.decode(cf_ids[idx:idx+1], skip_special_tokens=False)
                        print(f"    Position {idx}: '{token}'")
                
                if truncated_source:
                    print(f"    ... and {len(source_indices_for_example) - max_tokens} more tokens")


def _batched_interchange_intervention(pipeline, intervenable_model, batch, model_units_list, output_scores=True):
    """
    Perform interchange interventions on batched inputs using an intervenable model.

    This function executes the core intervention logic by:
    1. Preparing the base and counterfactual inputs for intervention
    2. Running the model with interventions at specified locations
    3. Moving tensors back to CPU to free GPU memory

    Args:
        pipeline (Pipeline): Neural model pipeline that handles tokenization and generation
        intervenable_model (IntervenableModel): PyVENE model with preset intervention locations
        batch (dict): Batch of data containing "input" and "counterfactual_inputs"
        model_units_list (List[List[AtomicModelUnit]]): Model components to intervene on
        output_scores (bool): Whether to include scores in output dictionary (default: True)

    Returns:
        dict: Dictionary with 'sequences' and optionally 'scores' keys
    """
    # Prepare inputs for intervention
    batched_base, batched_counterfactuals, inv_locations, feature_indices = _prepare_intervenable_inputs(
        pipeline, batch, model_units_list)

    # Execute the intervention via the pipeline
    output = pipeline.intervenable_generate(
        intervenable_model, batched_base, batched_counterfactuals, inv_locations, feature_indices,
        output_scores=output_scores)

    # Move tensors to CPU to free GPU memory
    for batched in [batched_base] + batched_counterfactuals:
        for k, v in batched.items():
            if v is not None and isinstance(v, torch.Tensor):
                batched[k] = v.cpu()

    return output

def _run_interchange_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: CounterfactualDataset,
    model_units_list: List[List[AtomicModelUnit]],
    verbose: bool = False,
    batch_size=32,
    output_scores=True):
    """
    Run interchange interventions on a full counterfactual dataset in batches.

    This function:
    1. Prepares an intervenable model configured for interchange interventions
    2. Processes the dataset in batches, applying interventions to each batch
    3. Manages memory between batches to prevent OOM errors
    4. Collects and returns results from all batches

    Args:
        pipeline (Pipeline): Neural model pipeline that handles tokenization and generation
        counterfactual_dataset (CounterfactualDataset): Dataset containing inputs and their counterfactuals
        model_units_list (List[List[AtomicModelUnit]]): Model components to intervene on, where inner
                                                      lists share counterfactual inputs
        verbose (bool): Whether to display progress bars during processing
        batch_size (int): Number of examples to process in each batch
        output_scores (bool): Whether to include scores in output dictionary (default: True)

    Returns:
        List[dict]: List of dictionaries, each with 'sequences' and optionally 'scores' keys
    """
    # Initialize intervenable model with interchange intervention type
    intervenable_model = _prepare_intervenable_model(
        pipeline,
        model_units_list,
        intervention_type="interchange")

    # Create data loader for batch processing
    dataloader = DataLoader(
        counterfactual_dataset.dataset,
        batch_size=batch_size,
        shuffle=False,  # Maintain dataset order
        collate_fn=shallow_collate_fn  # Use custom collate function to preserve nested structures
    )
    all_outputs = []

    # Process each batch with progress tracking
    for batch in tqdm(dataloader, desc="Processing batches", disable=not verbose, leave=False):
        with torch.no_grad():  # Disable gradient tracking for inference
            # Perform interchange interventions on the batch - returns dict
            output_dict = _batched_interchange_intervention(
                    pipeline, intervenable_model, batch, model_units_list,
                    output_scores=output_scores)

            # Collect outputs from this batch
            all_outputs.append(output_dict)

    # Clean up the intervenable model to free GPU memory
    _delete_intervenable_model(intervenable_model)
    
    return all_outputs

def _collect_features(dataset, pipeline, model_units_list, config, verbose=False, collect_counterfactuals=True):
    """
    Collect internal neural network activations (features) at specified model locations.
    
    This function:
    1. Creates an intervenable model configured for feature collection
    2. Processes the dataset in batches to extract activations at target locations
    3. Optionally extracts activations for counterfactual inputs as well
    4. Organizes the activations by model unit and concatenates across batches
    
    Args:
        dataset (Dataset): The dataset containing inputs to collect features from
        pipeline (Pipeline): Neural model pipeline for processing inputs
        model_units_list (List[List[AtomicModelUnit]]): Model components to collect features from
        config (dict): Configuration parameters including batch_size
        verbose (bool): Whether to print detailed information during processing
        collect_counterfactuals (bool): Whether to collect features from counterfactual inputs too
        
    Returns:
        List[torch.Tensor]: List of feature tensors for each model unit group in model_units_list,
                           where each tensor contains the activations for all inputs in the dataset
    """
    # Initialize model with "collect" intervention type (extracts activations without modifying them)
    intervenable_model = _prepare_intervenable_model(pipeline, model_units_list, intervention_type="collect")
    

    # Create data loader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=False,  # Preserve original order
        collate_fn=shallow_collate_fn  # Use custom collate function to preserve nested structures
    )
    
    # Initialize container for collected features: one list per model unit group
    data = [[[] for _ in range(len(model_units))] for model_units in model_units_list]
    
    # Process dataset in batches with progress tracking
    for batch in tqdm(dataloader, desc="Processing batches", disable=not verbose, leave=False):
        # Prepare batch data including base and counterfactual inputs
        batched_base, batched_counterfactuals, inv_locations, feature_indices = _prepare_intervenable_inputs(
            pipeline, batch, model_units_list)
        batch_len = batched_base["input_ids"].shape[0]
        
        # Extract indices for mapping between base and source
        source_indices, base_indices= inv_locations["sources->base"]

        # Create mapping for base input activations (identical source and target)
        base_map = {"sources->base": (base_indices, base_indices)}

        # Collect activations from base inputs
        # Returns a list of activation tensors, one per model unit
        # In pyvene 0.1.8+, each tensor contains all batch samples for that unit
        base_activations = intervenable_model(batched_base, unit_locations=base_map)[0][1]

        # Helper function to process activations from both base and counterfactual inputs
        def process_activations(activations_list, model_units_list, batch_len, data_container):
            """Process activations from pyvene and add them to the data container.
            
            Handles both pyvene 0.1.8+ format (one tensor per unit) and older formats.
            
            Args:
                activations_list: List of activation tensors from pyvene
                model_units_list: List of model unit groups  
                batch_len: Number of samples in the batch
                data_container: List of lists to store processed activations
            """
            total_units = sum(len(unit_group) for unit_group in model_units_list)
            
            if len(activations_list) == total_units:
                # pyvene 0.1.8+ format: one tensor per unit containing all batch samples
                activation_idx = 0
                for i in range(len(model_units_list)):
                    for j in range(len(model_units_list[i])):
                        unit_activations = activations_list[activation_idx]
                        hidden_size = unit_activations.shape[-1]
                        activations = unit_activations.reshape(-1, hidden_size)
                        data_container[i][j].extend(activations.cpu())
                        activation_idx += 1
            else:
                raise ValueError(
                    f"Unexpected activations format. Length: {len(activations_list)}, "
                    f"Expected either {total_units} or {total_units * batch_len}"
                )
        
        # Process base activations
        process_activations(base_activations, model_units_list, batch_len, data)
        del batched_base
        del base_activations

        # Optionally collect activations from counterfactual inputs
        if collect_counterfactuals:
            source_map = {"sources->base": (source_indices, source_indices)}
            
            for counterfactual in batched_counterfactuals:
                counterfactual_activations = intervenable_model(counterfactual, unit_locations=source_map)[0][1]
                process_activations(counterfactual_activations, model_units_list, batch_len, data)
                del counterfactual_activations
            
            del batched_counterfactuals

    # Stack collected activations into 2D tensors with shape (n_samples, n_features)
    data = [[torch.stack(datum) for datum in x] for x in data]

    if verbose:
        print(f"Collected features for {len(data)} unit groups")
        print(f"Units per group: {[len(x) for x in data]}")
        print(f"Feature tensor shape: {data[0][0].shape} (samples, features)")

    # Return nested list structure:
    # data[i][j] = tensor of shape (n_samples, n_features) for unit j in group i
    return data

def _train_intervention(pipeline: Pipeline,
                        model_units_list: List[AtomicModelUnit],
                        counterfactual_dataset: CounterfactualDataset,
                        intervention_type: str,
                        config: Dict, 
                        loss_and_metric_fn: callable
                        ):
    """
    Train intervention models on a counterfactual dataset.
    
    This function implements the training loop for neural network interventions, 
    supporting both "interchange" and "mask" intervention types. It optimizes
    intervention parameters while keeping the base model frozen.
    
    Args:
        pipeline (Pipeline): Neural model pipeline for tokenization and model execution
        model_units_list (List[List[AtomicModelUnit]]): Nested list of model components to 
                                                      intervene on, where inner lists share 
                                                      counterfactual inputs
        counterfactual_dataset (CounterfactualDataset): Dataset containing original inputs 
                                                      and their counterfactuals
        intervention_type (str): Type of intervention ("interchange" or "mask")
        config (Dict): Configuration parameters including:
            - batch_size (int): Number of examples per batch
            - training_epoch (int): Maximum number of training epochs
            - init_lr (float): Initial learning rate
            - regularization_coefficient (float): Weight for sparsity regularization (mask only)
            - log_dir (str): Directory for TensorBoard logs
            - temperature_schedule (tuple): Start and end temperature for mask annealing
            - temperature_annealing_fraction (float, optional): Fraction of training steps
                                                              to anneal temperature (default: 0.5)
            - patience (int, optional): Epochs without improvement before early stopping
                                      Set to None to disable early stopping
            - scheduler_type (str, optional): Learning rate scheduler type
                                           (default: "constant")
            - memory_cleanup_freq (int, optional): Batch frequency for memory cleanup
                                               (default: 50)
            - shuffle (bool, optional): Whether to shuffle data (default: True)
        loss_and_metric_fn (callable): Function computing loss and metrics for a batch
                                     with signature (pipeline, model, batch, units) ->
                                     (loss, metrics_dict, logging_info)
    
    Returns:
        None: The trained parameters are stored directly in the model_units' featurizers.
              For mask interventions, feature_indices are also set based on training.
    """
    # ----- Model Initialization ----- #
    intervenable_model = _prepare_intervenable_model(pipeline, model_units_list, intervention_type=intervention_type)
    intervenable_model.disable_model_gradients()
    intervenable_model.eval()

    # ----- Data Preparation ----- #
    dataloader = DataLoader(
        counterfactual_dataset,
        batch_size=config["train_batch_size"],
        shuffle=config.get("shuffle", True),
        collate_fn=shallow_collate_fn  # Use custom collate function to preserve nested structures
    )

    # ----- Logging Setup ----- #
    tb_writer = SummaryWriter(config['log_dir'])
    
    # ----- Configuration ----- #
    num_epoch = config['training_epoch']
    regularization_coefficient = config.get('masking', {}).get('regularization_coefficient', 1e-4)
    memory_cleanup_freq = config.get('memory_cleanup_freq', 50)
    patience = config.get('patience', None)  # Default to no early stopping
    scheduler_type = config.get('scheduler_type', 'constant')

    # ----- Early Stopping Setup ----- #
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_enabled = patience is not None

    # ----- Optimizer Configuration ----- #
    optimizer_params = []
    for k, v in intervenable_model.interventions.items():
        tb_writer.add_text("Intervention", f"Intervention: {k}")
        if isinstance(v, tuple):
            v = v[0]
        for i, param in enumerate(v.parameters()):
            tb_writer.add_text("Parameter", f"Parameter {i}: requires_grad = {param.requires_grad}, shape = {param.shape}")
        optimizer_params += list(v.parameters())
    
    optimizer = torch.optim.AdamW(optimizer_params,
                                  lr=config['init_lr'],
                                  weight_decay=0)
                                  
    scheduler = transformers.get_scheduler(scheduler_type,
                              optimizer=optimizer,
                              num_training_steps=num_epoch * len(dataloader))
    
    tb_writer.add_text("Parameters", f"Model trainable parameters: {pv.count_parameters(intervenable_model.model)}")
    tb_writer.add_text("Parameters", f"Intervention trainable parameters: {intervenable_model.count_parameters()}")
    
    # ----- Temperature Scheduling for Mask Interventions ----- #
    temperature_schedule = None
    if (intervention_type == "mask"):
        temperature_start, temperature_end = config.get('masking', {}).get('temperature_schedule', (1.0, 0.01))
        temperature_annealing_fraction = config.get('masking', {}).get('temperature_annealing_fraction', 0.5)

        # Calculate number of steps for annealing
        total_steps = num_epoch * len(dataloader)
        annealing_steps = int(total_steps * temperature_annealing_fraction)

        # Create schedule: anneal for first fraction of steps, then stay constant
        annealing_schedule = torch.linspace(temperature_start, temperature_end, annealing_steps + 1)
        constant_schedule = torch.full((total_steps - annealing_steps,), temperature_end)
        temperature_schedule = torch.cat([annealing_schedule, constant_schedule])
        temperature_schedule = temperature_schedule.to(pipeline.model.dtype).to(pipeline.model.device)

        # Set initial temperature for all mask interventions
        for k, v in intervenable_model.interventions.items():
            if isinstance(v, tuple):
                intervenable_model.interventions[k][0].set_temperature(
                    temperature_schedule[scheduler._step_count])
            else:
                intervenable_model.interventions[k].set_temperature(
                    temperature_schedule[scheduler._step_count])

    # ----- Training Loop ----- #
    train_iterator = tqdm(range(0, int(num_epoch)),
                         desc=f"Training {str(model_units_list)[:100]}...",
                         leave=False)
    for epoch in train_iterator:
        epoch_iterator = tqdm(dataloader,
                            desc=f"Epoch: {epoch}",
                            position=1,
                            leave=False)
        
        aggregated_stats = collections.defaultdict(list)
        
        for step, batch in enumerate(epoch_iterator):
            # Move batch data to device
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(pipeline.model.device)
                    
            # Run training step
            loss, eval_metrics, logging_info = loss_and_metric_fn(
                pipeline,
                intervenable_model,
                batch,
                model_units_list
            )

            # Add sparsity loss for mask interventions
            if intervention_type == "mask":
                masks = []
                temp = temperature_schedule[scheduler._step_count]
                for k, v in intervenable_model.interventions.items():
                    if isinstance(v, tuple):
                        loss = loss + regularization_coefficient * intervenable_model.interventions[k][0].get_sparsity_loss()
                        masks.append(intervenable_model.interventions[k][0].mask)
                        intervenable_model.interventions[k][0].set_temperature(temp)
                    else:
                        loss = loss + regularization_coefficient * intervenable_model.interventions[k].get_sparsity_loss()
                        masks.append(intervenable_model.interventions[k].mask)
                        intervenable_model.interventions[k].set_temperature(temp)
                if config["featurizer_kwargs"]["tie_masks"]:
                    masks = torch.cat(masks)
                    sparse_loss = torch.norm(torch.sigmoid(masks / temp,), p=1)
                    loss = loss + regularization_coefficient * sparse_loss


            # Update statistics
            aggregated_stats['loss'].append(loss.item())
            aggregated_stats['metrics'].append(eval_metrics)
            
            # Update progress bar
            postfix = {"loss": round(np.mean(aggregated_stats['loss']), 2)}
            for k, v in eval_metrics.items():
                postfix[k] = round(np.mean(v), 2)
            epoch_iterator.set_postfix(postfix)

            # Optimization step
            loss.backward()
            optimizer.step()
            scheduler.step()
            intervenable_model.set_zero_grad()

            # Logging
            if step % 10 == 0:
                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], scheduler._step_count)
                tb_writer.add_scalar("loss", loss, scheduler._step_count)
            if step < 2 and epoch == 0:
                for k, v in logging_info.items():
                    tb_writer.add_text(k, str(v), scheduler._step_count)
                    
            # Periodic memory cleanup
            if step % memory_cleanup_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update progress bar with epoch summary
        epoch_avg_loss = np.mean(aggregated_stats['loss'])
        postfix_dict = {"loss": f"{epoch_avg_loss:.4f}"}

        if aggregated_stats['metrics']:
            # Aggregate metrics across all batches in the epoch
            all_metrics = {}
            for batch_metrics in aggregated_stats['metrics']:
                for k, v in batch_metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
            # Add metrics to postfix
            for k, v in all_metrics.items():
                postfix_dict[k] = f"{np.mean(v):.4f}"

        train_iterator.set_postfix(postfix_dict)

        # Early stopping check at end of epoch
        if early_stopping_enabled:
            epoch_avg_loss = np.mean(aggregated_stats['loss'])
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}/{num_epoch}")
                    break
    


    # ----- Finalize Logging ----- #
    tb_writer.flush()
    tb_writer.close()

    # ----- Feature Selection for Mask Interventions ----- #
    if intervention_type == "mask":
        # Flatten model_units_list
        model_units = [model_unit for model_units in model_units_list for model_unit in model_units]
        
        for kv, model_unit in zip(intervenable_model.interventions.items(), model_units):
            k, v = kv
            if isinstance(v, tuple):
                v = v[0]
                
            if config["featurizer_kwargs"]["tie_masks"]:
                # If masks are tied, use the average mask across all units
                if torch.sigmoid(v.mask[0]) > 0.5:
                    indices = None
                else:
                    indices = []
            else:
                # Get binary mask and indices
                mask_binary = (torch.sigmoid(v.mask) > 0.5).float().cpu()
                indices = torch.nonzero(mask_binary).numpy().flatten().tolist()
            
            # Update model unit
            model_unit.set_feature_indices(indices)

            # Log selected features
            num_features = "all" if indices is None else len(indices)
            tb_writer.add_text("Selected features", f"Number Selected features: {num_features}")
            tb_writer.add_text("Selected features", f"Selected features: {indices}")
            
    # ----- Cleanup ----- #
    _delete_intervenable_model(intervenable_model)

    summary = f"Trained intervention for {str(model_units_list)[:200]}"
    summary += "\nFinal metrics: " + " ".join([f"{k}: {v}" for k, v in postfix_dict.items()])
    return summary