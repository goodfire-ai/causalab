from typing import List, Dict, Callable, Tuple, Union
import gc, json, os, collections, random, copy
from itertools import chain

import pyvene as pv
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import TruncatedSVD 
from tqdm import tqdm, trange
from datasets import Dataset

from neural.model_units import *
from neural.pipeline import Pipeline
from causal.causal_model import CausalModel
from causal.counterfactual_dataset import CounterfactualDataset
from experiments.pyvene_core import _run_interchange_interventions, _train_intervention, _collect_features, shallow_collate_fn
from experiments.config import DEFAULT_CONFIG


def _extract_top_k_logits(output_dict, k, tokenizer):
    """
    Extract top-k logits with token information in JSON-serializable format.

    Args:
        output_dict: Dictionary with 'scores' (list of tensors) and 'sequences' keys
        k: Number of top logits to extract per position
        tokenizer: Tokenizer to decode token IDs

    Returns:
        List of dicts with top-k information for each position, or None if no scores
    """
    if "scores" not in output_dict or not output_dict["scores"]:
        return None

    top_k_results = []
    for position_logits in output_dict["scores"]:
        # position_logits shape: (batch_size, vocab_size)
        # We assume batch_size=1 for individual examples
        logits = position_logits[0]  # Get first (and only) example

        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, k=min(k, len(logits)))

        # Convert to JSON-serializable format
        position_result = []
        for value, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
            token = tokenizer.decode([idx])
            position_result.append({
                "token_id": idx,
                "logit": value,
                "token": token
            })
        top_k_results.append(position_result)

    return top_k_results

class InterventionExperiment:
    """
    Base class for running causal abstraction experiments with neural networks.
    
    This class provides core functionality for performing interventions on model components,
    training feature representations, and evaluating causal effects in neural networks.
    It serves as the foundation for more specialized experiment types.
    
    Attributes:
        pipeline: Neural model execution pipeline
        causal_model: High-level causal model for comparison
        model_units_lists: Triple-nested list structure of model units:
            - Outermost list: Contains units for a single intervention experiment
            - Middle list: Groups units by counterfactual input (units sharing the same input)
            - Innermost list: Individual model units to be intervened upon using a specific counterfactual input
        checker: Function to evaluate output correctness
        metadata_fn: Function to extract metadata from model units
        config: Configuration parameters for experiments
    """
    def __init__(self,
            pipeline: Pipeline,
            causal_model: CausalModel,
            model_units_lists: List[AtomicModelUnit],
            checker: Callable,
            metadata_fn=lambda x: None,
            config=None):
        """
        Initialize an InterventionExperiment with neural network and causal model.
        
        Args:
            pipeline: Neural model execution pipeline
            causal_model: High-level causal model for comparison
            model_units_lists: Components of the neural network to intervene on
            checker: Function that evaluates if model output matches expected output
            metadata_fn: Function to extract metadata from model units for analysis
            config: Configuration dictionary with experiment parameters
        """
        self.pipeline = pipeline
        self.causal_model = causal_model 
        self.model_units_lists = model_units_lists
        self.checker = checker
        self.metadata_fn = metadata_fn
        
        # Use DEFAULT_CONFIG as base and merge with provided config
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)

    def perform_interventions(self, datasets, verbose: bool = False, target_variables_list: List[List[str]] = None, save_dir=None, include_actual_outputs: bool = False) -> Dict:
        """
        Compute intervention scores across multiple counterfactual datasets and model units.

        This method runs interchange interventions on the model, comparing the outputs against
        the expectations from the causal model. It evaluates how well different neural network
        components represent the causal variables of interest.

        Args:
            datasets: Dictionary mapping dataset names to CounterfactualDataset objects
            verbose: Whether to show progress bars during execution
            target_variables_list: List of causal variable groups to evaluate
            save_dir: Directory to save results (if provided)
            include_actual_outputs: If True, also compute outputs without intervention

        Returns:
            Dictionary containing the experiment results with scores for each model unit and dataset

        Note:
            The structure of model_units_lists determines how interventions are performed:
            - Each group in the middle level shares a counterfactual input
            - Each unit in the innermost level is intervened upon using that shared input
            - This allows for complex interventions where multiple components are modified simultaneously
        """
        if isinstance(datasets, CounterfactualDataset):
            datasets = {datasets.id: datasets}
        # Initialize results structure
        results = {"method_name": self.config["method_name"],
                    "model_name": self.pipeline.model.__class__.__name__,
                    "task_name": self.causal_model.id,
                    "dataset": {dataset_name: {
                        "model_unit": {str(units_list): None for units_list in self.model_units_lists}}
                    for dataset_name in datasets.keys()}}

        # Process each dataset and model unit combination
        total_combinations = len(datasets) * len(self.model_units_lists)
        progress_bar = tqdm(total=total_combinations, desc="Running interventions", disable=not verbose)

        for dataset_name in datasets.keys():
            # Compute actual outputs (no intervention) if requested
            if include_actual_outputs:
                actual_outputs = self._compute_actual_outputs(
                    datasets[dataset_name], verbose
                )
                # Move to CPU but keep as tensors
                actual_outputs = self._move_outputs_to_cpu(actual_outputs)
                # Store at dataset level
                if "raw_outputs_no_intervention" not in results["dataset"][dataset_name]:
                    results["dataset"][dataset_name]["raw_outputs_no_intervention"] = actual_outputs

            for model_units_list in self.model_units_lists:
                progress_bar.set_postfix({"dataset": dataset_name, "units": model_units_list})
                progress_bar.update(1)
                
                # Execute interchange interventions using pyvene
                raw_outputs = _run_interchange_interventions(
                    pipeline=self.pipeline,
                    counterfactual_dataset=datasets[dataset_name],
                    model_units_list=model_units_list,
                    verbose=verbose,
                    output_scores=self.config["output_scores"],
                    batch_size=self.config["evaluation_batch_size"]
                )
                # Move to CPU but keep as tensors
                raw_outputs = self._move_outputs_to_cpu(raw_outputs)

                # Generate causal model inputs for each example in the dataset
                # This ensures the kth causal input corresponds to the kth neural network output
                causal_model_inputs = []
                for example in datasets[dataset_name]:
                    # Run the causal model to get the expected input representation
                    input_setting = example["input"]
                    counterfactual_inputs = example["counterfactual_inputs"]
                    causal_model_inputs.append({
                        "base_input": input_setting,
                        "counterfactual_inputs": counterfactual_inputs
                    })

                # Extract metadata and feature indices for this model unit
                metadata = self.metadata_fn(model_units_list)
                feature_indices = {}
                for i, model_units in enumerate(model_units_list):
                    for j, model_unit in enumerate(model_units):
                        unit_key = f"{model_unit.id}"
                        indices = model_unit.get_feature_indices()
                        feature_indices[unit_key] = indices

                results["dataset"][dataset_name]["model_unit"][str(model_units_list)] = {
                    "raw_outputs": raw_outputs,
                    "causal_model_inputs": causal_model_inputs,
                    "metadata": metadata,
                    "feature_indices": feature_indices}

                # Process and decode model outputs from dictionaries
                dumped_outputs = []
                flattened_outputs = []
                for batch_dict in raw_outputs:
                    # Use the string field that's already in batch_dict
                    batch_strings = batch_dict["string"]
                    if isinstance(batch_strings, list):
                        dumped_outputs.extend(batch_strings)
                        # Create individual output dicts for each example in the batch
                        for idx, decoded_str in enumerate(batch_strings):
                            example_dict = {"sequences": batch_dict["sequences"][idx:idx+1]}
                            if "scores" in batch_dict:
                                example_dict["scores"] = [score[idx:idx+1] for score in batch_dict["scores"]]
                            example_dict["string"] = decoded_str
                            flattened_outputs.append(example_dict)
                    else:
                        dumped_outputs.append(batch_strings)
                        flattened_outputs.append(batch_dict)
                
                # Evaluate results for each target variable group
                if target_variables_list is not None:
                    for target_variables in target_variables_list:
                        target_variable_str = "-".join(target_variables)

                        # Generate expected outputs from causal model
                        labeled_data = self.causal_model.label_counterfactual_data(datasets[dataset_name], target_variables)
                        assert len(labeled_data) == len(dumped_outputs), f"Length mismatch: {len(labeled_data)} vs {len(dumped_outputs)}"
                        assert len(labeled_data) == len(flattened_outputs), f"Length mismatch: {len(labeled_data)} vs {len(flattened_outputs)}"

                        # Compute intervention scores - pass neural dict and expected label
                        scores = []
                        for example, output_dict in zip(labeled_data, flattened_outputs):
                            score = self.checker(output_dict, example["label"])
                            if isinstance(score, torch.Tensor):
                                score = score.item()
                            scores.append(float(score))
                        
                        # Store processed results (without raw outputs for efficiency)
                        results["dataset"][dataset_name]["model_unit"][str(model_units_list)][target_variable_str] = {}
                        results["dataset"][dataset_name]["model_unit"][str(model_units_list)][target_variable_str]["scores"] = scores
                        results["dataset"][dataset_name]["model_unit"][str(model_units_list)][target_variable_str]["average_score"] = np.mean(scores)
                        # Save processed results to directory if provided


                # Note: raw_outputs are now kept in results as CPU tensors for caller to use
                # They will be excluded from saved JSON files automatically

        progress_bar.close()

        if save_dir is not None:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Create a copy of results for saving to avoid modifying the original
            results_for_saving = copy.deepcopy(results)

            # Process outputs for saving: convert raw tensors to JSON-serializable format
            k = self.config.get("save_top_k_logits", 0)
            for dataset_name in results_for_saving["dataset"].keys():
                # Convert actual outputs (no intervention) to serializable format
                if "raw_outputs_no_intervention" in results_for_saving["dataset"][dataset_name]:
                    actual_outputs = results_for_saving["dataset"][dataset_name]["raw_outputs_no_intervention"]
                    serializable_actual_outputs = []
                    for batch_dict in actual_outputs:
                        serializable_batch_dict = {}
                        # Convert sequences tensor to list
                        if "sequences" in batch_dict:
                            serializable_batch_dict["sequences"] = batch_dict["sequences"].detach().cpu().tolist()
                        # Convert scores tensors to lists if present
                        if "scores" in batch_dict and batch_dict["scores"]:
                            serializable_batch_dict["scores"] = [score.detach().cpu().tolist() for score in batch_dict["scores"]]
                        # Keep string field as-is (already serializable)
                        if "string" in batch_dict:
                            serializable_batch_dict["string"] = batch_dict["string"]
                        serializable_actual_outputs.append(serializable_batch_dict)
                    results_for_saving["dataset"][dataset_name]["raw_outputs_no_intervention"] = serializable_actual_outputs

                for model_unit in results_for_saving["dataset"][dataset_name]["model_unit"].values():
                    # Convert raw_outputs to top-k logits if requested
                    if "raw_outputs" in model_unit and k and k > 0:
                        serializable_outputs = []
                        for batch_dict in model_unit["raw_outputs"]:
                            # Handle both batch dicts and individual example dicts
                            num_examples = batch_dict["sequences"].shape[0]

                            for idx in range(num_examples):
                                example_dict = {
                                    "sequences": batch_dict["sequences"][idx].tolist(),
                                    "string": self.pipeline.dump(batch_dict["sequences"][idx:idx+1])
                                }

                                # Add top-k logits if scores are present
                                if "scores" in batch_dict:
                                    example_output = {
                                        "scores": [score[idx:idx+1] for score in batch_dict["scores"]]
                                    }
                                    top_k = _extract_top_k_logits(example_output, k, self.pipeline.tokenizer)
                                    if top_k:
                                        example_dict["top_k_logits"] = top_k

                                serializable_outputs.append(example_dict)

                        model_unit["outputs"] = serializable_outputs

                    # Remove raw tensor outputs from the copy only
                    if "raw_outputs" in model_unit:
                        del model_unit["raw_outputs"]

                    # Remove per-target-variable scores from the copy only (these are intermediate computation results)
                    for target_variable in model_unit.keys():
                        if model_unit[target_variable] is not None and isinstance(model_unit[target_variable], dict) and "scores" in model_unit[target_variable]:
                            del model_unit[target_variable]["scores"]

            # Generate meaningful filename based on experiment parameters
            file_name = "results.json"
            total_target_str = ""
            if target_variables_list is not None:
                for target_variables in target_variables_list:
                    target_variable_str = "-".join(target_variables)
                    total_target_str += target_variable_str + "_"
            file_name =  total_target_str + "_" + file_name
            for k in ["method_name", "model_name", "task_name"]:
                file_name = results[k] + "_" + file_name
            with open(os.path.join(save_dir, file_name), "w") as f:
                json.dump(results_for_saving, f, indent=2)
                

        return results

    def _move_outputs_to_cpu(self, outputs):
        """
        Move all tensors in outputs to CPU and detach from computation graph.
        Keeps tensors as tensors (doesn't convert to lists).

        Args:
            outputs: List of output dictionaries or single dictionary

        Returns:
            Same structure with all tensors moved to CPU and detached
        """
        def move_dict_to_cpu(d):
            """Helper to move a single dictionary's tensors to CPU."""
            result = {}
            for key, value in d.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.detach().cpu()
                elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                    result[key] = [v.detach().cpu() for v in value]
                else:
                    result[key] = value
            return result

        if isinstance(outputs, list):
            return [move_dict_to_cpu(d) for d in outputs]
        else:
            return move_dict_to_cpu(outputs)

    def _compute_actual_outputs(self, dataset: CounterfactualDataset, verbose: bool = False):
        """
        Compute outputs from the model without any interventions (actual/baseline outputs).

        This processes only the base inputs from the dataset through the pipeline
        to capture the model's natural behavior, following the same approach as FilterExperiment.

        Args:
            dataset: CounterfactualDataset containing base inputs
            verbose: Whether to show progress

        Returns:
            List of output dictionaries with same structure as intervention outputs
        """

        # Extract only base inputs (not counterfactuals)
        # Each example in dataset.dataset has structure: {"input": {...}, "counterfactual_inputs": [...]}
        # We want the "input" part which contains the raw_input
        base_inputs = [example["input"] for example in dataset.dataset]
        base_dataset = Dataset.from_list(base_inputs)

        # Create dataloader for batch processing
        dataloader = DataLoader(
            base_dataset,
            batch_size=self.config["evaluation_batch_size"],
            shuffle=False,  # Maintain order for correspondence
            collate_fn=shallow_collate_fn
        )

        all_outputs = []

        # Process each batch
        for batch in tqdm(dataloader, desc="Computing actual outputs", disable=not verbose, leave=False):
            with torch.no_grad():
                # The shallow_collate_fn returns {"raw_input": [input1, input2, ...], ...}
                # Pipeline.generate expects a list of dictionaries, so we need to reconstruct them
                batch_inputs = []
                batch_size = len(batch["raw_input"])
                for i in range(batch_size):
                    example_dict = {key: batch[key][i] for key in batch.keys()}
                    batch_inputs.append(example_dict)

                # Generate outputs without intervention - pipeline.generate handles loading internally
                output_dict = self.pipeline.generate(
                    batch_inputs,
                    output_scores=self.config["output_scores"]
                )

                # Add string field for consistency with intervention outputs
                output_dict["string"] = self.pipeline.dump(output_dict["sequences"])

                # Keep the same tensor format as raw_outputs for consistency with custom scoring functions
                # Only convert to serializable format when saving (handled in main method)
                all_outputs.append(output_dict)

        return all_outputs

    def save_featurizers(self, model_units, model_dir):
        """
        Save featurizers and feature indices for model units to disk.
        
        Args:
            model_units: List of model units whose featurizers should be saved
            model_dir: Directory to save the featurizers to
            
        Returns:
            Tuple of paths to the saved featurizer, inverse featurizer, and indices files
        """
        if model_units is None or len(model_units) == 0:
            model_units = [model_unit for model_units_list in self.model_units_lists for model_units in model_units_list for model_unit in model_units]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        f_dirs, invf_dirs, indices_dir = [], [], []
        
        for model_unit in model_units:
            # Create a filename based on the model unit's ID
            filename = os.path.join(model_dir, model_unit.id)
            
            # Save the featurizer and inverse featurizer modules
            f_dir, invf_dir = model_unit.featurizer.save_modules(filename)
            
            # Save the feature indices separately
            with open(filename + "_indices", "w") as f:
                indices = model_unit.get_feature_indices()
                if indices is not None:
                    json.dump([int(i) for i in indices], f)
                else:
                    json.dump(None, f)
            
            # Collect paths for return
            f_dirs.append(f_dir)
            invf_dirs.append(invf_dir)
            indices_dir.append(filename + "_indices")
            
        return f_dirs, invf_dirs, indices_dir

    def load_featurizers(self, model_dir):
        """
        Load saved featurizers and feature indices for model units from disk.
        
        Args:
            model_dir: Directory containing the saved featurizers
        """
        for model_units_list in self.model_units_lists:
            for model_units in model_units_list:
                for model_unit in model_units:
                    # Construct the filename based on the model unit's component ID
                    filename = os.path.join(model_dir, model_unit.id)
                    
                    # Load the featurizer and inverse featurizer if they exist
                    if os.path.exists(filename + "_featurizer") and os.path.exists(filename + "_inverse_featurizer"):
                        model_unit.set_featurizer(Featurizer.load_modules(filename))
                    
                    # Load the feature indices if they exist
                    if os.path.exists(filename + "_indices"):
                        indices = None
                        with open(filename + "_indices", "r") as f:
                            indices = json.load(f)
                        model_unit.set_feature_indices(indices)
        return

    def build_SVD_feature_interventions(self, datasets, n_components=None, verbose=False, collect_counterfactuals=True, PCA=False, algorithm="randomized", flatten=True):
        """
        Build feature interventions using SVD/PCA on collected activations.
        
        This method extracts activations from the model at the specified model units,
        performs dimensionality reduction via SVD or PCA, and sets up the model units
        to use these reduced-dimension representations.
        
        Args:
            datasets: Dictionary of datasets to collect activations from
            n_components: Number of SVD/PCA components to use (defaults to max possible)
            verbose: Whether to show progress and component information
            collect_counterfactuals: Whether to include counterfactual inputs in feature extraction
            PCA: Whether to normalize features before SVD (making it equivalent to PCA)
            algorithm: SVD algorithm to use ("arpack" is memory-efficient for large matrices)
        
        Returns:
            List of rotation matrices (featurizers) created for each model unit
        """
        # Flatten the dataset dictionary into a single list
        counterfactual_dataset = []
        for dataset in datasets.values():
            counterfactual_dataset += dataset
        
        
        #  To understand this, let's break down the structure:

        #   1. self.model_units_lists is a triple-nested list with structure:
        #     - Outermost: Different intervention experiments
        #     - Middle: Groups of units sharing the same counterfactual input
        #     - Innermost: Individual model units to intervene on
        #   2. zip(*self.model_units_lists) transposes the outermost dimension,
        #      grouping together units from the same position across all experiments.
        #   3. chain.from_iterable flattens the middle and inner dimensions into a single list.

        #   Result: zipped_model_units is a list where each element contains all model units that share 
        #   the same counterfactual input position across all experiments, flattened into a single list.

        #   For example, if self.model_units_lists has shape [2, 3, 4] (2 experiments, 3 counterfactual groups, 4 units each), 
        #   then zipped_model_units would have shape [3, 8] (3 counterfactual groups, 8 units total from both experiments).

        zipped_model_units = [list(chain.from_iterable(model_units_list)) 
                              for model_units_list in zip(*self.model_units_lists)]

        # The features variable returned by _collect_features has the following structure:
        # 1. Outer dimension: Corresponds to the groups in zipped_model_units (same as middle dimension of self.model_units_lists)
        # 2. Middle dimension: Corresponds to individual model units within each group
        # 3. Inner structure: Each element is a PyTorch tensor with shape (n_samples, n_features)
        features = _collect_features(
            counterfactual_dataset,
            self.pipeline,
            zipped_model_units,
            self.config,
            collect_counterfactuals=collect_counterfactuals,
            verbose=verbose
        )

        # Restructure features to match the original model_units_lists structure
        # This is necessary because _collect_features returns a flat list of features
        # where each element corresponds to a model unit in zipped_model_units.
        # We need to map these back to the original nested structure of model_units_lists.
        # 1. Outer dimension: Different intervention experiments
        # 2. Middle dimension: Groups of model units sharing the same counterfactual input
        # 3. Inner dimension: Individual model units within each group
        restructured_features = []
        for i, model_units_list in enumerate(self.model_units_lists):
            experiment_features = []
            for j, model_units in enumerate(model_units_list):
                start = sum(len(self.model_units_lists[k][j]) for k in range(i))
                end = start + len(model_units)
                experiment_features.append(features[j][start:end])
            restructured_features.append(experiment_features)
        features = restructured_features

        
                    
        for i, model_units_list in enumerate(self.model_units_lists):
            for j, model_units in enumerate(model_units_list):
                for k, model_unit in enumerate(model_units):
                    X = features[i][j][k]
                    # Calculate maximum possible components (min of sample count and feature dimension, minus 1)
                    n = min(X.shape[0], X.shape[1]) - 1
                    n = min(n, n_components) if n_components is not None else n
                    
                    # Normalize input features if using PCA
                    if PCA:
                        pca_mean = X.mean(axis=0, keepdim=True)
                        pca_std = X.var(axis=0)**0.5
                        epsilon = 1e-6  # Prevent division by zero
                        pca_std = np.where(pca_std < epsilon, epsilon, pca_std)
                        X = (X - pca_mean) / pca_std
                    
                    # Perform SVD/PCA
                    svd = TruncatedSVD(n_components=n, algorithm=algorithm)
                    svd.fit(X)
                    components = svd.components_.copy()
                    rotation = torch.tensor(components).to(X.dtype)
                    
                    if verbose:
                        print(f'SVD explained variance: {[round(float(x),2) for x in svd.explained_variance_ratio_]}')

                    model_unit.set_featurizer(
                        SubspaceFeaturizer(
                            rotation_subspace=rotation.T,
                            trainable=False,
                            id="SVD",
                            **self.config.get('featurizer_kwargs', {})
                        )
                    )
                    model_unit.set_feature_indices(None)  # Use all components initially
                

    def train_interventions(self, datasets, target_variables, method="DAS", model_dir=None, verbose=False, checker=None):
        """
        Train interventions to identify neural representations of causal variables.
        
        This method trains intervention parameters to locate where and how the target
        causal variables are represented in the neural network. The training respects
        the nested structure of model_units_lists:
        
        - For each experiment configuration (outer list)
        - For each group sharing counterfactual inputs (middle list)
            - For each model unit in the group (inner list)
            - Configure the appropriate featurizer based on method
        
        The training process learns how to intervene on these units to align
        with the expected behavior from the causal model.
        
        Supports two primary methods:
        - DAS (Distributed Alignment Search): Learns orthogonal directions representing variables
        - DBM (Desiderata-Based Masking): Learns binary masks over features 
        
        Args:
            datasets: Dictionary of counterfactual datasets for training
            target_variables: Causal variables to locate in the neural network
            method: Either "DAS" or "DBM"
            model_dir: Directory to save trained models (optional)
            verbose: Whether to show training progress
            checker: Optional function for metric evaluation during training.
                    If None, uses exact token matching. Should match signature of self.checker.

        Returns:
            Self (for method chaining)
        """
        # Label the datasets with the target variables
        if isinstance(datasets, CounterfactualDataset):
            datasets = {datasets.id: datasets}
        counterfactual_dataset = []
        for dataset in datasets.values():
            counterfactual_dataset += self.causal_model.label_counterfactual_data(dataset, target_variables)

        # Validate that required training parameters are present
        required_params = ["training_epoch", "init_lr"]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required training parameter '{param}' not found in config")

        # Validate method-specific parameters
        if method == "DAS" and "DAS" not in self.config:
            raise ValueError("DAS config not found in config")
        if method == "DBM" and "masking" not in self.config:
            raise ValueError("masking config not found in config") 

        # Validate method
        assert method in ["DAS", "DBM"]

        # Set intervention type based on method
        if method == "DAS":
            intervention_type = "interchange"
        elif method == "DBM":
            intervention_type = "mask"

        # Configure and train featurizers for each model unit
        for model_units_list in self.model_units_lists:
            for model_units in model_units_list:
                for model_unit in model_units:
                    if method == "DAS":
                        # For DAS, use trainable subspace featurizer
                        model_unit.set_featurizer(
                            SubspaceFeaturizer(
                                shape=(model_unit.shape[0], self.config["DAS"]["n_features"]),
                                trainable=True,
                                id="DAS"
                            )
                        )
                        model_unit.set_feature_indices(None)  # Use all features

            # Create a wrapper for the loss function that includes the checker
            def loss_and_metric_fn_with_checker(pipeline, intervenable_model, batch, model_units_list):
                # Use passed checker, or fall back to experiment's checker, or default
                effective_checker = checker if checker is not None else getattr(self, 'checker', None)
                return self.loss_and_metric_fn(
                    pipeline, intervenable_model, batch, model_units_list,
                    checker=effective_checker
                )

            # Train the intervention
            printout = _train_intervention(self.pipeline, model_units_list, counterfactual_dataset,
                               intervention_type, self.config, loss_and_metric_fn_with_checker)
            if verbose:
                print(printout)
            
            # Save trained models if directory provided
            if model_dir is not None:
                self.save_featurizers([model_unit for model_units in model_units_list for model_unit in model_units], model_dir) 
                
        return self