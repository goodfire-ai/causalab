import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Callable, Tuple, Union
import gc, json, os, collections, random
from itertools import chain

import pyvene as pv
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD 
from tqdm import tqdm, trange
from datasets import Dataset

from neural.model_units import *
from neural.pipeline import Pipeline
from causal.causal_model import CausalModel
from causal.counterfactual_dataset import CounterfactualDataset
from experiments.pyvene_core import _run_interchange_interventions, _train_intervention, _collect_features
from experiments.config import DEFAULT_CONFIG

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
        
        # Ensure evaluation_batch_size defaults to batch_size if not set
        if self.config.get("evaluation_batch_size") is None:
            self.config["evaluation_batch_size"] = self.config["batch_size"]

    def perform_interventions(self, datasets, verbose: bool = False, target_variables_list: List[List[str]] = None, save_dir=None) -> Dict:
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

        Returns:
            Dictionary containing the experiment results with scores for each model unit and dataset
            
        Note:
            The structure of model_units_lists determines how interventions are performed:
            - Each group in the middle level shares a counterfactual input
            - Each unit in the innermost level is intervened upon using that shared input
            - This allows for complex interventions where multiple components are modified simultaneously
        """
        # Initialize results structure
        results = {"method_name": self.config["method_name"],
                    "model_name": self.pipeline.model.__class__.__name__,
                    "task_name": self.causal_model.id,
                    "dataset": {dataset_name: {
                        "model_unit": {str(units_list): None for units_list in self.model_units_lists}}
                    for dataset_name in datasets.keys()}}
        
        # Process each dataset and model unit combination
        for dataset_name in datasets.keys():
            for model_units_list in self.model_units_lists:
                # Run interventions
                if verbose:
                    print(f"Running interventions for {dataset_name} with model units {model_units_list}")
                
                # Execute interchange interventions using pyvene
                raw_outputs = _run_interchange_interventions(
                    pipeline=self.pipeline,
                    counterfactual_dataset=datasets[dataset_name],
                    model_units_list=model_units_list,
                    verbose=verbose,
                    output_scores=self.config["output_scores"],
                    batch_size=self.config["evaluation_batch_size"]
                )
                
                # Extract metadata for this model unit
                metadata = self.metadata_fn(model_units_list)
                results["dataset"][dataset_name]["model_unit"][str(model_units_list)] = {
                    "raw_outputs": raw_outputs,
                    "metadata": metadata}

                # Process and decode model outputs
                dumped_outputs = []
                for raw_output in raw_outputs:
                    dump_result = self.pipeline.dump(raw_output, is_logits=self.config["output_scores"])
                    if isinstance(dump_result, list):
                        dumped_outputs.extend(dump_result)
                    else:
                        dumped_outputs.append(dump_result)

                # Flatten the nested raw_outputs for processing
                raw_outputs = [item for sublist in raw_outputs for item in sublist]
                
                # Evaluate results for each target variable group
                for target_variables in target_variables_list:
                    target_variable_str = "-".join(target_variables)
                    
                    # Generate expected outputs from causal model
                    labeled_data = self.causal_model.label_counterfactual_data(datasets[dataset_name], target_variables)
                    assert len(labeled_data) == len(dumped_outputs), f"Length mismatch: {len(labeled_data)} vs {len(dumped_outputs)}"
                    assert len(labeled_data) == len(raw_outputs), f"Length mismatch: {len(labeled_data)} vs {len(raw_outputs)}"
                    
                    # Compute intervention scores
                    scores = []
                    for example, output, raw_output in zip(labeled_data, dumped_outputs, raw_outputs):
                        if self.config["check_raw"]:
                            score = self.checker(raw_output, example["setting"])
                        else:
                            score = self.checker(output, example["label"])
                        if isinstance(score, torch.Tensor):
                            score = score.item()
                        scores.append(float(score))
                    
                    # Store processed results (without raw outputs for efficiency)
                    results["dataset"][dataset_name]["model_unit"][str(model_units_list)][target_variable_str] = {}
                    results["dataset"][dataset_name]["model_unit"][str(model_units_list)][target_variable_str]["scores"] = scores
                    results["dataset"][dataset_name]["model_unit"][str(model_units_list)][target_variable_str]["average_score"] = np.mean(scores)
                    # Save processed results to directory if provided


                # Remove raw_outputs to save memory in the results dictionary
                if not self.config["raw_outputs"]:
                    del results["dataset"][dataset_name]["model_unit"][str(model_units_list)]["raw_outputs"]

        if save_dir is not None:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # remove scores and raw_outputs from results to save space
            for dataset_name in results["dataset"].keys():
                for model_unit in results["dataset"][dataset_name]["model_unit"].values():
                    if "raw_outputs" in model_unit:
                        del model_unit["raw_outputs"]
                    for target_variable in model_unit.keys():
                        if model_unit[target_variable] is not None and "scores" in model_unit[target_variable]:
                            del model_unit[target_variable]["scores"]
            # Generate meaningful filename based on experiment parameters
            file_name = "results.json"
            total_target_str = ""
            for target_variables in target_variables_list:
                target_variable_str = "-".join(target_variables)
                total_target_str += target_variable_str + "_"
            file_name =  total_target_str + "_" + file_name
            for k in ["method_name", "model_name", "task_name"]:
                file_name = results[k] + "_" + file_name
            with open(os.path.join(save_dir, file_name), "w") as f:
                json.dump(results, f, indent=2)
                

        return results

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
                

    def train_interventions(self, datasets, target_variables, method="DAS", model_dir=None, verbose=False):
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
                
        Returns:
            Self (for method chaining)
        """
        # Label the datasets with the target variables
        counterfactual_dataset = []
        for dataset in datasets.values():
            counterfactual_dataset += self.causal_model.label_counterfactual_data(dataset, target_variables)

        # The config now comes from DEFAULT_CONFIG, so we don't need to set defaults here
        # Just validate that required training parameters are present
        required_params = ["training_epoch", "init_lr", "n_features"]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required training parameter '{param}' not found in config") 

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
                                shape=(model_unit.shape[0], self.config["n_features"]), 
                                trainable=True,
                                id="DAS"
                            )
                        )
                        model_unit.set_feature_indices(None)  # Use all features

            # Train the intervention
            _train_intervention(self.pipeline, model_units_list, counterfactual_dataset, 
                               intervention_type, self.config, self.loss_and_metric_fn)
            
            # Save trained models if directory provided
            if model_dir is not None:
                self.save_featurizers([model_unit for model_units in model_units_list for model_unit in model_units], model_dir) 
                
        return self