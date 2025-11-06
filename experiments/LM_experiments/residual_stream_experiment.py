import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Callable, Tuple, Optional, Union, Any
import os
import gc
import torch
import logging
from collections import Counter

from .LM_utils import LM_loss_and_metric_fn
from experiments.visualizations import (
    create_heatmap,
    create_text_output_grid,
    print_text_heatmap,
    print_das_mask_grid,
    create_das_mask_grid_plot
)
from experiments.intervention_experiment import *
from causal.causal_model import CausalModel
from causal.causal_utils import compute_interchange_scores
from neural.LM_units import *
from neural.model_units import *
from neural.featurizers import *
from neural.pipeline import LMPipeline

class PatchResidualStream(InterventionExperiment):
    """
    Experiment for analyzing residual stream interventions in language models.
    
    The residual stream is a fundamental concept in transformer architectures:
    - It represents the hidden representation that flows through the network
    - Each transformer layer adds its computation results to this stream
    - At any given layer L, the residual stream contains the sum of:
      * The original token embeddings
      * The outputs of all previous layers 0 to L-1
    
    This class enables interventions directly on the residual stream at specific points:
    - Layer index: Which transformer layer to target (0 to num_layers-1)
    - Token position: Which token in the sequence to modify
    
    By modifying the residual stream at strategic points and observing the effect on model outputs,
    we can identify where specific information is represented and how it's processed through
    the network. This approach is central to mechanistic interpretability, which aims to
    reverse-engineer the algorithms implemented by neural networks.
    
    Attributes:
        featurizers (Dict): Mapping of (layer, position) tuples to Featurizer instances
        loss_and_metric_fn (Callable): Function to compute loss and metrics
        layers (List[int]): Layer indices to analyze
        token_positions (List[TokenPosition]): Token positions to analyze
    """

    def __init__(self,
                 pipeline: LMPipeline,
                 layers: List[int],
                 token_positions: List[TokenPosition],
                 featurizers: Dict[Tuple[int, str], Featurizer] = None,
                 loss_and_metric_fn: Callable = LM_loss_and_metric_fn,
                 **kwargs):
        """
        Initialize ResidualStreamExperiment for analyzing residual stream interventions.

        Args:
            pipeline: LMPipeline object for model execution
            layers: List of layer indices to analyze
            token_positions: List of ComponentIndexers for token positions
            featurizers: Dict mapping (layer, position.id) to Featurizer instances
            **kwargs: Additional configuration options
        """
        self.featurizers = featurizers if featurizers is not None else {}
        self.loss_and_metric_fn = loss_and_metric_fn

        # Extract featurizer_kwargs from config if present
        config = kwargs.get('config', {})
        featurizer_kwargs = config.get('featurizer_kwargs', {})

        # Generate all combinations of model units without feature_indices
        model_units_lists = []
        for layer in layers:
            for pos_list in token_positions:
                if isinstance(pos_list, TokenPosition):
                    pos_list = [pos_list]
                block = []
                for pos in pos_list:
                    featurizer = self.featurizers.get((layer, pos.id),
                                                    Featurizer(n_features=pipeline.model.config.hidden_size,
                                                            **featurizer_kwargs))
                    target_output = True
                    actual_layer = layer
                    if layer == -1:
                        actual_layer = 0
                        target_output = False
                    block.append(
                        ResidualStream(
                            layer=actual_layer,
                            token_indices=pos,
                            featurizer=featurizer,
                            shape=(pipeline.model.config.hidden_size,),
                            feature_indices=None,
                            target_output=target_output
                        )
                    )
                model_units_lists.append([block])

        metadata_fn = lambda x: {
            "layer": -1 if (x[0][0].component.component_type == "block_input" and
                            x[0][0].component.get_layer() == 0)
                      else x[0][0].component.get_layer(),
            "position": x[0][0].component.get_index_id()
        }

        super().__init__(
            pipeline=pipeline,
            model_units_lists=model_units_lists,
            metadata_fn=metadata_fn,
            **kwargs
        )
        
        self.layers = layers
        self.token_positions = token_positions
        self._token_positions_sorted = False  # Track if we've sorted yet

    def perform_interventions(self, datasets, verbose: bool = False, save_dir=None, include_actual_outputs: bool = False):
        """
        Override to sort token positions based on first input before running interventions.
        """
        # Sort token positions using the first input from the first dataset
        if not self._token_positions_sorted:
            if datasets:
                # Get first dataset
                first_dataset = next(iter(datasets.values())) if isinstance(datasets, dict) else datasets
                if len(first_dataset) > 0:
                    # Get first example's input
                    first_input = first_dataset[0]["input"]
                    # Sort token positions
                    self.token_positions = self._sort_token_positions_by_first_input(self.token_positions, first_input)
                    self._token_positions_sorted = True

        # Call parent implementation
        return super().perform_interventions(datasets, verbose, save_dir, include_actual_outputs)

    def _sort_token_positions_by_first_input(self, token_positions: List[TokenPosition], sample_input: Dict) -> List[TokenPosition]:
        """
        Sort token positions by their actual indices in a sample input.
        """
        if not token_positions:
            return token_positions

        # For each token position, get its first index
        position_indices = {}
        for i, token_pos in enumerate(token_positions):
            try:
                indices = token_pos.index(sample_input)
                # Use the first index for sorting
                first_idx = indices[0] if isinstance(indices, list) else indices
                position_indices[i] = first_idx
            except Exception:
                # If we can't determine the index, put it at the end
                position_indices[i] = float('inf')

        # Sort token positions by their indices
        sorted_indices = sorted(range(len(token_positions)), key=lambda i: position_indices[i])
        return [token_positions[i] for i in sorted_indices]

    def build_SAE_feature_intervention(self, sae_loader: Callable[[int], Any]) -> None:
        """
        Apply Sparse Autoencoder (SAE) features to model units.
        
        This method takes a function that loads SAEs for specific layers and 
        applies them to the appropriate model units. It handles memory cleanup 
        between loading SAEs for different layers to prevent OOM errors.
        
        Args:
            sae_loader: A function that takes a layer index and returns an SAE instance.
                For example:
                ```python
                def sae_loader(layer):
                    sae, _, _ = SAE.from_pretrained(
                        release = "gemma-scope-2b-pt-res-canonical",
                        sae_id = f"layer_{layer}/width_16k/canonical",
                        device = "cpu",
                    )
                    return sae
                ```
        
        Raises:
            RuntimeError: If SAE loading fails for a specific layer
        """
        try:
            # Process each model units list
            for model_units_list in self.model_units_lists:
                for model_units in model_units_list:
                    for unit in model_units:
                        layer = unit.component.get_layer()
                        
                        try:
                            # Load SAE for the specific layer
                            sae = sae_loader(layer)
                            
                            # Set the SAE featurizer for this unit
                            unit.set_featurizer(SAEFeaturizer(sae, **self.config.get('featurizer_kwargs', {})))
                            
                            # Clear GPU memory after loading each SAE
                            del sae
                            self._clean_memory()
                            
                        except Exception as e:
                            # Continue with next unit rather than failing the entire experiment
                            continue
                            
            
        except Exception as e:
            raise RuntimeError(f"Failed to apply SAE features: {str(e)}")

    def _clean_memory(self):
        """
        Clean up memory to prevent OOM errors.
        
        This method performs garbage collection and clears CUDA cache
        to ensure memory is available for subsequent operations.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def plot_heatmaps(self, results: Dict, target_variables, save_path: str = None, average_counterfactuals: bool = False):
        """
        Generate heatmaps visualizing intervention scores across layers and positions.

        Args:
            results: Dictionary containing experiment results from interpret_results()
            target_variables: List of variable names being analyzed
            save_path: Optional path to save the generated plots. If None, displays plots interactively.
            average_counterfactuals: If True, averages scores across counterfactual datasets
        """
        target_variables_str = "-".join(target_variables)

        token_ids = [token_pos.id for token_pos in self.token_positions]
        layers = list(reversed(self.layers))


        if average_counterfactuals:
            self._plot_average_heatmap(results, layers, token_ids, target_variables_str, save_path)
        else:
            self._plot_individual_heatmaps(results, layers, token_ids, target_variables_str, save_path)

    @staticmethod
    def plot_heatmaps_from_results(results: Dict, target_variables: List[str],
                                     layers: List[int], token_position_ids: List[str],
                                     save_path: str = None, average_counterfactuals: bool = False):
        """
        Generate heatmaps without requiring an experiment object or pipeline.

        This static method allows visualization of results without loading the full model,
        which is much faster when you just want to create heatmaps from saved results.

        Args:
            results: Dictionary containing experiment results with scores
            target_variables: List of variable names being analyzed
            layers: List of layer indices used in the experiment
            token_position_ids: List of token position IDs (strings)
            save_path: Optional path to save the generated plots. If None, displays plots interactively.
            average_counterfactuals: If True, averages scores across counterfactual datasets

        Example:
            >>> # No need to load model!
            >>> PatchResidualStream.plot_heatmaps_from_results(
            ...     results=scored_results,
            ...     target_variables=["digit_0_0"],
            ...     layers=[0, 1, 2, 3],
            ...     token_position_ids=["number_0_tok0", "last_token"],
            ...     save_path="output/heatmap_digit_0_0"
            ... )
        """
        target_variables_str = "-".join(target_variables)
        reversed_layers = list(reversed(layers))

        # Create a temporary instance just to access the helper methods
        # This avoids code duplication while not requiring pipeline
        temp_instance = object.__new__(PatchResidualStream)

        if average_counterfactuals:
            temp_instance._plot_average_heatmap(results, reversed_layers, token_position_ids, target_variables_str, save_path)
        else:
            temp_instance._plot_individual_heatmaps(results, reversed_layers, token_position_ids, target_variables_str, save_path)

    def _build_score_matrix(self,
                            results: Dict,
                            layers: List,
                            positions: List,
                            target_variables_str: str,
                            dataset_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract score matrices from results for specified datasets.

        Args:
            results: Dictionary containing experiment results
            layers: List of layer indices
            positions: List of position IDs
            target_variables_str: String identifier for target variables
            dataset_names: List of dataset names to process. If None, processes all datasets.

        Returns:
            Dictionary mapping dataset names to their score matrices
        """
        if dataset_names is None:
            dataset_names = list(results["dataset"].keys())

        matrices = {}

        for dataset_name in dataset_names:
            score_matrix = np.zeros((len(layers), len(positions)))
            valid_entries = False

            # Fill score matrix for this dataset
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                        if "metadata" in unit_data and target_variables_str in unit_data:
                            if "average_score" in unit_data[target_variables_str]:
                                metadata = unit_data["metadata"]
                                if metadata.get("layer") == layer and metadata.get("position") == pos:
                                    score_matrix[i, j] = unit_data[target_variables_str]["average_score"]
                                    valid_entries = True

            # Only include datasets with valid entries
            if valid_entries:
                matrices[dataset_name] = score_matrix

        return matrices

    def _aggregate_matrices(self,
                           matrices: Dict[str, np.ndarray],
                           aggregation: str = "average") -> np.ndarray:
        """
        Aggregate multiple score matrices using specified method.

        Args:
            matrices: Dictionary mapping dataset names to score matrices
            aggregation: Aggregation method - "average", "sum", "max", or "min"

        Returns:
            Aggregated score matrix

        Raises:
            ValueError: If aggregation method is not supported or no matrices provided
        """
        if not matrices:
            raise ValueError("No valid matrices to aggregate")

        matrix_array = np.stack(list(matrices.values()))

        if aggregation == "average":
            return np.mean(matrix_array, axis=0)
        elif aggregation == "sum":
            return np.sum(matrix_array, axis=0)
        elif aggregation == "max":
            return np.max(matrix_array, axis=0)
        elif aggregation == "min":
            return np.min(matrix_array, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

    def _plot_average_heatmap(self, results: Dict, layers: List, positions: List,
                             target_variables_str: str, save_path: Optional[str] = None):
        """Create and save/display an averaged heatmap across all datasets."""
        # Build score matrices for all datasets
        matrices = self._build_score_matrix(results, layers, positions, target_variables_str)

        if not matrices:
            return

        # Aggregate matrices
        score_matrix = self._aggregate_matrices(matrices, aggregation="average")

        # Use the last dataset name for filename (kept for backwards compatibility)
        dataset_name = list(matrices.keys())[-1]
        safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')

        # Create the heatmap
        self._create_heatmap(
            score_matrix=score_matrix,
            layers=layers,
            positions=positions,
            title=f'Intervention Accuracy - Dataset: {dataset_name}\nExperiment: {results["experiment_id"]}\nIntervened Variables: {target_variables_str}',
            save_path=os.path.join(save_path, f'heatmap_dataset_{safe_dataset_name}_{results["experiment_id"]}_variables_{target_variables_str}.png') if save_path else None
        )
    
    def _plot_individual_heatmaps(self, results: Dict, layers: List, positions: List,
                                 target_variables_str: str, save_path: Optional[str] = None):
        """Create and save/display individual heatmaps for each dataset."""
        # Build score matrices for all datasets
        matrices = self._build_score_matrix(results, layers, positions, target_variables_str)

        # Create individual heatmaps for each dataset
        for dataset_name, score_matrix in matrices.items():
            # Convert dataset name to a safe filename
            safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')

            # Create the heatmap
            self._create_heatmap(
                score_matrix=score_matrix,
                layers=layers,
                positions=positions,
                title=f'Intervention Accuracy - Dataset: {dataset_name}\nExperiment: {results["experiment_id"]}\nIntervened Variables: {target_variables_str}',
                save_path=os.path.join(save_path, f'heatmap_dataset_{safe_dataset_name}_{results["experiment_id"]}_variables_{target_variables_str}.png') if save_path else None
            )
    
    def _create_heatmap(self, score_matrix: np.ndarray, layers: List, positions: List,
                       title: str, save_path: Optional[str] = None):
        """
        Create and save/display a single heatmap.

        Args:
            score_matrix: 2D numpy array with scores for each (layer, position) pair
            layers: List of layer indices
            positions: List of position names
            title: Title for the heatmap
            save_path: Path to save the heatmap, or None to display it
        """
        # Use the consolidated visualization function
        create_heatmap(
            score_matrix=score_matrix,
            x_labels=positions,
            y_labels=layers,
            title=title,
            save_path=save_path,
            x_label='Position',
            y_label='Layer',
            use_custom_bounds=False,
            cbar_label='Accuracy (%)',
            figsize=(10, 6)
        )

    def print_text_analysis(self, results: Dict, target_variables: List[str],
                           save_path: Optional[str] = None) -> None:
        """
        Print a detailed text-based analysis of intervention accuracy scores using region-based breakdown.

        This method identifies distinct regions in the score matrix (groups of cells with similar scores)
        and reports statistics for each region, capturing both high AND low accuracy patterns.

        Args:
            results: Results dictionary from perform_interventions
            target_variables: List of variable names being analyzed
            save_path: Optional path to save the text output to a file
        """
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("ACTIVATION PATCHING TEXT ANALYSIS")
        output_lines.append("=" * 80)

        token_ids = [token_pos.id for token_pos in self.token_positions]
        layers = self.layers

        # Analyze each variable
        for target_vars in target_variables:
            if not isinstance(target_vars, list):
                target_vars = [target_vars]

            target_variables_str = "-".join(target_vars)

            output_lines.append(f"\n{'=' * 80}")
            output_lines.append(f"Variable: {target_variables_str}")
            output_lines.append(f"{'=' * 80}")

            # Build score matrices for all datasets
            matrices = self._build_score_matrix(results, layers, token_ids, target_variables_str)

            if not matrices:
                output_lines.append("  No data available for this variable")
                continue

            # Analyze each dataset
            for dataset_name, score_matrix in matrices.items():
                output_lines.append(f"\n{'-' * 80}")
                output_lines.append(f"Dataset: {dataset_name}")
                output_lines.append(f"{'-' * 80}")

                # Overall statistics
                mean_score = score_matrix.mean()
                std_score = score_matrix.std()
                min_score = score_matrix.min()
                max_score = score_matrix.max()

                output_lines.append(f"\nOverall: mean={mean_score:.1%} ± {std_score:.1%}, range=[{min_score:.1%}, {max_score:.1%}]")

                if max_score == 0:
                    output_lines.append("No localization found (all scores = 0)")
                    continue

                # Identify regions using clustering on score values
                output_lines.append(f"\nRegion Breakdown (using natural clustering):")

                # Get unique score values and cluster them
                flat_scores = score_matrix.flatten()
                unique_scores = np.unique(flat_scores)

                # Use simple thresholding to identify clusters:
                # Group scores that are within 10% of each other
                clusters = []
                for score in sorted(unique_scores):
                    # Check if this score belongs to an existing cluster
                    added = False
                    for cluster in clusters:
                        if abs(score - np.mean(cluster)) <= 0.10:
                            cluster.append(score)
                            added = True
                            break
                    if not added:
                        clusters.append([score])

                # Merge small clusters that are close together
                merged_clusters = []
                for cluster in clusters:
                    if merged_clusters and abs(np.mean(cluster) - np.mean(merged_clusters[-1])) <= 0.15:
                        merged_clusters[-1].extend(cluster)
                    else:
                        merged_clusters.append(cluster)

                # For each cluster, create a region
                for cluster_idx, cluster in enumerate(merged_clusters):
                    cluster_min = min(cluster)
                    cluster_max = max(cluster)
                    cluster_mean = np.mean(cluster)

                    # Find all cells in this score range
                    mask = (score_matrix >= cluster_min - 0.01) & (score_matrix <= cluster_max + 0.01)

                    if np.sum(mask) == 0:
                        continue

                    region_scores = score_matrix[mask]
                    region_mean = region_scores.mean()
                    region_std = region_scores.std()
                    region_size = np.sum(mask)

                    # Collect actual (layer, position) pairs in this region
                    region_pairs = []
                    for idx in np.ndindex(score_matrix.shape):
                        if mask[idx]:
                            layer = layers[idx[0]]
                            pos = token_ids[idx[1]]
                            score = score_matrix[idx]
                            region_pairs.append((layer, pos, score))

                    # Sort by score descending
                    region_pairs.sort(key=lambda x: x[2], reverse=True)

                    # Report region
                    output_lines.append(f"\n  Region {cluster_idx + 1}: {cluster_min:.0%}-{cluster_max:.0%}")
                    output_lines.append(f"    Mean: {region_mean:.1%} ± {region_std:.1%}")
                    output_lines.append(f"    Size: {region_size} cells ({region_size/score_matrix.size*100:.1f}%)")

                    # Show actual (layer, position) pairs
                    if region_size <= 20:
                        # Small region - show all pairs
                        output_lines.append(f"    Cells:")
                        for layer, pos, score in region_pairs:
                            output_lines.append(f"      L{layer:2d} @ {pos:15s} ({score:.1%})")
                    else:
                        # Large region - show sample and summary
                        output_lines.append(f"    Sample cells (top 5 by score):")
                        for layer, pos, score in region_pairs[:5]:
                            output_lines.append(f"      L{layer:2d} @ {pos:15s} ({score:.1%})")

                        # Summarize coverage
                        layers_in_region = set(layer for layer, pos, score in region_pairs)
                        positions_in_region = set(pos for layer, pos, score in region_pairs)

                        if len(layers_in_region) == len(layers):
                            output_lines.append(f"    Coverage: All {len(layers)} layers")
                        else:
                            output_lines.append(f"    Coverage: {len(layers_in_region)}/{len(layers)} layers")

                        if len(positions_in_region) == len(token_ids):
                            output_lines.append(f"              All {len(token_ids)} positions")
                        else:
                            output_lines.append(f"              {len(positions_in_region)}/{len(token_ids)} positions: {', '.join(sorted(list(positions_in_region)[:5]))}{', ...' if len(positions_in_region) > 5 else ''}")

                # Show best position for clarity
                max_idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
                best_layer = layers[max_idx[0]]
                best_pos = token_ids[max_idx[1]]
                output_lines.append(f"\nBest single localization: L{best_layer} @ {best_pos} ({max_score:.1%})")

        output_lines.append("\n" + "=" * 80)
        output_lines.append("ANALYSIS COMPLETE")
        output_lines.append("=" * 80)

        # Join all lines
        full_output = "\n".join(output_lines)

        # Print to console
        print(full_output)

        # Optionally save to file
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(full_output)
            print(f"\nText analysis saved to: {save_path}")

    def plot_das_mask_grid(self, results: Dict, target_variables: List[str], dataset_name: str = None,
                           save_path: str = None, show_text: bool = True):
        """
        Visualize DAS+DBM mask counts and layer scores in a grid format.

        This method creates visualizations showing:
        - Number of selected DAS features (mask=1) for each (layer, position)
        - Accuracy score for each layer
        - Both text and graphical outputs

        Designed for results from DAS+DBM training with tie_masks=False.

        Args:
            results: Results dictionary from perform_interventions with computed scores
            target_variables: List of target variable names to analyze (e.g., ["answer"], ["answer", "answer_position"])
            dataset_name: Optional dataset name. If None, uses the first dataset found.
            save_path: Optional base path for saving visualizations.
                      Text will be saved to {save_path}_text.txt
                      Plot will be saved to {save_path}_plot.png
            show_text: If True, prints text visualization to console (default: True)

        Example:
            >>> experiment = PatchResidualStream(pipeline, layers, positions, config=config)
            >>> experiment.train_interventions(labeled_data, method="DAS+DBM")
            >>> raw_results = experiment.perform_interventions(dataset)
            >>> results = compute_interchange_scores(raw_results, causal_model, dataset,
            ...                                      target_variables_list=[["answer"]], checker=checker)
            >>> experiment.plot_das_mask_grid(results, ["answer"], save_path="das_dbm_results")
        """
        # Text visualization
        if show_text:
            text_save_path = f"{save_path}_text.txt" if save_path else None
            print_das_mask_grid(results, target_variables, dataset_name, text_save_path)

        # Graphical visualization
        plot_save_path = f"{save_path}_plot.png" if save_path else None
        create_das_mask_grid_plot(results, target_variables, dataset_name, plot_save_path)


class SameLengthResidualStreamTracing:
    """
    Experiment for tracing through all token positions at all layers with a single counterfactual.
    
    This experiment is designed to comprehensively analyze how information flows through 
    the residual stream by testing interventions at every possible location (layer, position).
    
    Key constraints:
    - Works with a single counterfactual example at a time
    - Requires that the original and counterfactual inputs have the same number of tokens
    - Uses the default featurizer (full vector without transformations)
    - Produces binary accuracy results (0 or 1) for each intervention location
    
    The experiment systematically:
    1. Takes an original input and a counterfactual input of the same length
    2. Runs a PatchResidualStream experiment for each layer in the model and each token position.
    3. Generates a heatmap visualization showing the binary results
    
    This approach helps identify how the causal effect of crucial input tokens are mediated
    through the model's layers and token positions until the final output.
    """
    
    def __init__(self,
                 pipeline: LMPipeline,
                 loss_and_metric_fn: Callable = LM_loss_and_metric_fn):
        """
        Initialize the SameLengthResidualStreamTracing experiment.

        Args:
            pipeline: LMPipeline object for model execution
            loss_and_metric_fn: Function to compute loss and metrics
        """
        self.pipeline = pipeline
        self.loss_and_metric_fn = loss_and_metric_fn
        
        # Get model configuration
        self.num_layers = pipeline.model.config.num_hidden_layers
        self.hidden_size = pipeline.model.config.hidden_size
        
        # Store results for visualization
        self.results = None
        self.token_length = None
    
    def run(self, 
            base_input: Union[str, Dict],
            counterfactual_input: Union[str, Dict],
            save_path: Optional[str] = None) -> Dict:
        """
        Run the tracing experiment with a single counterfactual example.
        
        This method efficiently tests interventions at every (layer, position) combination
        using a single call to perform_interventions with all locations.
        
        Args:
            base_input: The original input (string or dict with 'input' key)
            counterfactual_input: The counterfactual input (must have same token length as base)
            target_variables: List of variable names being analyzed
            
        Returns:
            Dict: Results dictionary containing accuracy scores for each (layer, position) pair
            
        Raises:
            ValueError: If base and counterfactual inputs have different token lengths
        """
        # Tokenize inputs to check length
        base_ids = self.pipeline.load(base_input)
        cf_ids = self.pipeline.load(counterfactual_input)
        self.base_tokens = self.pipeline.tokenizer.convert_ids_to_tokens(base_ids['input_ids'][0])
        self.cf_tokens = self.pipeline.tokenizer.convert_ids_to_tokens(cf_ids['input_ids'][0])

        # Verify same length
        base_length = len(self.base_tokens)
        cf_length = len(self.cf_tokens)
        # Ensure both inputs have the same number of tokens
        if base_length != cf_length:
            raise ValueError(f"Base input has {base_length} tokens but counterfactual has {cf_length} tokens. "
                           f"They must have the same length for this experiment.")
        
        # Store the token length for later use
        self.token_length = base_length
        
        # Create a CounterfactualDataset with just this one example
        data_dict = {
            'input': [base_input],
            'counterfactual_inputs': [[counterfactual_input]],
        }
        dataset = CounterfactualDataset.from_dict(data_dict, id="tracing_example")

        # Create all token position indexers for all positions
        seen_labels = dict()  # To track unique labels
        token_positions = []
        for position in range(self.token_length):
            if self.base_tokens[position] == self.pipeline.tokenizer.pad_token:
                continue  # Skip padding tokens
            # Create a proper closure to capture the position value
            def make_position_indexer(pos):
                return lambda _: [pos]
            
            position_indexer = make_position_indexer(position)
            label = self.base_tokens[position]
            if self.base_tokens[position] != self.cf_tokens[position]:
                label = self.cf_tokens[position] + " -> " + label
            if label in seen_labels:
                seen_labels[label] += 1
                label = label + f"_{seen_labels[label]}"
            else:
                seen_labels[label] = 1

            token_position = TokenPosition(position_indexer, self.pipeline, id=label)
            token_positions.append(token_position)

        # Store token positions for plotting
        self.token_positions = token_positions
        
        # Create all layers list
        layers = [-1] + list(range(self.num_layers))
        
        # Create single PatchResidualStream experiment with all layers and positions
        experiment = PatchResidualStream(
            pipeline=self.pipeline,
            layers=layers,
            token_positions=token_positions,
            featurizers=None,  # Use default featurizer
            loss_and_metric_fn=self.loss_and_metric_fn,
            config={"batch_size": 1, "output_scores": True, "id": "tracing_experiment"},  # Single example
        )
        
        # Run the experiment once with all locations (get raw results)
        raw_results = experiment.perform_interventions(
            {"tracing_example": dataset}
        )

        # Clean up memory after the experiment
        experiment._clean_memory()
        del experiment
        return raw_results
    
    def plot_raw_outputs(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Display the raw generated outputs in a grid format with color coding.

        This method creates a visualization showing the actual tokens generated under
        intervention at each (layer, position) combination. Unlike the heatmap which
        shows accuracy scores, this displays the raw text outputs with cells colored
        based on the output frequency (top 5 most frequent outputs get unique colors).

        Args:
            results: Results dictionary from perform_interventions (must have raw_outputs preserved)
            save_path: Optional path to save the plot. If None, displays interactively.

        Raises:
            ValueError: If raw_outputs are not found in the results
        """
        token_positions = self.token_positions
        # Get dimensions for the grid
        layers = [-1] + list(range(self.num_layers))
        positions = [tp.id for tp in token_positions]

        # Extract raw outputs and convert to text
        dataset_name = "tracing_example"  # This is the dataset name used in run()

        # Create a matrix to store text outputs
        text_outputs = [['' for _ in positions] for _ in layers]

        # Process results to extract raw outputs
        if dataset_name in results["dataset"]:
            for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                if "raw_outputs" not in unit_data:
                    raise ValueError("raw_outputs not found in results. Ensure config['output_scores']=True when running the experiment.")

                if "metadata" in unit_data:
                    layer = unit_data["metadata"].get("layer")
                    position_str = unit_data["metadata"].get("position")

                    # Find position index
                    try:
                        pos_idx = positions.index(position_str)
                    except ValueError:
                        # Try to find by position number if position_str is a token
                        for i, p in enumerate(positions):
                            if str(i) in str(position_str) or position_str == p:
                                pos_idx = i
                                break
                        else:
                            continue

                    # Map layer to list index (layer -1 -> index 0, layer 0 -> index 1, etc.)
                    layer_idx = layers.index(layer)

                    # Get the decoded string directly
                    if unit_data["raw_outputs"]:
                        raw_outputs = unit_data["raw_outputs"]
                        if isinstance(raw_outputs, list) and len(raw_outputs) > 0:
                            # raw_outputs is a list of batch dicts
                            first_batch = raw_outputs[0]
                            if isinstance(first_batch, dict) and "string" in first_batch:
                                decoded_text = first_batch["string"]
                                if isinstance(decoded_text, list):
                                    decoded_text = decoded_text[0]
                                text_outputs[layer_idx][pos_idx] = decoded_text

        # Format layer labels with 'L' prefix
        y_labels = [f'L{layer}' for layer in layers]

        # Use the consolidated visualization function
        create_text_output_grid(
            text_matrix=text_outputs,
            x_labels=positions,
            y_labels=y_labels,
            title=f'Raw Outputs Under Intervention - Experiment: {results["experiment_id"]}',
            save_path=save_path,
            x_label='Token Position',
            y_label='Layer',
            figsize=None,  # Auto-computed
            fontsize=42
        )

    def print_text_analysis(self, results: Dict, original_output: str = None, cf_output: str = None,
                           save_path: Optional[str] = None) -> None:
        """
        Print a text-based representation of the tracing results showing outputs at each location.

        This displays the full matrix of outputs for each (layer, position) combination,
        making it easy to see where interventions change the model's output.

        Args:
            results: Results dictionary from perform_interventions
            original_output: The original model output (optional, for reference in title)
            cf_output: The counterfactual output (optional, for reference in title)
            save_path: Optional path to save the text output to a file
        """
        dataset_name = "tracing_example"

        # Get dimensions
        layers = [-1] + list(range(self.num_layers))
        positions = [tp.id for tp in self.token_positions]

        # Create matrix to store outputs as text
        text_outputs = [['' for _ in positions] for _ in layers]

        # Extract outputs from results
        if dataset_name in results["dataset"]:
            for unit_data in results["dataset"][dataset_name]["model_unit"].values():
                if "metadata" in unit_data and "raw_outputs" in unit_data:
                    layer = unit_data["metadata"].get("layer")
                    position_str = unit_data["metadata"].get("position")

                    # Find position index
                    try:
                        pos_idx = positions.index(position_str)
                    except ValueError:
                        continue

                    # Map layer to list index (layer -1 -> index 0, layer 0 -> index 1, etc.)
                    layer_idx = layers.index(layer)

                    # Extract the output string
                    if unit_data["raw_outputs"]:
                        raw_outputs = unit_data["raw_outputs"]
                        if isinstance(raw_outputs, list) and len(raw_outputs) > 0:
                            first_batch = raw_outputs[0]
                            if isinstance(first_batch, dict) and "string" in first_batch:
                                decoded_text = first_batch["string"]
                                if isinstance(decoded_text, list):
                                    decoded_text = decoded_text[0]
                                text_outputs[layer_idx][pos_idx] = decoded_text

        # Count output frequencies for additional context
        output_counter = Counter()
        for layer_outputs in text_outputs:
            for output in layer_outputs:
                if output:
                    output_counter[output] += 1

        # Build title with context
        title_parts = [f"Tracing Results - Experiment: {results['experiment_id']}"]
        if original_output:
            title_parts.append(f"Original output: '{original_output}'")
        if cf_output:
            title_parts.append(f"Counterfactual output: '{cf_output}'")
        title = "\n".join(title_parts)

        # Add frequency distribution
        freq_info = ["\nOutput Frequency Distribution:"]
        total_positions = len(layers) * len(positions)
        for output, count in output_counter.most_common():
            pct = count / total_positions * 100
            freq_info.append(f"  '{output}': {count} times ({pct:.1f}%)")

        print("\n".join(freq_info))
        print()

        # Format layer labels with 'L' prefix
        y_labels = [f'L{layer}' for layer in layers]

        # Custom formatter for text outputs (just return the string as-is)
        def format_text(val):
            return str(val) if val else "---"

        # Create a pseudo-matrix for printing (we'll use strings but the function expects floats)
        # We'll work around this by temporarily modifying the function or using it creatively
        # Actually, let's just build the text output manually since we have strings not numbers

        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append(title.upper())
        output_lines.append("=" * 80)
        output_lines.append("\nLayer vs Position (showing output strings):\n")

        # Calculate column widths
        max_label_width = max(len(str(label)) for label in positions)
        max_value_width = max(len(str(val)) for row in text_outputs for val in row if val)
        max_value_width = max(max_value_width, 10)  # Minimum width
        col_width = max(max_label_width, max_value_width, 8) + 2

        y_label_width = max(len(str(label)) for label in y_labels) + 2

        # Print column headers
        header = " " * y_label_width + "|"
        for pos in positions:
            # Truncate long labels
            label = str(pos)
            if len(label) > col_width:
                label = label[:col_width-3] + "..."
            header += f" {label:^{col_width}} |"
        output_lines.append(header)
        output_lines.append("-" * len(header))

        # Print each row
        for i, y_label_val in enumerate(y_labels):
            row_str = f"{str(y_label_val):>{y_label_width}}|"
            for j, pos in enumerate(positions):
                value = text_outputs[i][j] if text_outputs[i][j] else "---"
                # Truncate long values
                if len(value) > col_width:
                    value = value[:col_width-3] + "..."
                row_str += f" {value:^{col_width}} |"
            output_lines.append(row_str)

        output_lines.append("\n" + "=" * 80)
        output_lines.append("END OF ANALYSIS")
        output_lines.append("=" * 80)

        # Join all lines
        full_output = "\n".join(output_lines)

        # Print to console
        print(full_output)

        # Optionally save to file
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Combine frequency info with matrix output
            combined_output = "\n".join(freq_info) + "\n\n" + full_output
            with open(save_path, 'w') as f:
                f.write(combined_output)
            print(f"\nText analysis saved to: {save_path}")
