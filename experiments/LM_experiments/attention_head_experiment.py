import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Callable, Tuple, Optional

from experiments.intervention_experiment import *
from neural.LM_units import *
from neural.model_units import *
from neural.pipeline import LMPipeline
from causal.causal_model import CausalModel

from experiments.pyvene_core import _prepare_intervenable_inputs

from .LM_utils import LM_loss_and_metric_fn
from experiments.visualizations import create_heatmap, create_binary_mask_heatmap


class PatchAttentionHeads(InterventionExperiment):
    """
    Generic experiment for analyzing attention head interventions in language models.
    
    This class enables interventions on specific attention heads at various positions
    in the input sequence. By modifying attention head outputs and observing the effect
    on model outputs, we can identify which heads are responsible for specific behaviors.
    
    Attributes:
        layer_head_list (List[Tuple[int, int]]): List of (layer, head) tuples to intervene on
        featurizers (Dict): Mapping of (layer, head, position) tuples to Featurizer instances
        token_position (TokenPosition): Token positions to analyze
    """
    
    def __init__(self,
                 pipeline: LMPipeline,
                 layer_head_lists: List[List[Tuple[int, int]]],
                 token_position: TokenPosition,
                 featurizers: Dict[Tuple[int, int, str], Featurizer] = None,
                 loss_and_metric_fn: Callable = LM_loss_and_metric_fn,
                 config: Dict = None,
                 **kwargs):
        """
        Initialize PatchAttentionHeads for analyzing attention head interventions.

        Args:
            pipeline: LMPipeline object for model execution
            layer_head_lists: List of lists of (layer, head) tuples specifying which heads to intervene on
            token_position: TokenPosition object storing token indices
            featurizers: Dict mapping (layer, head, position.id) to Featurizer instances
            config: Configuration dictionary (must include "id" field)
            **kwargs: Additional configuration options
        """
        self.layer_head_lists = layer_head_lists
        self.featurizers = featurizers if featurizers is not None else {}

        # Extract featurizer_kwargs from config if present
        if config is None:
            config = kwargs.get('config', {})
        featurizer_kwargs = config.get('featurizer_kwargs', {})

        # Generate all combinations of model units
        # Different model architectures use different attribute names for number of heads
        p_config = pipeline.model.config
        if hasattr(p_config, 'head_dim'):
            head_size = p_config.head_dim
        else:
            if hasattr(p_config, 'n_head'):
                num_heads = p_config.n_head
            elif hasattr(p_config, 'num_attention_heads'):
                num_heads = p_config.num_attention_heads
            elif hasattr(p_config, 'num_heads'):
                num_heads = p_config.num_heads
            head_size = pipeline.model.config.hidden_size // num_heads


        model_units_lists = []
        for layer_head_list in layer_head_lists:
            model_units = []
            for layer, head in layer_head_list:
                # Get or create featurizer for this head
                featurizer_key = (layer, head)
                featurizer = self.featurizers.get(
                    featurizer_key,
                    Featurizer(n_features=head_size, **featurizer_kwargs)
                )

                model_units.append(
                    AttentionHead(
                        layer=layer,
                        head=head,
                        token_indices=token_position,
                        featurizer=featurizer,
                        feature_indices=None,
                        target_output=True,
                        shape=(head_size,)
                    )
                )
            model_units_lists.append([model_units])

        # Metadata function to extract layer and head information
        metadata = lambda x: {
            "layer": x[0][0].component.get_layer(),
            "head": x[0][0].head,
            "position": x[0][0].component.get_index_id()
        }

        super().__init__(
            pipeline=pipeline,
            model_units_lists=model_units_lists,
            metadata_fn=metadata,
            config=config,
            **kwargs
        )
        self.loss_and_metric_fn = loss_and_metric_fn
        
        self.token_position = token_position

    def plot_heatmaps(self, results: Dict, target_variables, save_path: str = None, average_counterfactuals: bool = False):
        """
        Generate heatmaps visualizing intervention scores with heads vs layers.

        Results must already have scores computed (via compute_interchange_scores() or compute_custom_scores()).

        Args:
            results: Dictionary containing experiment results with pre-computed scores
            target_variables: List of variable names being analyzed (or metric name for custom scores)
            save_path: Optional path to save the generated plots. If None, displays plots interactively.
            average_counterfactuals: If True, averages scores across counterfactual datasets
        """
        # Validate that each layer_head_list contains exactly one entry for heatmap compatibility
        for i, layer_head_list in enumerate(self.layer_head_lists):
            if len(layer_head_list) != 1:
                raise ValueError(f"For heatmap visualization, each layer_head_list must contain exactly one (layer, head) pair. "
                               f"layer_head_lists[{i}] contains {len(layer_head_list)} entries: {layer_head_list}")

        target_variables_str = "-".join(target_variables)

        # Find the range of layers and heads from the layer_head_lists
        all_layers = [layer_head_list[0][0] for layer_head_list in self.layer_head_lists]
        all_heads = [layer_head_list[0][1] for layer_head_list in self.layer_head_lists]

        layers = list(range(min(all_layers), max(all_layers) + 1))
        heads = list(range(min(all_heads), max(all_heads) + 1))

        if average_counterfactuals:
            self._plot_average_heatmap(results, layers, heads, target_variables_str, save_path)
        else:
            self._plot_individual_heatmaps(results, layers, heads, target_variables_str, save_path)

    def print_text_analysis(self, results: Dict, target_variables: List[str],
                           save_path: Optional[str] = None, custom_scoring_fn: Callable = None,
                           use_actual_outputs: bool = False) -> None:
        """
        Print a text-based representation of intervention accuracy scores.

        This method displays the full score matrix in text format for each variable and dataset,
        along with summary statistics. This is useful for analyzing results in environments
        where graphical displays are not available or for logging detailed results.

        Args:
            results: Results dictionary from perform_interventions
            target_variables: List of variable names being analyzed
            save_path: Optional path to save the text output to a file
            custom_scoring_fn: Optional function to compute scores from causal_model_inputs and raw_outputs
            use_actual_outputs: If True, passes actual outputs to custom scoring function
        """
        # Find the range of layers and heads from the layer_head_lists
        all_layers = [layer_head_list[0][0] for layer_head_list in self.layer_head_lists]
        all_heads = [layer_head_list[0][1] for layer_head_list in self.layer_head_lists]

        layers = list(range(min(all_layers), max(all_layers) + 1))
        heads = list(range(min(all_heads), max(all_heads) + 1))

        all_outputs = []

        # Analyze each variable
        for target_vars in target_variables:
            if not isinstance(target_vars, list):
                target_vars = [target_vars]

            target_variables_str = "-".join(target_vars)

            # Build score matrices for all datasets
            matrices = self._build_score_matrix(
                results, layers, heads, target_variables_str,
                dataset_names=None,
                custom_scoring_fn=custom_scoring_fn,
                use_actual_outputs=use_actual_outputs
            )

            if not matrices:
                print(f"No data available for variable: {target_variables_str}")
                continue

            # Analyze each dataset
            for dataset_name, score_matrix in matrices.items():
                # Transpose the score matrix so layers are on y-axis and heads are on x-axis
                transposed_score_matrix = score_matrix.T

                # Format labels with prefixes
                x_labels = [f"H{head}" for head in heads]
                y_labels = [f"L{layer}" for layer in layers]

                title = (f"Attention Head Patching Analysis\n"
                        f"Experiment: {results['experiment_id']}\n"
                        f"Variable: {target_variables_str}\n"
                        f"Dataset: {dataset_name}")

                # Determine format function based on whether custom scoring is used
                if custom_scoring_fn is not None:
                    # For custom scoring, show raw values
                    format_fn = lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
                else:
                    # For accuracy metrics, show percentages
                    format_fn = lambda x: f"{x:.1%}" if not np.isnan(x) else "N/A"

                # Use the consolidated text heatmap function
                from experiments.visualizations import print_text_heatmap
                output = print_text_heatmap(
                    score_matrix=transposed_score_matrix,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    title=title,
                    save_path=None,  # We'll save combined output at the end
                    x_label='Head',
                    y_label='Layer',
                    format_fn=format_fn,
                    include_stats=True
                )
                all_outputs.append(output)

        # Save all outputs to file if requested
        if save_path and all_outputs:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write("\n\n".join(all_outputs))
            print(f"\nText analysis saved to: {save_path}")

    def _build_score_matrix(self,
                            results: Dict,
                            layers: List,
                            heads: List,
                            target_variables_str: str,
                            dataset_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract score matrices from results for specified datasets.

        Scores must already be computed via compute_interchange_scores() or compute_custom_scores().

        Args:
            results: Dictionary containing experiment results with pre-computed scores
            layers: List of layer indices
            heads: List of head indices
            target_variables_str: String identifier for target variables (or metric name)
            dataset_names: List of dataset names to process. If None, processes all datasets.

        Returns:
            Dictionary mapping dataset names to their score matrices (with NaN for missing entries)
        """
        if dataset_names is None:
            dataset_names = list(results["dataset"].keys())

        matrices = {}

        for dataset_name in dataset_names:
            # Initialize score matrix with NaN (will show as blank)
            score_matrix = np.full((len(heads), len(layers)), np.nan)
            valid_entries = False

            # Fill score matrix
            for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                if "metadata" in unit_data:
                    metadata = unit_data["metadata"]
                    layer = metadata.get("layer")
                    head = metadata.get("head")

                    # Check if this layer/head combination is in our ranges
                    if layer in layers and head in heads:
                        layer_idx = layers.index(layer)
                        head_idx = heads.index(head)

                        # Use pre-computed average score
                        if target_variables_str in unit_data and "average_score" in unit_data[target_variables_str]:
                            score_matrix[head_idx, layer_idx] = unit_data[target_variables_str]["average_score"]
                            valid_entries = True

            # Only include datasets with valid entries
            if valid_entries:
                matrices[dataset_name] = score_matrix

        return matrices

    def _aggregate_matrices(self,
                           matrices: Dict[str, np.ndarray],
                           aggregation: str = "average") -> np.ndarray:
        """
        Aggregate multiple score matrices using specified method with NaN support.

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
            return np.nanmean(matrix_array, axis=0)
        elif aggregation == "sum":
            return np.nansum(matrix_array, axis=0)
        elif aggregation == "max":
            return np.nanmax(matrix_array, axis=0)
        elif aggregation == "min":
            return np.nanmin(matrix_array, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

    def _build_mask_matrix(self, dataset_results: Dict, dataset_name: str) -> Optional[Tuple[np.ndarray, List[int], List[int]]]:
        """
        Extract binary mask matrix from dataset results showing which attention heads are selected.

        This helper method parses feature_indices data to build a mask matrix where:
        - 1 indicates the head is selected (feature_indices is None)
        - 0 indicates the head is not selected (feature_indices is [])
        - NaN indicates missing data

        Args:
            dataset_results: Results dictionary for a specific dataset
            dataset_name: Name of the dataset (for error messages)

        Returns:
            Tuple of (mask_matrix, layers, heads) if successful, None if data is invalid.
            - mask_matrix: 2D numpy array of shape (len(heads), len(layers)) with binary values
            - layers: Sorted list of layer indices
            - heads: Sorted list of head indices
        """
        if "model_unit" not in dataset_results:
            print(f"Warning: No model_unit data found for dataset {dataset_name}")
            return None

        # Find the single model unit key (it's a long string with all heads)
        model_unit_keys = list(dataset_results["model_unit"].keys())
        if len(model_unit_keys) != 1:
            print(f"Warning: Expected exactly 1 model_unit key, found {len(model_unit_keys)}. "
                  f"This method is designed for results with a single model_units_list containing all heads.")
            return None

        model_unit_key = model_unit_keys[0]
        model_unit_data = dataset_results["model_unit"][model_unit_key]

        if "feature_indices" not in model_unit_data:
            print(f"Warning: No feature_indices found in model_unit data for dataset {dataset_name}")
            return None

        feature_indices = model_unit_data["feature_indices"]

        # Parse attention head information and create binary mask
        head_info = []
        for head_key, indices in feature_indices.items():
            # Parse layer and head from key like "AttentionHead(Layer-0,Head-0,Token-last_token)"
            if "AttentionHead(Layer-" in head_key:
                try:
                    # Extract layer and head numbers
                    layer_part = head_key.split("Layer-")[1].split(",")[0]
                    head_part = head_key.split("Head-")[1].split(",")[0]
                    layer = int(layer_part)
                    head = int(head_part)

                    # Create binary mask: 0 for [], 1 for [0]
                    mask_value = 1 if indices is None else 0

                    head_info.append((layer, head, mask_value))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse head key {head_key}: {e}")
                    continue

        if not head_info:
            print(f"Warning: No valid attention head data found for dataset {dataset_name}")
            return None

        # Determine the grid dimensions
        layers = sorted(list(set(info[0] for info in head_info)))
        heads = sorted(list(set(info[1] for info in head_info)))

        # Create binary mask matrix
        mask_matrix = np.full((len(heads), len(layers)), np.nan)

        for layer, head, mask_value in head_info:
            if layer in layers and head in heads:
                layer_idx = layers.index(layer)
                head_idx = heads.index(head)
                mask_matrix[head_idx, layer_idx] = mask_value

        return mask_matrix, layers, heads

    def _plot_average_heatmap(self, results: Dict, layers: List, heads: List,
                             target_variables_str: str, save_path: Optional[str] = None):
        """Create and save/display an averaged heatmap across all datasets."""
        # Build score matrices for all datasets
        matrices = self._build_score_matrix(results, layers, heads, target_variables_str, dataset_names=None)

        if not matrices:
            print("Warning: No valid data found for visualization.")
            return

        # Aggregate matrices
        score_matrix = self._aggregate_matrices(matrices, aggregation="average")

        # Use "averaged" for filename
        safe_dataset_name = "averaged_across_datasets"

        # Create the heatmap
        self._create_heatmap(
            score_matrix=score_matrix,
            layers=layers,
            heads=heads,
            title=f'Attention Head Intervention Accuracy (Averaged)\nExperiment: {results["experiment_id"]}',
            save_path=os.path.join(save_path, f'attention_heatmap_{safe_dataset_name}_{results["experiment_id"]}.png') if save_path else None,
            use_custom_bounds=False
        )

    def _plot_individual_heatmaps(self, results: Dict, layers: List, heads: List,
                                 target_variables_str: str, save_path: Optional[str] = None):
        """Create and save/display individual heatmaps for each dataset."""
        # Build score matrices for all datasets
        matrices = self._build_score_matrix(results, layers, heads, target_variables_str, dataset_names=None)

        if not matrices:
            print("Warning: No valid data found for visualization.")
            return

        # Create individual heatmaps for each dataset
        for dataset_name, score_matrix in matrices.items():
            # Convert dataset name to a safe filename
            safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')

            # Create the heatmap
            self._create_heatmap(
                score_matrix=score_matrix,
                layers=layers,
                heads=heads,
                title=f'Attention Head Intervention Accuracy - Dataset: {dataset_name}\nExperiment: {results["experiment_id"]}',
                save_path=os.path.join(save_path, f'attention_heatmap_{safe_dataset_name}_{results["experiment_id"]}.png') if save_path else None,
                use_custom_bounds=False
            )

    def _create_heatmap(self, score_matrix: np.ndarray, layers: List, heads: List,
                       title: str, save_path: str = None, use_custom_bounds: bool = False):
        """
        Create and save/display a single heatmap with heads vs layers.

        Args:
            score_matrix: 2D numpy array with scores for each (head, layer) pair
            layers: List of layer indices
            heads: List of head indices
            title: Title for the heatmap
            save_path: Path to save the heatmap, or None to display it
            use_custom_bounds: If True, automatically infer vmin/vmax from the data (for custom metrics)
        """
        # Transpose the score matrix so layers are on y-axis and heads are on x-axis
        transposed_score_matrix = score_matrix.T

        # Format labels with prefixes
        x_labels = [f"H{head}" for head in heads]
        y_labels = [f"L{layer}" for layer in layers]

        # Determine colorbar label and figure size
        cbar_label = 'Score' if use_custom_bounds else 'Accuracy (%)'
        figsize = (max(12, len(heads) * 0.6), max(6, len(layers) * 0.8))

        # Use the consolidated visualization function
        create_heatmap(
            score_matrix=transposed_score_matrix,
            x_labels=x_labels,
            y_labels=y_labels,
            title=title,
            save_path=save_path,
            x_label='Head',
            y_label='Layer',
            use_custom_bounds=use_custom_bounds,
            cbar_label=cbar_label,
            figsize=figsize
        )

    def plot_mask_heatmap(self, results: Dict, save_path: str = None):
        """
        Generate binary heatmap showing which attention heads have features selected (mask=1) or not (mask=0).

        This method is designed for results from experiments with a single model_units_list containing
        many attention heads, where each head has feature_indices that are either [] (mask=0) or [0] (mask=1).

        Args:
            results: Dictionary containing experiment results with feature_indices
            save_path: Optional path to save the generated plot. If None, displays plot interactively.
        """
        # Get dataset names
        dataset_names = list(results["dataset"].keys())

        if not dataset_names:
            print("Warning: No datasets found in results.")
            return

        # Process each dataset
        for dataset_name in dataset_names:
            dataset_results = results["dataset"][dataset_name]

            # Use helper method to build mask matrix
            result = self._build_mask_matrix(dataset_results, dataset_name)
            if result is None:
                continue

            mask_matrix, layers, heads = result

            # Create the mask heatmap
            self._create_mask_heatmap(
                mask_matrix=mask_matrix,
                layers=layers,
                heads=heads,
                title=f'Attention Head Mask - Dataset: {dataset_name}\nExperiment: {results.get("experiment_id", "Unknown Experiment")}',
                save_path=os.path.join(save_path, f'attention_mask_{dataset_name.replace(" ", "_")}_{results.get("experiment_id", "unknown_task").replace(" ", "_")}.png') if save_path else None
            )

    def _create_mask_heatmap(self, mask_matrix: np.ndarray, layers: List, heads: List,
                            title: str, save_path: str = None):
        """
        Create and save/display a binary mask heatmap.

        Args:
            mask_matrix: 2D numpy array with binary values (0, 1, or NaN for missing)
            layers: List of layer indices
            heads: List of head indices
            title: Title for the heatmap
            save_path: Path to save the heatmap, or None to display it
        """
        # Transpose the mask matrix so layers are on y-axis and heads are on x-axis
        transposed_mask_matrix = mask_matrix.T

        # Format labels with prefixes
        x_labels = [f"H{head}" for head in heads]
        y_labels = [f"L{layer}" for layer in layers]

        # Use the consolidated visualization function
        create_binary_mask_heatmap(
            mask_matrix=transposed_mask_matrix,
            x_labels=x_labels,
            y_labels=y_labels,
            title=title,
            save_path=save_path,
            x_label='Head',
            y_label='Layer',
            figsize=(max(12, len(heads) * 0.6), max(6, len(layers) * 0.8))
        )

    def print_text_analysis(self, results: Dict, target_variables: List[str],
                           save_path: Optional[str] = None) -> None:
        """
        Print a text-based representation of DBM mask results showing which attention heads were selected.

        This method displays all attention heads with nonzero masks (selected by DBM) along with
        accuracy scores for each dataset.

        Args:
            results: Results dictionary from perform_interventions
            target_variables: List of variable names being analyzed
            save_path: Optional path to save the text output to a file
        """
        # Get dataset names
        dataset_names = list(results["dataset"].keys())

        if not dataset_names:
            print("Warning: No datasets found in results.")
            return

        all_outputs = []

        # Process each dataset
        for dataset_name in dataset_names:
            dataset_results = results["dataset"][dataset_name]

            if "model_unit" not in dataset_results:
                print(f"Warning: No model_unit data found for dataset {dataset_name}")
                continue

            # Find the single model unit key (it's a long string with all heads)
            model_unit_keys = list(dataset_results["model_unit"].keys())
            if len(model_unit_keys) != 1:
                print(f"Warning: Expected exactly 1 model_unit key, found {len(model_unit_keys)}. "
                      "This method is designed for results with a single model_units_list containing all heads.")
                continue

            model_unit_key = model_unit_keys[0]
            model_unit_data = dataset_results["model_unit"][model_unit_key]

            if "feature_indices" not in model_unit_data:
                print(f"Warning: No feature_indices found in model_unit data for dataset {dataset_name}")
                continue

            feature_indices = model_unit_data["feature_indices"]

            # Parse attention head information and identify nonzero masks
            selected_heads = []
            for head_key, indices in feature_indices.items():
                # Parse layer and head from key like "AttentionHead(Layer-0,Head-0,Token-last_token)"
                if "AttentionHead(Layer-" in head_key:
                    try:
                        # Extract layer and head numbers
                        layer_part = head_key.split("Layer-")[1].split(",")[0]
                        head_part = head_key.split("Head-")[1].split(",")[0]
                        layer = int(layer_part)
                        head = int(head_part)

                        # Nonzero mask: indices is None means mask=1 (selected)
                        if indices is None:
                            selected_heads.append((layer, head))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse head key {head_key}: {e}")
                        continue

            # Sort by layer, then by head
            selected_heads.sort()

            # Get accuracy scores for the target variables
            target_variables_str = "-".join(target_variables) if isinstance(target_variables, list) else target_variables
            accuracy_info = ""
            if target_variables_str in model_unit_data and "average_score" in model_unit_data[target_variables_str]:
                accuracy = model_unit_data[target_variables_str]["average_score"]
                accuracy_info = f"\nInterchange Intervention Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)"

            # Build output text
            output_lines = []
            output_lines.append("=" * 80)
            output_lines.append(f"DBM ATTENTION HEAD MASK ANALYSIS")
            output_lines.append("=" * 80)
            output_lines.append(f"Experiment: {results.get('experiment_id', 'Unknown Experiment')}")
            output_lines.append(f"Method: {results.get('method_name', 'Unknown Experiment')}")
            output_lines.append(f"Dataset: {dataset_name}")
            output_lines.append(f"Target Variable(s): {target_variables_str}")
            if accuracy_info:
                output_lines.append(accuracy_info)
            output_lines.append("")
            output_lines.append(f"Selected Attention Heads (nonzero masks): {len(selected_heads)}")
            output_lines.append("-" * 80)

            if selected_heads:
                # Group by layer for readability
                current_layer = None
                for layer, head in selected_heads:
                    if layer != current_layer:
                        if current_layer is not None:
                            output_lines.append("")
                        output_lines.append(f"Layer {layer}:")
                        current_layer = layer
                    output_lines.append(f"  Head {head}")
            else:
                output_lines.append("(No heads selected - sparse/distributed representation)")

            output_lines.append("=" * 80)

            output = "\n".join(output_lines)
            all_outputs.append(output)
            print(output)

        # Save all outputs to file if requested
        if save_path and all_outputs:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            with open(save_path, 'w') as f:
                f.write("\n\n".join(all_outputs))
            print(f"\nText analysis saved to: {save_path}")
