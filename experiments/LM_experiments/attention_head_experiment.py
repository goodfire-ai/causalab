import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Callable, Tuple, Optional

from experiments.intervention_experiment import *
from neural.LM_units import *
from neural.model_units import *
from neural.pipeline import LMPipeline
from causal.causal_model import CausalModel

from experiments.pyvene_core import _prepare_intervenable_inputs

from .LM_utils import LM_loss_and_metric_fn


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
                 causal_model: CausalModel,
                 layer_head_lists: List[List[Tuple[int, int]]],
                 token_position: TokenPosition,
                 checker: Callable,
                 featurizers: Dict[Tuple[int, int, str], Featurizer] = None,
                 loss_and_metric_fn: Callable = LM_loss_and_metric_fn,
                 config: Dict = None,
                 **kwargs):
        """
        Initialize PatchAttentionHeads for analyzing attention head interventions.
        
        Args:
            pipeline: LMPipeline object for model execution
            causal_model: CausalModel object containing the task
            layers: List of layer indices (kept for compatibility but not used directly)
            layer_head_list: List of (layer, head) tuples specifying which heads to intervene on
            token_position: TokenPosition object storing token indices
            checker: Function to evaluate output accuracy
            featurizers: Dict mapping (layer, head, position.id) to Featurizer instances
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
            causal_model=causal_model,
            model_units_lists=model_units_lists,
            checker=checker,
            metadata_fn=metadata,
            config=config,
            **kwargs
        )
        self.loss_and_metric_fn = loss_and_metric_fn
        
        self.token_position = token_position

    def plot_heatmaps(self, results: Dict, target_variables, save_path: str = None, average_counterfactuals: bool = False, custom_scoring_fn: Callable = None, use_actual_outputs: bool = False):
        """
        Generate heatmaps visualizing intervention scores with heads vs layers.

        Args:
            results: Dictionary containing experiment results from interpret_results()
            target_variables: List of variable names being analyzed
            save_path: Optional path to save the generated plots. If None, displays plots interactively.
            average_counterfactuals: If True, averages scores across counterfactual datasets
            custom_scoring_fn: Optional function with signature (causal_model_input: Dict, raw_output: Dict, actual_output: Dict = None) -> float
                             that processes each input/output pair. If None, uses pre-computed average_score.
            use_actual_outputs: If True and custom_scoring_fn provided, passes actual outputs to custom function
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
            self._plot_average_heatmap(results, layers, heads, target_variables_str, save_path, custom_scoring_fn, use_actual_outputs)
        else:
            self._plot_individual_heatmaps(results, layers, heads, target_variables_str, save_path, custom_scoring_fn, use_actual_outputs)

    def _build_score_matrix(self,
                            results: Dict,
                            layers: List,
                            heads: List,
                            target_variables_str: str,
                            dataset_names: Optional[List[str]] = None,
                            custom_scoring_fn: Callable = None,
                            use_actual_outputs: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract score matrices from results for specified datasets.

        Args:
            results: Dictionary containing experiment results
            layers: List of layer indices
            heads: List of head indices
            target_variables_str: String identifier for target variables
            dataset_names: List of dataset names to process. If None, processes all datasets.
            custom_scoring_fn: Optional function to compute scores from causal_model_inputs and raw_outputs.
                             If None, uses pre-computed average_score.
            use_actual_outputs: If True, passes actual outputs to custom scoring function.

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

                        # Compute score using custom function or pre-computed average
                        if custom_scoring_fn is not None:
                            # Use custom scoring function with causal inputs and raw outputs
                            if "causal_model_inputs" in unit_data and "raw_outputs" in unit_data:
                                scores = []
                                causal_inputs = unit_data["causal_model_inputs"]
                                raw_outputs = unit_data["raw_outputs"]

                                # Get actual outputs if requested
                                actual_outputs = None
                                if use_actual_outputs and "raw_outputs_no_intervention" in results["dataset"][dataset_name]:
                                    actual_outputs = results["dataset"][dataset_name]["raw_outputs_no_intervention"]

                                # Process batches simultaneously
                                input_idx = 0
                                for batch_idx, intervention_batch in enumerate(raw_outputs):
                                    actual_batch = actual_outputs[batch_idx] if actual_outputs else None
                                    batch_size = intervention_batch["sequences"].shape[0]

                                    for i in range(batch_size):
                                        if input_idx < len(causal_inputs):
                                            # Create intervention output dict
                                            intervention_output = {"sequences": intervention_batch["sequences"][i:i+1]}
                                            if "scores" in intervention_batch:
                                                intervention_output["scores"] = [score[i:i+1] for score in intervention_batch["scores"]]
                                            if "string" in intervention_batch:
                                                if isinstance(intervention_batch["string"], list):
                                                    intervention_output["string"] = intervention_batch["string"][i]
                                                else:
                                                    intervention_output["string"] = intervention_batch["string"]

                                            # Create actual output dict if available
                                            actual_output = None
                                            if actual_batch is not None:
                                                actual_output = {"sequences": actual_batch["sequences"][i:i+1]}
                                                if "scores" in actual_batch:
                                                    actual_output["scores"] = [score[i:i+1] for score in actual_batch["scores"]]
                                                if "string" in actual_batch:
                                                    if isinstance(actual_batch["string"], list):
                                                        actual_output["string"] = actual_batch["string"][i]
                                                    else:
                                                        actual_output["string"] = actual_batch["string"]

                                            # Apply custom scoring function
                                            if use_actual_outputs and actual_output is not None:
                                                score = custom_scoring_fn(causal_inputs[input_idx], intervention_output, actual_output=actual_output)
                                            else:
                                                score = custom_scoring_fn(causal_inputs[input_idx], intervention_output)
                                            scores.append(score)
                                            input_idx += 1

                                if scores:
                                    score_matrix[head_idx, layer_idx] = np.mean(scores)
                                    valid_entries = True
                            else:
                                print(f"Warning: Custom scoring function provided but causal_model_inputs or raw_outputs not found for {unit_str}")
                        else:
                            # Use pre-computed average score (default behavior)
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

    def _plot_average_heatmap(self, results: Dict, layers: List, heads: List,
                             target_variables_str: str, save_path: Optional[str] = None, custom_scoring_fn: Callable = None, use_actual_outputs: bool = False):
        """Create and save/display an averaged heatmap across all datasets."""
        # Build score matrices for all datasets
        matrices = self._build_score_matrix(results, layers, heads, target_variables_str, dataset_names=None, custom_scoring_fn=custom_scoring_fn, use_actual_outputs=use_actual_outputs)

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
            title=f'Attention Head Intervention Accuracy (Averaged)\nTask: {results["task_name"]}',
            save_path=os.path.join(save_path, f'attention_heatmap_{safe_dataset_name}_{results["task_name"]}.png') if save_path else None,
            use_custom_bounds=(custom_scoring_fn is not None)
        )

    def _plot_individual_heatmaps(self, results: Dict, layers: List, heads: List,
                                 target_variables_str: str, save_path: Optional[str] = None, custom_scoring_fn: Callable = None, use_actual_outputs: bool = False):
        """Create and save/display individual heatmaps for each dataset."""
        # Build score matrices for all datasets
        matrices = self._build_score_matrix(results, layers, heads, target_variables_str, dataset_names=None, custom_scoring_fn=custom_scoring_fn, use_actual_outputs=use_actual_outputs)

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
                title=f'Attention Head Intervention Accuracy - Dataset: {dataset_name}\nTask: {results["task_name"]}',
                save_path=os.path.join(save_path, f'attention_heatmap_{safe_dataset_name}_{results["task_name"]}.png') if save_path else None,
                use_custom_bounds=(custom_scoring_fn is not None)
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
        plt.figure(figsize=(max(12, len(heads) * 0.6), max(6, len(layers) * 0.8)))

        # Determine vmin and vmax based on whether we're using custom bounds
        if use_custom_bounds:
            # For custom metrics, infer bounds from the actual data (ignoring NaN)
            data_min = np.nanmin(score_matrix)
            data_max = np.nanmax(score_matrix)

            # Check if we have both positive and negative values
            if data_min < 0 and data_max > 0:
                # Center the colormap at 0 for diverging data
                abs_max = max(abs(data_min), abs(data_max))
                # Add a small margin
                margin = abs_max * 0.05 if abs_max != 0 else 0.1
                vmin = -abs_max - margin
                vmax = abs_max + margin
                # Use a diverging colormap with white at center
                # Note: 'coolwarm_r' has red for negative, white at 0, blue for positive
                cmap = 'coolwarm_r'  # Red for negative, white at 0, blue for positive
            else:
                # For all-positive or all-negative data, use standard bounds
                margin = (data_max - data_min) * 0.05 if data_max != data_min else 0.1
                vmin = data_min - margin
                vmax = data_max + margin
                cmap = 'Reds'  # Use red-based colormap

            cbar_label = 'Score'  # Generic label for custom metrics
            # Don't multiply by 100 for custom metrics
            display_matrix = np.round(score_matrix, 4)
        else:
            # Default bounds for accuracy metrics
            vmin = 0
            vmax = 1
            cmap = 'Reds'
            cbar_label = 'Accuracy (%)'
            # Multiply by 100 for percentage display
            display_matrix = np.round(score_matrix * 100, 2)

        # Create annotation matrix - show values for non-NaN, empty string for NaN
        annot_matrix = np.where(np.isnan(score_matrix), "", display_matrix.astype(str))

        # Transpose the matrices to swap axes
        transposed_score_matrix = score_matrix.T
        transposed_annot_matrix = annot_matrix.T

        # Create the heatmap using seaborn with swapped axes
        sns.heatmap(
            transposed_score_matrix,
            xticklabels=[f"H{head}" for head in heads],
            yticklabels=[f"L{layer}" for layer in layers],
            cmap=cmap,
            annot=transposed_annot_matrix,
            fmt="",  # Use string format since we're providing custom annotations
            cbar_kws={'label': cbar_label},
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            center=0 if (use_custom_bounds and vmin < 0 and vmax > 0) else None
        )

        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.title(title)
        plt.tight_layout()

        if save_path:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

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

            if "model_unit" not in dataset_results:
                print(f"Warning: No model_unit data found for dataset {dataset_name}")
                continue

            # Find the single model unit key (it's a long string with all heads)
            model_unit_keys = list(dataset_results["model_unit"].keys())
            if len(model_unit_keys) != 1:
                print(f"Warning: Expected exactly 1 model_unit key, found {len(model_unit_keys)}. This method is designed for results with a single model_units_list containing all heads.")
                continue

            model_unit_key = model_unit_keys[0]
            model_unit_data = dataset_results["model_unit"][model_unit_key]

            if "feature_indices" not in model_unit_data:
                print(f"Warning: No feature_indices found in model_unit data for dataset {dataset_name}")
                continue

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
                continue

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

            # Create the mask heatmap
            self._create_mask_heatmap(
                mask_matrix=mask_matrix,
                layers=layers,
                heads=heads,
                title=f'Attention Head Mask - Dataset: {dataset_name}\nTask: {results.get("task_name", "Unknown Task")}',
                save_path=os.path.join(save_path, f'attention_mask_{dataset_name.replace(" ", "_")}_{results.get("task_name", "unknown_task").replace(" ", "_")}.png') if save_path else None
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
        plt.figure(figsize=(max(12, len(heads) * 0.6), max(6, len(layers) * 0.8)))

        # Create custom colormap: light red for 0, dark blue for 1, light gray for NaN
        from matplotlib.colors import ListedColormap
        colors = ['#FFB3BA', '#00008B']  # Light red for 0, dark blue for 1
        cmap = ListedColormap(colors)

        # Create annotation matrix - show "0" and "1" for valid values, empty for NaN
        annot_matrix = np.where(np.isnan(mask_matrix), "", mask_matrix.astype(int).astype(str))

        # Transpose the matrices to swap axes
        transposed_mask_matrix = mask_matrix.T
        transposed_annot_matrix = annot_matrix.T

        # Create the heatmap with swapped axes
        sns.heatmap(
            transposed_mask_matrix,
            xticklabels=[f"H{head}" for head in heads],
            yticklabels=[f"L{layer}" for layer in layers],
            cmap=cmap,
            annot=transposed_annot_matrix,
            fmt="",  # Use string format since we're providing custom annotations
            cbar_kws={'label': 'Mask Value', 'ticks': [0, 1]},
            vmin=0,
            vmax=1,
            cbar=True,
            linewidths=0.5,
            linecolor='lightgray'
        )

        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.title(title)
        plt.tight_layout()

        if save_path:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()