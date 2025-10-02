import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
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

from experiments.LM_helpers import LM_loss_and_metric_fn
from experiments.intervention_experiment import *
from causal.causal_model import CausalModel
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
                 causal_model: CausalModel,
                 layers: List[int],
                 token_positions: List[TokenPosition],
                 checker: Callable,
                 featurizers: Dict[Tuple[int, str], Featurizer] = None,
                 loss_and_metric_fn: Callable = LM_loss_and_metric_fn,
                 **kwargs):
        """
        Initialize ResidualStreamExperiment for analyzing residual stream interventions.
        
        Args:
            pipeline: LMPipeline object for model execution
            causal_model: CausalModel object for causal analysis
            layers: List of layer indices to analyze
            token_positions: List of ComponentIndexers for token positions
            checker: Function to evaluate output accuracy
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
            for pos in token_positions:
                featurizer = self.featurizers.get((layer, pos.id),
                                                 Featurizer(n_features=pipeline.model.config.hidden_size,
                                                           **featurizer_kwargs))
                model_units_lists.append([[
                    ResidualStream(
                        layer=layer,
                        token_indices=pos,
                        featurizer=featurizer,
                        shape=(pipeline.model.config.hidden_size,),
                        feature_indices=None,
                        target_output=True
                    )
                ]])

        metadata_fn = lambda x: {"layer": x[0][0].component.get_layer(), 
                                "position": x[0][0].component.get_index_id()}

        super().__init__(
            pipeline=pipeline,
            causal_model=causal_model,
            model_units_lists=model_units_lists,
            checker=checker,
            metadata_fn=metadata_fn,
            **kwargs
        )
        
        self.layers = layers
        self.token_positions = token_positions

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
    
    def _plot_average_heatmap(self, results: Dict, layers: List, positions: List, 
                             target_variables_str: str, save_path: Optional[str] = None):
        """Create and save/display an averaged heatmap across all datasets."""
        # Initialize score matrix and counter
        score_matrix = np.zeros((len(layers), len(positions)))
        dataset_count = 0.0
        
        # Sum scores across all datasets
        for dataset_name in results["dataset"]:
            temp_matrix = np.zeros((len(layers), len(positions)))
            valid_entries = False
            
            # Fill temporary matrix for this dataset
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                        if "metadata" in unit_data and target_variables_str in unit_data:
                            if "average_score" in unit_data[target_variables_str]:
                                metadata = unit_data["metadata"]
                                if metadata.get("layer") == layer and metadata.get("position") == pos:
                                    temp_matrix[i, j] = unit_data[target_variables_str]["average_score"]
                                    valid_entries = True
            
            # Only include datasets with valid entries
            if valid_entries:
                score_matrix += temp_matrix
                dataset_count += 1
        
        # Calculate average across datasets
        if dataset_count > 0:
            score_matrix /= dataset_count
            
            # Create the heatmap
            self._create_heatmap(
                score_matrix=score_matrix,
                layers=layers,
                positions=positions,
                title=f'Intervention Accuracy - Average across {dataset_count} datasets\nTask: {results["task_name"]}',
                save_path=os.path.join(save_path, f'heatmap_average_{results["task_name"]}.png') if save_path else None
            )
    
    def _plot_individual_heatmaps(self, results: Dict, layers: List, positions: List, 
                                 target_variables_str: str, save_path: Optional[str] = None):
        """Create and save/display individual heatmaps for each dataset."""
        # Get dataset names
        dataset_names = list(results["dataset"].keys())
        
        # Track if we have valid data for any dataset
        any_valid_entries = False
        
        # Create individual heatmaps for each dataset
        for dataset_name in dataset_names:
            score_matrix = np.zeros((len(layers), len(positions)))
            valid_entries = False
            
            # Fill score matrix
            for i, layer in enumerate(layers):
                for j, pos in enumerate(positions):
                    for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                        if "metadata" in unit_data and target_variables_str in unit_data:
                            if "average_score" in unit_data[target_variables_str]:
                                metadata = unit_data["metadata"]
                                if metadata.get("layer") == layer and metadata.get("position") == pos:
                                    score_matrix[i, j] = unit_data[target_variables_str]["average_score"]
                                    valid_entries = True
            
            if valid_entries:
                any_valid_entries = True
                
                # Convert dataset name to a safe filename
                safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                
                # Create the heatmap
                self._create_heatmap(
                    score_matrix=score_matrix,
                    layers=layers,
                    positions=positions,
                    title=f'Intervention Accuracy - Dataset: {dataset_name}\nTask: {results["task_name"]}',
                    save_path=os.path.join(save_path, f'heatmap_{safe_dataset_name}_{results["task_name"]}.png') if save_path else None
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
        plt.figure(figsize=(10, 6))
        display_matrix = np.round(score_matrix * 100, 2)
        
        # Create the heatmap using seaborn
        sns.heatmap(
            score_matrix,
            xticklabels=positions,
            yticklabels=layers,
            cmap='viridis',
            annot=display_matrix,
            fmt="g",
            cbar_kws={'label': 'Accuracy (%)'},
            vmin=0,
            vmax=1,
        )
        
        plt.yticks(rotation=0)
        plt.xlabel('Position')
        plt.ylabel('Layer')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


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
                 causal_model: CausalModel,
                 checker: Callable,
                 loss_and_metric_fn: Callable = LM_loss_and_metric_fn):
        """
        Initialize the SameLengthResidualStreamTracing experiment.
        
        Args:
            pipeline: LMPipeline object for model execution
            causal_model: CausalModel object for causal analysis
            checker: Function to evaluate output accuracy (should return binary 0/1)
            loss_and_metric_fn: Function to compute loss and metrics
            **kwargs: Additional configuration options passed to PatchResidualStream
        """
        self.pipeline = pipeline
        self.causal_model = causal_model
        self.checker = checker
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
        
        # Create all layers list
        layers = list(range(self.num_layers))
        
        # Create single PatchResidualStream experiment with all layers and positions
        experiment = PatchResidualStream(
            pipeline=self.pipeline,
            causal_model=self.causal_model,
            layers=layers,
            token_positions=token_positions,
            checker=self.checker,
            featurizers=None,  # Use default featurizer
            loss_and_metric_fn=self.loss_and_metric_fn,
            config={"batch_size": 1, "raw_outputs":True},  # Single example
        )
        
        # Run the experiment once with all locations
        results = experiment.perform_interventions(
            {"tracing_example": dataset}, 
            target_variables_list=[["raw_output"]],  # Use raw_output for binary accuracy
        )

        experiment.plot_heatmaps(
            results=results, 
            target_variables=["raw_output"], 
            save_path=save_path,  # Display interactively
            average_counterfactuals=False,  # No averaging since we only have one example
        )
        
        # Also plot the raw outputs
        self.plot_raw_outputs(results, token_positions, save_path=save_path)
        
        # Clean up memory after the experiment
        experiment._clean_memory()
        del experiment
        return results
    
    def plot_raw_outputs(self, results: Dict, token_positions: List[TokenPosition], save_path: Optional[str] = None) -> None:
        """
        Display the raw generated outputs in a grid format with color coding.
        
        This method creates a visualization showing the actual tokens generated under
        intervention at each (layer, position) combination. Unlike the heatmap which
        shows accuracy scores, this displays the raw text outputs with cells colored
        based on the output frequency (top 5 most frequent outputs get unique colors).
        
        Args:
            results: Results dictionary from perform_interventions (must have raw_outputs preserved)
            token_positions: List of TokenPosition objects used in the experiment
            save_path: Optional path to save the plot. If None, displays interactively.
        
        Raises:
            ValueError: If raw_outputs are not found in the results
        """
        
        # Get dimensions for the grid
        layers = list(range(self.num_layers))
        positions = [tp.id for tp in token_positions]
        
        # Create figure with appropriate size - add extra width for legend
        fig_width = max(16, len(positions) * 2 + 4)  # Added space for legend
        fig_height = max(8, len(layers) * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Hide axes
        ax.set_xlim(0, len(positions))
        ax.set_ylim(0, len(layers))
        ax.axis('off')
        
        # Extract raw outputs and convert to text
        dataset_name = "tracing_example"  # This is the dataset name used in run()
        
        # Create a matrix to store text outputs
        text_outputs = [['' for _ in positions] for _ in layers]
        
        # First pass: collect all outputs to count frequencies
        output_counter = Counter()
        
        # Process results to extract and count raw outputs
        if dataset_name in results["dataset"]:
            for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
                if "raw_outputs" not in unit_data:
                    raise ValueError("raw_outputs not found in results. Ensure config['raw_outputs']=True when running the experiment.")
                
                if "metadata" in unit_data and unit_data["raw_outputs"]:
                    # Get the raw output and decode it
                    raw_output = unit_data["raw_outputs"][0][0] if unit_data["raw_outputs"][0] else None
                    if raw_output is not None:
                        decoded_text = self.pipeline.dump(raw_output, is_logits=False)
                        if isinstance(decoded_text, list):
                            decoded_text = decoded_text[0]
                        output_counter[decoded_text] += 1
        
        # Define light colors for the top 5 most frequent outputs
        light_colors = [
            '#FFFFE0',  # Light yellow
            '#FFE4E1',  # Light red/pink
            '#E0FFE0',  # Light green
            '#E0E0FF',  # Light blue
            '#F5DEB3',  # Light brown/wheat
        ]
        
        # Get top 5 most frequent outputs and create color mapping
        top_outputs = [output for output, _ in output_counter.most_common(5)]
        output_color_map = {output: light_colors[i] for i, output in enumerate(top_outputs)}
        default_color = 'white'  # For outputs not in top 5
        
        # Second pass: process results and store outputs
        if dataset_name in results["dataset"]:
            for unit_str, unit_data in results["dataset"][dataset_name]["model_unit"].items():
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
                    
                    # Get the raw output and decode it
                    if unit_data["raw_outputs"]:
                        # raw_outputs is a list of lists, get the first output
                        raw_output = unit_data["raw_outputs"][0][0] if unit_data["raw_outputs"][0] else None
                        
                        if raw_output is not None:
                            # Use pipeline.dump to decode the output
                            decoded_text = self.pipeline.dump(raw_output, is_logits=False)
                            if isinstance(decoded_text, list):
                                decoded_text = decoded_text[0]
                            
                            text_outputs[layer][pos_idx] = decoded_text
        
        # Create the table/grid
        cell_height = 0.8 / len(layers)
        cell_width = 0.9 / len(positions)
        
        for i, layer in enumerate(layers):
            for j, pos in enumerate(positions):
                # Calculate cell position
                x = 0.05 + j * cell_width
                y = 0.1 + i * cell_height
                
                # Get the text output for this cell to determine color
                text = text_outputs[layer][j]
                
                # Determine the background color based on the output
                cell_color = output_color_map.get(text, default_color)
                
                # Create a rectangle for the cell with appropriate color
                rect = mpatches.Rectangle((x, y), cell_width, cell_height,
                                        linewidth=1, edgecolor='black',
                                        facecolor=cell_color, transform=ax.transAxes)
                ax.add_patch(rect)
                # Check if text is just whitespace and add quotes if so
                if not text or text.strip() == '':
                    text = f'"{text}"'
                
                # Wrap text if it's too long
                max_chars_per_line = int(cell_width * 100)  # Rough estimate
                if len(text) > max_chars_per_line:
                    # Simple text wrapping
                    wrapped_lines = []
                    words = text.split()
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 > max_chars_per_line:
                            wrapped_lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            current_line.append(word)
                            current_length += len(word) + 1
                    
                    if current_line:
                        wrapped_lines.append(' '.join(current_line))
                    
                    text = '\n'.join(wrapped_lines[:3])  # Limit to 3 lines
                    if len(wrapped_lines) > 3:
                        text += '...'
                
                ax.text(x + cell_width/2, y + cell_height/2, text,
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=42, wrap=True)
        
        # Add labels
        # Layer labels (y-axis)
        for i, layer in enumerate(layers):
            ax.text(0.02, 0.1 + i * cell_height + cell_height/2, f'L{layer}',
                   ha='right', va='center', transform=ax.transAxes, fontsize=48)
        
        # Position labels (x-axis)
        for j, pos in enumerate(positions):
            # Truncate long position labels
            label = str(pos)
            if len(label) > 15:
                label = label[:12] + '...'
            
            ax.text(0.05 + j * cell_width + cell_width/2, 0.05, label,
                   ha='center', va='top', transform=ax.transAxes, fontsize=42,
                   rotation=45 if len(label) > 5 else 0)
        
        # Add title
        ax.text(0.5, 0.98, f'Raw Outputs Under Intervention - Task: {results["task_name"]}',
               ha='center', va='top', transform=ax.transAxes, fontsize=54, weight='bold')
        
        # Add axis labels
        ax.text(0.5, -0.1, 'Token Position', ha='center', va='bottom',
               transform=ax.transAxes, fontsize=48)
        ax.text(-0.1, 0.5, 'Layer', ha='center', va='center', rotation=90,
               transform=ax.transAxes, fontsize=48)
        
        # Add legend to the right side if there are colored outputs
        if output_color_map:
            legend_x = 1.1  # Further right, outside the main plot area
            legend_y_start = 0.7  # Starting y position for legend
            legend_spacing = 0.08  # Spacing between legend items
            
            # Add legend title
            ax.text(legend_x, legend_y_start + legend_spacing, 'Top 5 Outputs:',
                   ha='left', va='top', transform=ax.transAxes, fontsize=44, weight='bold')
            
            # Add legend items
            for i, (output, color) in enumerate(output_color_map.items()):
                y_pos = legend_y_start - i * legend_spacing
                
                # Create small colored rectangle
                legend_rect = mpatches.Rectangle((legend_x, y_pos - 0.03), 0.04, 0.05,
                                               linewidth=1, edgecolor='black',
                                               facecolor=color, transform=ax.transAxes)
                ax.add_patch(legend_rect)
                
                # Add text label - truncate if too long
                label_text = output if len(output) <= 20 else output[:17] + '...'
                # Show quotes for empty/whitespace strings
                if not output or output.strip() == '':
                    label_text = f'"{output}"'
                
                ax.text(legend_x + 0.05, y_pos, label_text,
                       ha='left', va='center', transform=ax.transAxes, fontsize=40)
        
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

        