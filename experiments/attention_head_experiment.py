import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Callable, Tuple

from experiments.intervention_experiment import *
from neural.LM_units import *
from neural.model_units import *
from neural.pipeline import LMPipeline
from causal.causal_model import CausalModel

from experiments.pyvene_core import _prepare_intervenable_inputs


class PatchAttentionHeads(InterventionExperiment):
    """
    Generic experiment for analyzing attention head interventions in language models.
    
    This class enables interventions on specific attention heads at various positions
    in the input sequence. By modifying attention head outputs and observing the effect
    on model outputs, we can identify which heads are responsible for specific behaviors.
    
    Attributes:
        layer_head_list (List[Tuple[int, int]]): List of (layer, head) tuples to intervene on
        featurizers (Dict): Mapping of (layer, head, position) tuples to Featurizer instances
        token_positions (List[TokenPosition]): Token positions to analyze
    """
    
    def __init__(self,
                 pipeline: LMPipeline,
                 causal_model: CausalModel,
                 layer_head_list: List[Tuple[int, int]],
                 token_positions: List[TokenPosition],
                 checker: Callable,
                 featurizers: Dict[Tuple[int, int, str], Featurizer] = None,
                 config: Dict = None,
                 **kwargs):
        """
        Initialize PatchAttentionHeads for analyzing attention head interventions.
        
        Args:
            pipeline: LMPipeline object for model execution
            causal_model: CausalModel object containing the task
            layers: List of layer indices (kept for compatibility but not used directly)
            layer_head_list: List of (layer, head) tuples specifying which heads to intervene on
            token_positions: List of TokenPosition objects for token positions
            checker: Function to evaluate output accuracy
            featurizers: Dict mapping (layer, head, position.id) to Featurizer instances
            **kwargs: Additional configuration options
        """
        self.layer_head_list = layer_head_list
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


        model_units = []
        for layer, head in layer_head_list:
            # Get or create featurizer for this head
            featurizer_key = (layer, head)
            featurizer = self.featurizers.get(
                featurizer_key,
                Featurizer(n_features=head_size, **featurizer_kwargs)
            )
            
            # Create model unit list for this head
            for token_position in token_positions:
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
        model_units_lists = [[model_units]]
            
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
        if "loss_and_metric_fn" in self.config:
            self.loss_and_metric_fn = self.config["loss_and_metric_fn"]
        
        self.token_positions = token_positions