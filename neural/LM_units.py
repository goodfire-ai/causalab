"""
LM_units.py
===========
Helpers that bind the *core* component / featurizer abstractions from
`model_units.py` to language-model pipelines.  They let you refer to:

* A **ResidualStream** slice: hidden state of one or more token positions.
* An **AttentionHead** value: output for a single attention head.

All helpers inherit from :class:`model_units.AtomicModelUnit`, so they carry
the full featurizer + feature indexing machinery.
"""

import sys, os, json
from pathlib import Path
from typing import List, Union

sys.path.append(str(Path(__file__).resolve().parent.parent))  # non-pkg path hack

from neural.model_units import (  # noqa: E402  (import after path hack)
    AtomicModelUnit,
    Component,
    StaticComponent,
    ComponentIndexer,
    Featurizer,
)
from neural.pipeline import LMPipeline


# --------------------------------------------------------------------------- #
#  Token-level helper                                                         #
# --------------------------------------------------------------------------- #
class TokenPosition(ComponentIndexer):
    """Dynamic indexer: returns position(s) of interest for a prompt.

    Attributes
    ----------
    pipeline :
        The :class:`neural.pipeline.LMPipeline` supplying the tokenizer.
    """

    def __init__(self, indexer, pipeline: LMPipeline, **kwargs):
        super().__init__(indexer, **kwargs)
        self.pipeline = pipeline

    # ------------------------------------------------------------------ #
    def highlight_selected_token(self, input: dict) -> str:
        """Return *prompt* with selected token(s) wrapped in ``**bold**``.

        The method tokenizes *prompt*, calls self.index to obtain the
        positions, then re-assembles a detokenised string with the
        selected token(s) wrapped in ``**bold**``.  The rest of the
        prompt is unchanged.

        Note that whitespace handling may be approximate for tokenizers 
        that encode leading spaces as special glyphs (e.g. ``Ġ``).
        """
        ids = self.pipeline.load(input)["input_ids"][0]
        highlight = self.index(input)
        highlight = highlight if isinstance(highlight, list) else [highlight]

        return "".join(
            f"**{self.pipeline.tokenizer.decode(t)}**" if i in highlight else self.pipeline.tokenizer.decode(t)
            for i, t in enumerate(ids)
        )


# Convenience indexers
def get_last_token_index(input: dict, pipeline: LMPipeline):
    """Return a one-element list containing the *last* token index."""
    ids = list(pipeline.load(input)["input_ids"][0])
    return [len(ids) - 1]

def get_all_tokens(input: dict, pipeline: LMPipeline):
    """Return a list containing all token indices."""
    ids = list(pipeline.load(input)["input_ids"][0])
    return list(range(len(ids)))


# --------------------------------------------------------------------------- #
#  LLM-specific AtomicModelUnits                                              #
# --------------------------------------------------------------------------- #
class ResidualStream(AtomicModelUnit):
    """Residual-stream slice at *layer* for given token position(s)."""

    def __init__(
        self,
        layer: int,
        token_indices: Union[List[int], ComponentIndexer],
        *,
        featurizer: Featurizer | None = None,
        shape=None,
        feature_indices=None,
        target_output: bool = False,
    ):
        component_type = "block_output" if target_output else "block_input"
        self.token_indices = token_indices
        tok_id = token_indices.id if isinstance(token_indices, ComponentIndexer) else token_indices
        uid = f"ResidualStream(Layer-{layer},Token-{tok_id})"

        unit = "pos"
        if isinstance(token_indices, list):
            component = StaticComponent(layer, component_type, token_indices, unit)
        else:
            component = Component(layer, component_type, token_indices, unit)

        super().__init__(
            component=component,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )

    @classmethod
    def load_modules(cls, base_name: str, dir: str, token_positions): 
        # Extract layer number plus one additonal 
        # character after "Layer" for the _ or :
        layer_start = base_name.find("Layer") + 6 
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])
        
        # Extract token position plus one additional 
        # character after "Token" for the _ or :
        token_start = base_name.find("Token") + 6
        token_end = base_name.find(")", token_start)
        tok_id = base_name[token_start:token_end]
        # Find the element of the list with a .id that matches tok_id
        if isinstance(token_positions, list):
            token_indices = next((tp for tp in token_positions if tp.id == tok_id), None)
            if token_indices is None:
                raise ValueError(f"Token position with id '{tok_id}' not found in provided list.")
        
        
        # Check if all required files exist
        base_path = os.path.join(dir, base_name)
        featurizer_path = base_path + "_featurizer"
        inverse_featurizer_path = base_path + "_inverse_featurizer"
        indices_path = base_path + "_indices"

        if not all(os.path.exists(p) for p in [featurizer_path, inverse_featurizer_path]):
            print(f"Missing featurizer or inverse_featurizer files for {base_name}")
        
        # Load the featurizer
        featurizer = Featurizer.load_modules(base_path)
        
        # Load and set indices if they exist
        try:
            with open(indices_path, 'r') as f:
                indices = json.load(f)
            if indices is not None:
                featurizer.set_feature_indices(indices)
        except Exception as e:
            print(f"Warning: Could not load indices for {base_name}: {e}")
        return cls(
            layer=layer,
            token_indices=token_indices,
            featurizer=featurizer,
            )


class AttentionHead(AtomicModelUnit):
    """Attention-head value stream at (*layer*, *head*) for token position(s)."""

    def __init__(
        self,
        layer: int,
        head: int,
        token_indices: Union[List[int], ComponentIndexer],
        *,
        featurizer: Featurizer | None = None,
        shape=None,
        feature_indices=None,
        target_output: bool = True,
    ):
        self.head = head
        component_type = (
            "head_attention_value_output" if target_output else "head_attention_value_input"
        )

        tok_id = token_indices.id if isinstance(token_indices, ComponentIndexer) else token_indices
        uid = f"AttentionHead(Layer-{layer},Head-{head},Token-{tok_id})"

        unit = "h.pos"

        if isinstance(token_indices, list):
            component = StaticComponent(layer, component_type, token_indices, unit)
        else:
            component = Component(layer, component_type, token_indices, unit)
        


        super().__init__(
            component=component,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )

    @classmethod
    def load_modules(cls, base_name: str, submission_folder_path: str, token_positions):
        """Load AttentionHead from a base name and submission folder path."""
        # Check if the base name starts with "AttentionHead"
        # Extract layer number plus 
        layer_start = base_name.find("Layer") + 6
        layer_end = base_name.find(",", layer_start)
        layer = int(base_name[layer_start:layer_end])
        
        # Extract head number
        head_start = base_name.find(",Head") + 6
        head_end = base_name.find(",", head_start)
        head = int(base_name[head_start:head_end])
        
        # Check if all required files exist
        base_path = os.path.join(submission_folder_path, base_name)
        featurizer_path = base_path + "_featurizer"
        inverse_featurizer_path = base_path + "_inverse_featurizer"
        indices_path = base_path + "_indices"
        
        if not all(os.path.exists(p) for p in [featurizer_path, inverse_featurizer_path]):
            print(f"Missing featurizer or inverse_featurizer files for {base_name}")
        
        # Load the featurizer
        featurizer = Featurizer.load_modules(base_path)
        
        # Load and set indices if they exist
        try:
            with open(indices_path, 'r') as f:
                indices = json.load(f)
            if indices is not None:
                featurizer.set_feature_indices(indices)
        except Exception as e:
            print(f"Warning: Could not load indices for {base_name}: {e}")
        
        return cls(
            layer=layer,
            head=head,
            token_indices=token_positions,
            featurizer=featurizer,
        )

    # ------------------------------------------------------------------ #

    def index_component(self, input, batch=False):
        """Return indices for *input* by delegating to wrapped function."""
        if batch:
            return [[[self.head]]*len(input), [self.component.index(x) for x in input]]
        return [[[self.head]], [self.component.index(input)]]
    