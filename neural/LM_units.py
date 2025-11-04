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

import os
import json
from typing import List, Union

from neural.model_units import (
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
    is_original : bool
        Whether this indexer is for original inputs (True) or counterfactual inputs (False).
        Default is True for backward compatibility.
    """

    def __init__(self, indexer, pipeline: LMPipeline, is_original: bool = True, **kwargs):
        super().__init__(indexer, **kwargs)
        self.pipeline = pipeline
        self.is_original = is_original

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

        pad_token_id = self.pipeline.tokenizer.pad_token_id

        return "".join(
            f"**{self.pipeline.tokenizer.decode(t)}**" if i in highlight else self.pipeline.tokenizer.decode(t)
            for i, t in enumerate(ids) if t != pad_token_id
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


def get_substring_token_ids(
    text: str,
    substring: str,
    pipeline: LMPipeline,
    add_special_tokens: bool = False,
    occurrence: int = 0,
    strict: bool = False
) -> List[int]:
    """Return token position indices for tokens that overlap with a substring.

    Given a text and a substring that occurs within it, returns the list of
    token position indices corresponding to tokens that overlap with the substring.
    When the substring boundaries fall in the middle of a token, that token is
    included in the result.

    Parameters
    ----------
    text : str
        The full input text to tokenize.
    substring : str
        A substring that occurs within `text`. Must be present in the text.
    pipeline : LMPipeline
        The pipeline containing the tokenizer to use.
    add_special_tokens : bool, optional
        Whether to add special tokens (BOS/EOS) during tokenization. Default is False.
    occurrence : int, optional
        Which occurrence of the substring to use (0-indexed). Default is 0 (first occurrence).
    strict : bool, optional
        If True, raises ValueError when multiple occurrences exist. Default is False.

    Returns
    -------
    List[int]
        A list of token position indices (0-indexed) for tokens overlapping the substring.

    Raises
    ------
    ValueError
        If substring is empty, text is empty, substring is not found, the specified
        occurrence doesn't exist, or (when strict=True) multiple occurrences exist.

    Examples
    --------
    >>> text = "The sum of 5 and 5 is 10"
    >>> substring = "5"
    >>> # Get first occurrence (default)
    >>> indices = get_substring_token_ids(text, substring, pipeline)
    >>> # Get second occurrence explicitly
    >>> indices = get_substring_token_ids(text, substring, pipeline, occurrence=1)
    >>> # Fail if ambiguous
    >>> indices = get_substring_token_ids(text, substring, pipeline, strict=True)  # Raises!

    Notes
    -----
    - This function is inclusive: any token with any character overlap gets included.
    - Handles tokenizer-specific behaviors like leading space encoding (e.g., Ġ in GPT-2).
    - When multiple occurrences exist and strict=False, uses the first by default.
    """
    # Validation
    if not text:
        raise ValueError("Text cannot be empty")
    if not substring:
        raise ValueError("Substring cannot be empty")
    if substring not in text:
        raise ValueError(f"Substring '{substring}' not found in text")
    if occurrence < 0:
        raise ValueError(f"occurrence must be non-negative, got {occurrence}")
    
    # Find all occurrences
    occurrences = []
    start = 0
    while True:
        pos = text.find(substring, start)
        if pos == -1:
            break
        occurrences.append(pos)
        start = pos + 1

    num_occurrences = len(occurrences)

    # Check for ambiguity in strict mode
    if strict and num_occurrences > 1:
        raise ValueError(
            f"Found {num_occurrences} occurrences of '{substring}' in the text. "
            f"Please either:\n"
            f"  1. Use more specific context to make substring unique\n"
            f"  2. Specify which occurrence with occurrence parameter (0 to {num_occurrences-1})\n"
            f"  3. Set strict=False to use first occurrence (default behavior)"
        )

    # Validate occurrence parameter
    if occurrence >= num_occurrences:
        raise ValueError(
            f"Requested occurrence {occurrence} but only found {num_occurrences} "
            f"occurrence(s) of '{substring}'"
        )

    # Find substring position in original text
    substring_start = occurrences[occurrence]
    substring_end = substring_start + len(substring)

    # Tokenize the text
    input_dict = {"raw_input": text}
    token_ids = pipeline.load(input_dict, add_special_tokens=add_special_tokens)["input_ids"][0]
    token_ids_list = token_ids.tolist()

    # Build character-to-token mapping by decoding each token
    # and tracking its position in the reconstructed text
    char_to_token = []
    current_pos = 0

    for token_idx, token_id in enumerate(token_ids_list):
        # Decode individual token
        token_str = pipeline.tokenizer.decode([token_id], skip_special_tokens=False)

        # Handle potential leading space normalization
        # Some tokenizers store spaces as part of the token (e.g., Ġ in GPT-2)
        # but decode() might render them as actual spaces
        token_length = len(token_str)

        # Track which characters belong to this token
        for _ in range(token_length):
            if current_pos < len(text):
                char_to_token.append(token_idx)
                current_pos += 1

    # For tokenizers that don't perfectly reconstruct (common with BPE),
    # we need a more robust approach: match decoded tokens to original text
    # Build a mapping by decoding progressively
    char_to_token = {}
    reconstructed = ""

    for token_idx, token_id in enumerate(token_ids_list):
        # Decode up to and including this token
        decoded_so_far = pipeline.tokenizer.decode(
            token_ids_list[:token_idx + 1],
            skip_special_tokens=not add_special_tokens
        )

        # Mark the new characters as belonging to this token
        start_pos = len(reconstructed)
        end_pos = len(decoded_so_far)

        for char_pos in range(start_pos, end_pos):
            char_to_token[char_pos] = token_idx

        reconstructed = decoded_so_far

    # Find which tokens overlap with the substring in the reconstructed text
    # We need to map the original text positions to reconstructed text positions
    # For most tokenizers, they should match, but let's be safe

    # Find substring in reconstructed text
    try:
        reconstructed_start = reconstructed.index(substring)
        reconstructed_end = reconstructed_start + len(substring)
    except ValueError:
        # Substring might not appear identically after tokenization roundtrip
        # Fall back to approximate matching
        # This can happen with special whitespace handling
        reconstructed_start = None
        for i in range(len(reconstructed)):
            if reconstructed[i:i+len(substring)] == substring:
                reconstructed_start = i
                break

        if reconstructed_start is None:
            # Try matching with normalized whitespace
            import re
            normalized_substring = re.sub(r'\s+', ' ', substring.strip())
            normalized_reconstructed = re.sub(r'\s+', ' ', reconstructed)

            if normalized_substring in normalized_reconstructed:
                reconstructed_start = normalized_reconstructed.index(normalized_substring)
                reconstructed_end = reconstructed_start + len(normalized_substring)
            else:
                raise ValueError(
                    f"Substring '{substring}' not found in tokenizer-reconstructed text. "
                    f"Original: '{text}', Reconstructed: '{reconstructed}'"
                )
        else:
            reconstructed_end = reconstructed_start + len(substring)

    # Collect all unique token indices that overlap with the substring
    overlapping_tokens = set()
    for char_pos in range(reconstructed_start, reconstructed_end):
        if char_pos in char_to_token:
            overlapping_tokens.add(char_to_token[char_pos])

    # Return sorted list of token indices
    return sorted(list(overlapping_tokens))


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
        # Include component_type in the UID to distinguish between block_input (embeddings/layer -1)
        # and block_output (normal layers) when they target the same layer number
        uid = f"ResidualStream(Layer-{layer},{component_type},Token-{tok_id})"

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

    def index_component(self, input, batch=False, **kwargs):
        """Return indices for *input* by delegating to wrapped function."""
        if batch:
            return [[[self.head]]*len(input), [self.component.index(x, **kwargs) for x in input]]
        return [[[self.head]], [self.component.index(input, **kwargs)]]
    