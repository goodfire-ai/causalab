"""
Token position functions for the entity binding task.

Provides functions to locate specific tokens in entity binding prompts,
such as entity tokens in statement and question regions.

Uses prefix-tokenization to locate entity positions accurately,
handling the case where entities appear in both the statement and the question.
"""

from typing import Any, Dict, List, Optional

from causalab.tasks.token_positions import LMPipeline
from causalab.tasks.token_positions import (
    TokenPosition,
    build_token_positions,
)

from .config import EntityBindingTaskConfig


def _build_values_dict(
    input_sample: Dict[str, Any],
    config: EntityBindingTaskConfig,
) -> Dict[str, str]:
    """Build the values dict for filling the mega template."""
    active_groups = input_sample.get("active_groups")
    entities_per_group = input_sample.get("entities_per_group")
    query_indices = input_sample.get("query_indices")

    values = {}
    for g in range(active_groups):
        for e in range(entities_per_group):
            entity = input_sample.get(f"entities[{g},{e}]")
            values[f"g{g}_e{e}"] = entity if entity is not None else f"MISSING_{g}_{e}"

    values["query_entity"] = input_sample.get(f"queries[{query_indices[0]}]")
    for e in range(entities_per_group):
        role_name = config.entity_roles.get(e, f"entity{e}")
        query_val = input_sample.get(f"queries[{e}]")
        values[role_name] = query_val if query_val is not None else ""

    return values


def get_entity_token_positions(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    config: EntityBindingTaskConfig,
    group_idx: int,
    entity_idx: int,
    token_idx: Optional[int] = None,
) -> List[int]:
    """
    Get token positions for a statement entity using prefix tokenization.

    Handles entities appearing multiple times (statement and question) by
    targeting only the first (statement) occurrence.

    Args:
        input_sample: Input sample with entity values and raw_input
        pipeline: Pipeline with tokenizer
        config: Task configuration
        group_idx: Entity group index (0-indexed)
        entity_idx: Entity index within group (0-indexed)
        token_idx: If specified, return only the token at this index.
                   Supports negative indexing (-1 for last token).

    Returns:
        List of token position indices
    """
    active_groups = input_sample.get("active_groups")
    query_indices = input_sample.get("query_indices")
    answer_index = input_sample.get("answer_index")

    mega_template_str = config.build_mega_template(
        active_groups, query_indices, answer_index
    )
    values = _build_values_dict(input_sample, config)
    filled = config.fill_template(mega_template_str, values)

    # Find the variable name in the mega template to locate the entity
    var_name = f"g{group_idx}_e{entity_idx}"

    # Build a partial template with everything up to (but not including) this variable
    # to determine the character position of the entity in the filled string
    char_start = filled.find(values[var_name])
    if char_start == -1:
        raise ValueError(
            f"Entity value '{values[var_name]}' not found in filled template '{filled}'"
        )
    char_end = char_start + len(values[var_name])

    # Tokenize prefix up to end of entity (with special tokens to match pipeline)
    tokenizer = pipeline.tokenizer
    prefix_with = tokenizer.encode(filled[:char_end], add_special_tokens=True)
    prefix_before = tokenizer.encode(filled[:char_start], add_special_tokens=True)

    entity_token_start = len(prefix_before)
    entity_token_end = len(prefix_with)
    positions = list(range(entity_token_start, entity_token_end))

    if not positions:
        raise ValueError(
            f"No tokens found for entity at group {group_idx}, position {entity_idx}"
        )

    if token_idx is not None:
        if token_idx < -len(positions) or token_idx >= len(positions):
            raise ValueError(
                f"token_idx {token_idx} out of range for entity with {len(positions)} tokens"
            )
        positions = [positions[token_idx]]

    return positions


def get_question_entity_token_positions(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    config: EntityBindingTaskConfig,
    entity_idx: Optional[int] = None,
    role_name: Optional[str] = None,
    token_idx: Optional[int] = None,
) -> List[int]:
    """
    Get token positions for a question entity.

    Args:
        input_sample: Input sample
        pipeline: Pipeline with tokenizer
        config: Task configuration
        entity_idx: Entity index (0-indexed). Mutually exclusive with role_name.
        role_name: Role name from config.entity_roles. Mutually exclusive with entity_idx.
        token_idx: If specified, return only the token at this index.
    """
    if entity_idx is None and role_name is None:
        raise ValueError("Must specify either entity_idx or role_name")
    if entity_idx is not None and role_name is not None:
        raise ValueError("Cannot specify both entity_idx and role_name")

    if entity_idx is None:
        for idx, name in config.entity_roles.items():
            if name == role_name:
                entity_idx = idx
                break
        if entity_idx is None:
            raise ValueError(
                f"Role '{role_name}' not found in config.entity_roles: {config.entity_roles}"
            )

    query_indices = input_sample.get("query_indices")
    if entity_idx not in query_indices:
        raise ValueError(
            f"Entity index {entity_idx} is not in query_indices {query_indices}. "
            f"Only entities at positions {query_indices} appear in the question."
        )

    active_groups = input_sample.get("active_groups")
    answer_index = input_sample.get("answer_index")

    mega_template_str = config.build_mega_template(
        active_groups, query_indices, answer_index
    )
    values = _build_values_dict(input_sample, config)
    filled = config.fill_template(mega_template_str, values)

    resolved_name = config.entity_roles.get(entity_idx, f"entity{entity_idx}")
    entity_value = values.get(resolved_name, "")

    # Find LAST occurrence (question comes after statement)
    char_start = filled.rfind(entity_value)
    if char_start == -1:
        raise ValueError(
            f"Variable '{resolved_name}' value '{entity_value}' not found in '{filled}'"
        )
    char_end = char_start + len(entity_value)

    tokenizer = pipeline.tokenizer
    prefix_with = tokenizer.encode(filled[:char_end], add_special_tokens=True)
    prefix_before = tokenizer.encode(filled[:char_start], add_special_tokens=True)

    positions = list(range(len(prefix_before), len(prefix_with)))

    if not positions:
        raise ValueError(f"No tokens found for entity_idx {entity_idx}")

    if token_idx is not None:
        if token_idx < -len(positions) or token_idx >= len(positions):
            raise ValueError(
                f"token_idx {token_idx} out of range for entity with {len(positions)} tokens"
            )
        positions = [positions[token_idx]]

    return positions


def get_statement_entity_token_positions(
    input_sample: Dict[str, Any],
    pipeline: LMPipeline,
    config: EntityBindingTaskConfig,
    group_idx: int,
    entity_idx: Optional[int] = None,
    role_name: Optional[str] = None,
    token_idx: Optional[int] = None,
) -> List[int]:
    """
    Get token positions for a statement entity with role_name support.

    Args:
        input_sample: Input sample
        pipeline: Pipeline with tokenizer
        config: Task configuration
        group_idx: Entity group index (0-indexed)
        entity_idx: Entity index within group. Mutually exclusive with role_name.
        role_name: Role name from config.entity_roles. Mutually exclusive with entity_idx.
        token_idx: If specified, return only the token at this index.
    """
    if entity_idx is None and role_name is None:
        raise ValueError("Must specify either entity_idx or role_name")
    if entity_idx is not None and role_name is not None:
        raise ValueError("Cannot specify both entity_idx and role_name")

    if entity_idx is None:
        for idx, name in config.entity_roles.items():
            if name == role_name:
                entity_idx = idx
                break
        if entity_idx is None:
            raise ValueError(
                f"Role '{role_name}' not found in config.entity_roles: {config.entity_roles}"
            )

    return get_entity_token_positions(
        input_sample=input_sample,
        pipeline=pipeline,
        config=config,
        group_idx=group_idx,
        entity_idx=entity_idx,
        token_idx=token_idx,
    )


def create_token_positions(
    pipeline: LMPipeline,
    template: str,
    config: Optional[EntityBindingTaskConfig] = None,
) -> Dict[str, TokenPosition]:
    """
    Create token positions for the entity binding task.

    Returns a dict mapping position names to TokenPosition instances.

    Args:
        pipeline: The LMPipeline for tokenization
        template: Template string (used for declarative positions)
        config: Task configuration (uses default love config if not provided)

    Returns:
        Dict mapping position name → TokenPosition instance.
    """
    if config is None:
        from .config import create_sample_love_config

        config = create_sample_love_config()

    token_position_specs: dict = {
        "last": {"type": "index", "position": -1},
    }

    positions = build_token_positions(token_position_specs, template, pipeline)

    # Add custom positions for each entity in the statement (its last token).
    for g in range(config.max_groups):
        for e in range(config.max_entities_per_group):
            key = f"g{g}_e{e}_last"
            positions[key] = TokenPosition(
                indexer=lambda x, g=g, e=e, c=config: (
                    get_entity_token_positions(x, pipeline, c, g, e, token_idx=-1)
                ),
                pipeline=pipeline,
                id=key,
            )

    return positions
