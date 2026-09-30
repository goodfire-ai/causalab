"""
Template definitions and fill functions for the task.
"""

import random
from typing import cast

from .config import LETTERS, NUM_ICL_EXAMPLES, PATTERNS, PROMPT_MODE

# --- Code-mode templates (Python function definitions) ---
CODE_TEMPLATES = [
    "def double_equality(a, b, c, d):\n"
    "    x = (a == b)\n"
    "    y = (c == d)\n"
    "    return x == y",
    "def check_pairs(a, b, c, d):\n"
    "    first = (a == b)\n"
    "    second = (c == d)\n"
    "    return first == second",
    "def compare_pairs(p, q, r, s):\n"
    "    left = (p == q)\n"
    "    right = (r == s)\n"
    "    return left == right",
]

# For non-code modes the "template" variable is a single sentinel —
# the causal model still needs at least one value for sampling.
_ALGORITHMIC_TEMPLATES = ["algorithmic"]
_MINIMAL_FUNCTION_TEMPLATES = ["minimal_function"]

# TEMPLATES is what the rest of the codebase (causal_models, counterfactuals) imports.
# PROMPT_MODE is a module-level constant; pyright narrows to a single Literal
# so the other branches look "unreachable". They are still valid configurations
# when PROMPT_MODE is edited in config.py — keep the three-way dispatch.
_prompt_mode = cast(str, PROMPT_MODE)
if _prompt_mode == "algorithmic":
    TEMPLATES = _ALGORITHMIC_TEMPLATES
elif _prompt_mode == "minimal_function":
    TEMPLATES = _MINIMAL_FUNCTION_TEMPLATES  # pyright: ignore[reportConstantRedefinition]
else:
    TEMPLATES = CODE_TEMPLATES  # pyright: ignore[reportConstantRedefinition]


def get_func_name(template: str) -> str:
    """Extract function name from template code."""
    return template.split("def ")[1].split("(")[0]


def _sample_pattern_values(pattern: str, rng) -> tuple[str, str, str, str]:
    """Sample letter values according to a pattern."""
    if pattern == "AABB":
        a = rng.choice(LETTERS)
        c = rng.choice(LETTERS)
        return (a, a, c, c)
    elif pattern == "ABCD":
        chosen = rng.sample(LETTERS, 4)
        return (chosen[0], chosen[1], chosen[2], chosen[3])
    elif pattern == "ABCC":
        a = rng.choice(LETTERS)
        b = rng.choice([letter for letter in LETTERS if letter != a])
        c = rng.choice(LETTERS)
        return (a, b, c, c)
    elif pattern == "AABC":
        a = rng.choice(LETTERS)
        c = rng.choice(LETTERS)
        d = rng.choice([letter for letter in LETTERS if letter != c])
        return (a, a, c, d)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def _generate_icl_lines(n: int, fmt: str, rng) -> str:
    """Generate *n* balanced ICL lines in the given format.

    fmt: "algorithmic" → ``A A B B: 1``
         "minimal_function" → ``f(A,A,B,B)=1``
    """
    lines: list[str] = []
    per_pattern = n // len(PATTERNS)
    for pattern in PATTERNS:
        for _ in range(per_pattern):
            v1, v2, v3, v4 = _sample_pattern_values(pattern, rng)
            out = 1 if (v1 == v2) == (v3 == v4) else 0
            if fmt == "minimal_function":
                lines.append(f"f({v1},{v2},{v3},{v4})={out}")
            else:
                lines.append(f"{v1} {v2} {v3} {v4}: {out}")
    rng.shuffle(lines)
    return "\n".join(lines)


def generate_icl_examples(func_name: str, rng, n: int = NUM_ICL_EXAMPLES) -> str:
    """Generate n balanced ICL examples for code mode."""
    examples = []
    per_pattern = n // len(PATTERNS)

    for pattern in PATTERNS:
        for _ in range(per_pattern):
            a, b, c, d = _sample_pattern_values(pattern, rng)
            output = 1 if (a == b) == (c == d) else 0
            line = (
                f'The function call {func_name}("{a}", "{b}", "{c}", "{d}") '
                f"returns the value {output}"
            )
            examples.append(line)

    rng.shuffle(examples)
    return "\n".join(examples)


def fill_template(
    template: str, var_1: str, var_2: str, var_3: str, var_4: str, *, seed: int
) -> str:
    """Build the full prompt respecting PROMPT_MODE."""
    rng = random.Random(seed)
    if template == "algorithmic":
        icl = _generate_icl_lines(NUM_ICL_EXAMPLES, "algorithmic", rng)
        query = f"{var_1} {var_2} {var_3} {var_4}: "
        return icl + "\n" + query
    elif template == "minimal_function":
        icl = _generate_icl_lines(NUM_ICL_EXAMPLES, "minimal_function", rng)
        query = f"f({var_1},{var_2},{var_3},{var_4})="
        return icl + "\n" + query
    else:
        # code mode
        func_name = get_func_name(template)
        icl_lines = generate_icl_examples(func_name, rng)
        test_query = (
            f'The function call {func_name}("{var_1}", "{var_2}", "{var_3}", "{var_4}") '
            f"returns the value "
        )
        return template + "\n" + icl_lines + "\n" + test_query
