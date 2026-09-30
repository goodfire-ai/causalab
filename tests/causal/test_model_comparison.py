"""Test causal predictions and scores from saved intervention results."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from causalab.causal import Dom, V, mechanism
from causalab.causal.model import CausalModel
from causalab.causal.model_comparison import (
    can_distinguish_with_dataset,
    compute_interchange_scores,
    distinguishability_report,
    intervened_output_vector,
)
from tests._helpers.tiny import tiny_chain_model


@pytest.fixture
def simple_chain_model() -> CausalModel:
    """3-node A→B→C ``CausalModel`` from ``tests/_helpers/tiny.py``."""
    return tiny_chain_model()


# ============================================================================
# compute_interchange_scores
# ============================================================================


class MockDataset:
    """Mock dataset for testing (simulates list-like dataset interface).

    NOTE: per docs/TESTS.md mocking policy, this is a system-boundary stand-in
    for the dataset interface — score-aggregation tests exercise the
    transformation from raw_outputs+labels to score dicts, not the model's
    label-generation. The real ``label_counterfactual_data`` is tested
    separately in ``test_model.py``.
    """

    def __init__(self, data, dataset_id="test_dataset"):
        self.data = data
        self.id = dataset_id

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def mock_causal_model():
    """Mock for the score-aggregation tests.

    Only ``label_counterfactual_data`` is consumed by
    ``compute_interchange_scores``; replacing this with a real
    ``CausalModel`` would require hand-tuning the model's mechanisms to
    produce the deterministic ``expected_i`` labels the score tests rely on
    — that couples the test to the model's forward semantics, which are
    already covered in ``test_model.py``. Keeping the mock here
    isolates the score-aggregation contract.
    """
    model = Mock()

    def mock_label(dataset, target_vars):
        labeled_data = []
        for i, sample in enumerate(dataset):
            labeled_sample = sample.copy()
            labeled_sample["label"] = f"expected_{i}"
            labeled_data.append(labeled_sample)
        return MockDataset(labeled_data, dataset_id=dataset.id)

    model.label_counterfactual_data = mock_label
    return model


@pytest.fixture
def mock_raw_results():
    return {
        "experiment_id": "test_exp",
        "method_name": "PatchResidualStream",
        "model_name": "TestModel",
        "dataset": {
            "test_dataset": {
                "model_unit": {
                    "[[unit_1]]": {
                        "raw_outputs": [
                            {
                                "string": "output_0",
                                "sequences": torch.tensor([[1, 2, 3]]),
                            },
                            {
                                "string": "expected_1",
                                "sequences": torch.tensor([[4, 5, 6]]),
                            },
                            {
                                "string": "expected_2",
                                "sequences": torch.tensor([[7, 8, 9]]),
                            },
                        ],
                        "causal_model_inputs": [
                            {
                                "base_input": {"id": 0, "raw_input": "input_0"},
                                "counterfactual_inputs": [
                                    {"id": 0, "raw_input": "cf_0"}
                                ],
                            },
                            {
                                "base_input": {"id": 1, "raw_input": "input_1"},
                                "counterfactual_inputs": [
                                    {"id": 1, "raw_input": "cf_1"}
                                ],
                            },
                            {
                                "base_input": {"id": 2, "raw_input": "input_2"},
                                "counterfactual_inputs": [
                                    {"id": 2, "raw_input": "cf_2"}
                                ],
                            },
                        ],
                        "metadata": {"layers": [5], "position": "last"},
                        "feature_indices": None,
                    }
                }
            }
        },
    }


@pytest.fixture
def mock_dataset():
    data = [
        {"id": 0, "raw_input": "input_0", "counterfactual_inputs": [{"id": 0}]},
        {"id": 1, "raw_input": "input_1", "counterfactual_inputs": [{"id": 1}]},
        {"id": 2, "raw_input": "input_2", "counterfactual_inputs": [{"id": 2}]},
    ]
    return MockDataset(data, dataset_id="test_dataset")


@pytest.fixture
def exact_match_checker():
    def checker(output, expected):
        return 1.0 if output["string"] == expected else 0.0

    return checker


class TestComputeInterchangeScoresUnit:
    """``compute_interchange_scores`` — annotates raw intervention results with
    per-target-variable scores via a user-supplied checker."""

    pytestmark = pytest.mark.unit

    def test_basic_score_computation(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        target_variables_list = [["output"]]
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            target_variables_list,
            exact_match_checker,
        )

        assert "dataset" in results
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "output" in unit_data
        score_data = unit_data["output"]
        assert "scores" in score_data
        assert "average_score" in score_data
        assert isinstance(score_data["scores"], list)
        assert isinstance(score_data["average_score"], (float, np.floating))

    def test_score_accuracy(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            exact_match_checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        scores = unit_data["output"]["scores"]
        # output_0 != expected_0 -> 0.0; expected_1 == expected_1 -> 1.0; same for 2.
        assert scores == [0.0, 1.0, 1.0]
        assert abs(unit_data["output"]["average_score"] - 2 / 3) < 1e-6

    def test_multiple_target_variables(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"], ["answer"]],
            exact_match_checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "output" in unit_data
        assert "answer" in unit_data

    def test_raw_results_preserved(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            exact_match_checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert "raw_outputs" in unit_data
        assert "causal_model_inputs" in unit_data
        assert len(unit_data["raw_outputs"]) == 3

    def test_does_not_modify_input(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        original_keys = set(
            mock_raw_results["dataset"]["test_dataset"]["model_unit"][
                "[[unit_1]]"
            ].keys()
        )
        compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            exact_match_checker,
        )
        # Original raw_results must not have gained the score keys.
        current_keys = set(
            mock_raw_results["dataset"]["test_dataset"]["model_unit"][
                "[[unit_1]]"
            ].keys()
        )
        assert original_keys == current_keys

    def test_tensor_score_conversion(
        self, mock_raw_results, mock_causal_model, mock_dataset
    ):
        def tensor_checker(output, expected):
            return torch.tensor(1.0 if output["string"] == expected else 0.0)

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            tensor_checker,
        )
        scores = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"][
            "output"
        ]["scores"]
        for s in scores:
            assert isinstance(s, float)
            assert not isinstance(s, torch.Tensor)


class TestComputeInterchangeScoresProperty:
    """Invariances of ``compute_interchange_scores``."""

    pytestmark = pytest.mark.property

    def test_ordering_of_target_variables_list_is_independent(
        self, mock_raw_results, mock_causal_model, mock_dataset, exact_match_checker
    ):
        # Per-group scores must not depend on the outer ordering.
        results_ab = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["a"], ["b"]],
            exact_match_checker,
        )
        results_ba = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["b"], ["a"]],
            exact_match_checker,
        )
        unit_ab = results_ab["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        unit_ba = results_ba["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert unit_ab["a"]["scores"] == unit_ba["a"]["scores"]
        assert unit_ab["b"]["scores"] == unit_ba["b"]["scores"]

    def test_average_score_is_mean_of_scores(
        self, mock_raw_results, mock_causal_model, mock_dataset
    ):
        test_scores = [0.25, 0.5, 0.75]
        idx = [0]

        def checker(output, expected):
            score = test_scores[idx[0]]
            idx[0] += 1
            return score

        results = compute_interchange_scores(
            mock_raw_results,
            mock_causal_model,
            {"test_dataset": mock_dataset},
            [["output"]],
            checker,
        )
        unit_data = results["dataset"]["test_dataset"]["model_unit"]["[[unit_1]]"]
        assert unit_data["output"]["scores"] == test_scores
        assert abs(unit_data["output"]["average_score"] - np.mean(test_scores)) < 1e-6


# ============================================================================
# can_distinguish_with_dataset (module-level function — note: the CausalModel
# *method* with the same name lives in model.py and is tested there)
# ============================================================================


class TestCanDistinguishWithDatasetUnit:
    """``model_comparison.can_distinguish_with_dataset`` — module-level helper
    (note: ``CausalModel`` has a method with the same name; this is its
    free-function ancestor, currently unused but not dead code)."""

    pytestmark = pytest.mark.unit

    def test_identical_targets_yield_zero_count(self, simple_chain_model):
        model = simple_chain_model
        base = model.new_trace({"A": 0})
        cf = model.new_trace({"A": 1})
        dataset = [{"input": base, "counterfactual_inputs": [cf]}]
        result = can_distinguish_with_dataset(dataset, model, target_variables1=["B"])
        # Intervening on B with cf changes the chain -> output differs.
        assert result["count"] == 1
        assert result["proportion"] == 1.0


class TestCanDistinguishWithDatasetProperty:
    """Proportion invariants for the module-level helper."""

    pytestmark = pytest.mark.property

    def test_proportion_in_unit_interval(self, simple_chain_model):
        model = simple_chain_model
        base = model.new_trace({"A": 0})
        cf = model.new_trace({"A": 1})
        dataset = [{"input": base, "counterfactual_inputs": [cf]}]
        result = can_distinguish_with_dataset(dataset, model, target_variables1=["B"])
        assert 0.0 <= result["proportion"] <= 1.0


# ---------------------------------------------------------------------------
# distinguishability engine — intervened_output_vector + distinguishability_report
# ---------------------------------------------------------------------------


def _addition_model() -> CausalModel:
    """X+Y with explicit carry/ones, raw_output reconstructed from both."""
    digits = list(range(10))

    @mechanism
    def equations(X: Dom(digits), Y: Dom(digits)):
        total = V(X + Y, domain=Dom(list(range(19))))
        raw_input = V(f"{X}+{Y}=", domain=Dom(str))  # noqa: F841
        carry = V(total >= 10, domain=Dom([False, True]))
        ones = V(total % 10, domain=Dom(digits))
        raw_output = V(str(int(carry) * 10 + ones), domain=Dom(str))  # noqa: F841
        return ones

    return CausalModel(equations, id="addition")


def _pair(xb, yb, xc, yc) -> dict:
    return {
        "input": {"X": xb, "Y": yb},
        "counterfactual_inputs": [{"X": xc, "Y": yc}],
    }


class TestIntervenedOutputVector:
    pytestmark = pytest.mark.unit

    def test_patches_named_target_and_recomputes_downstream(self):
        model = _addition_model()
        # base 2+3 -> ones=5,carry=F,out="5";  cf 4+4 -> ones=8,carry=F,out="8"
        data = [_pair(2, 3, 4, 4)]
        assert intervened_output_vector(model, [], data) == ["5"]  # null = base
        assert intervened_output_vector(model, ["ones"], data) == ["8"]  # 0*10+8
        # cf carry == base carry (both False), so patching carry is a no-op
        assert intervened_output_vector(model, ["carry"], data) == ["5"]
        # transplant the whole output
        assert intervened_output_vector(model, ["raw_output"], data) == ["8"]

    def test_rejects_multiple_counterfactual_inputs(self):
        model = _addition_model()
        bad = [{"input": {"X": 1, "Y": 1}, "counterfactual_inputs": [{}, {}]}]
        with pytest.raises(AssertionError):
            intervened_output_vector(model, ["ones"], bad)


class TestDistinguishabilityReport:
    pytestmark = pytest.mark.unit

    def _hyps(self):
        return {
            "carry": ("add", ["carry"]),
            "ones": ("add", ["ones"]),
            "null": ("add", []),
            "all": ("add", ["raw_output"]),
        }

    def test_rates_and_always_confounded_groups(self):
        model = _addition_model()
        models_by_name = {"add": model}
        # Both pairs keep carry False (totals < 10), so patching `carry` is a
        # no-op == null; and `ones` == `all` here (both yield the cf ones digit).
        datasets = {"wide": [_pair(2, 3, 4, 4), _pair(1, 1, 3, 5)]}
        report = distinguishability_report(
            models_by_name, self._hyps(), ["ones"], datasets, datasets["wide"]
        )
        per = report["datasets"]["wide"]["per_target"]["ones"]
        assert per["vs_null"] == 1.0  # patching ones moves the output on both
        assert per["vs_all"] == 0.0  # ones == all on these pairs
        assert per["alternatives"]["carry"] == 1.0

        groups = {frozenset(g) for g in report["always_confounded"]}
        assert frozenset({"carry", "null"}) in groups
        assert frozenset({"ones", "all"}) in groups
        assert report["singletons"] == []

    def test_empty_datasets_still_groups_on_random_run(self):
        model = _addition_model()
        report = distinguishability_report(
            {"add": model}, self._hyps(), ["ones"], {}, [_pair(2, 3, 4, 4)]
        )
        assert report["datasets"] == {}
        # carry==null on this no-carry pair; ones==all -> two confounded pairs
        groups = {frozenset(g) for g in report["always_confounded"]}
        assert frozenset({"carry", "null"}) in groups
