"""
Benchmark experiment orchestration using Strategy Pattern.

This module provides a flexible framework for running benchmarks that compare
different intervention methods on neural networks. It uses the Strategy Pattern
to encapsulate different intervention methods (DAS, DBM, SAE, etc.) and provides
a unified BenchmarkRunner class to orchestrate experiments.

Key components:
- MethodStrategy: Abstract base class defining the interface for intervention methods
- Concrete strategies: FullVectorStrategy, DASStrategy, DBMStrategy, etc.
- BenchmarkRunner: Orchestrates the train→test→save→cleanup cycle for any experiment type
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
import os

from experiments.experiment_utils import clear_memory, ensure_dir, generate_model_dir_name
from .LM_experiments.LM_utils import get_sae_loader


class MethodStrategy(ABC):
    """
    Abstract base class for intervention method strategies.

    Each strategy encapsulates the logic for a specific intervention method
    (e.g., DAS, DBM, full vector). Strategies can optionally prepare the
    experiment (e.g., build SVD features) and train interventions.
    """

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of this intervention method."""
        pass

    def requires_training(self) -> bool:
        """
        Whether this method requires training before testing.

        Returns:
            True if train() should be called, False otherwise
        """
        return False

    def prepare(self, experiment, train_data: Dict, verbose: bool = False) -> None:
        """
        Optional preparation step before training (e.g., build SVD features).

        Args:
            experiment: The experiment instance to prepare
            train_data: Training data dictionary
            verbose: Whether to print verbose output
        """
        pass

    def train(self, experiment, train_data: Dict, target_variables: List[str],
              model_dir: Optional[str], verbose: bool = False) -> None:
        """
        Train the intervention (if applicable).

        Args:
            experiment: The experiment instance to train
            train_data: Training data dictionary
            target_variables: List of target variable names
            model_dir: Directory to save trained models
            verbose: Whether to print verbose output
        """
        pass


class FullVectorStrategy(MethodStrategy):
    """
    Strategy for full vector interventions (no dimensionality reduction).

    This method intervenes on the entire activation vector without any
    learned transformations or feature selection.
    """

    def get_method_name(self) -> str:
        return "full_vector"

    def train(self, experiment, train_data: Dict, target_variables: List[str],
              model_dir: Optional[str], verbose: bool = False) -> None:
        """Save featurizers as None (no training needed)."""
        if model_dir:
            experiment.save_featurizers(None, model_dir)


class DASStrategy(MethodStrategy):
    """
    Strategy for Distributed Alignment Search (DAS) interventions.

    DAS learns a linear subspace that aligns with the causal variable
    by training a rotation matrix to project activations.
    """

    def get_method_name(self) -> str:
        return "DAS"

    def requires_training(self) -> bool:
        return True

    def train(self, experiment, train_data: Dict, target_variables: List[str],
              model_dir: Optional[str], verbose: bool = False) -> None:
        experiment.train_interventions(
            train_data,
            target_variables,
            method="DAS",
            verbose=verbose,
            model_dir=model_dir
        )


class DBMStrategy(MethodStrategy):
    """
    Strategy for Distributed Binary Masking (DBM) interventions.

    DBM learns binary masks that select relevant features in the activation
    space, providing interpretable feature importance.
    """

    def get_method_name(self) -> str:
        return "DBM"

    def requires_training(self) -> bool:
        return True

    def train(self, experiment, train_data: Dict, target_variables: List[str],
              model_dir: Optional[str], verbose: bool = False) -> None:
        experiment.train_interventions(
            train_data,
            target_variables,
            method="DBM",
            verbose=verbose,
            model_dir=model_dir
        )


class DBMWithSVDStrategy(MethodStrategy):
    """
    Strategy for DBM with SVD-based feature extraction.

    This method first applies SVD to extract principal components from
    activations, then learns binary masks in this reduced space.
    """

    def get_method_name(self) -> str:
        return "DBM+SVD"

    def requires_training(self) -> bool:
        return True

    def prepare(self, experiment, train_data: Dict, verbose: bool = False) -> None:
        experiment.build_SVD_feature_interventions(train_data, verbose=verbose, PCA=False)

    def train(self, experiment, train_data: Dict, target_variables: List[str],
              model_dir: Optional[str], verbose: bool = False) -> None:
        experiment.train_interventions(
            train_data,
            target_variables,
            method="DBM",
            verbose=verbose,
            model_dir=model_dir
        )


class DBMWithPCAStrategy(MethodStrategy):
    """
    Strategy for DBM with PCA-based feature extraction.

    Similar to DBM+SVD but uses PCA (centered SVD) for feature extraction.
    """

    def get_method_name(self) -> str:
        return "DBM+PCA"

    def requires_training(self) -> bool:
        return True

    def prepare(self, experiment, train_data: Dict, verbose: bool = False) -> None:
        experiment.build_SVD_feature_interventions(train_data, verbose=verbose, PCA=True)

    def train(self, experiment, train_data: Dict, target_variables: List[str],
              model_dir: Optional[str], verbose: bool = False) -> None:
        experiment.train_interventions(
            train_data,
            target_variables,
            method="DBM",
            verbose=verbose,
            model_dir=model_dir
        )


class DBMWithSAEStrategy(MethodStrategy):
    """
    Strategy for DBM with Sparse Autoencoder (SAE) features.

    This method uses pre-trained SAEs to extract interpretable sparse features,
    then learns binary masks over these features. SAEs must be available for
    the specific model being used.
    """

    def get_method_name(self) -> str:
        return "DBM+SAE"

    def requires_training(self) -> bool:
        return True

    def prepare(self, experiment, train_data: Dict, verbose: bool = False) -> None:
        """
        Load SAE for the model and build SAE feature interventions.

        Raises:
            ValueError: If no SAE is available for this model
        """
        # Get model path from pipeline
        if not hasattr(experiment.pipeline.model, 'config') or \
           not hasattr(experiment.pipeline.model.config, '_name_or_path'):
            raise ValueError("Cannot determine model path for SAE loading")

        model_path = experiment.pipeline.model.config._name_or_path
        sae_loader = get_sae_loader(model_path)

        if sae_loader is None:
            raise ValueError(f"No SAE available for model: {model_path}")

        if verbose:
            print(f"Loading SAE for model: {model_path}")

        experiment.build_SAE_feature_intervention(sae_loader)

    def train(self, experiment, train_data: Dict, target_variables: List[str],
              model_dir: Optional[str], verbose: bool = False) -> None:
        experiment.train_interventions(
            train_data,
            target_variables,
            method="DBM",
            verbose=verbose,
            model_dir=model_dir
        )


class BenchmarkRunner:
    """
    Orchestrates benchmark experiments using different intervention methods.

    This class provides a unified interface for running benchmarks across multiple
    intervention methods. It handles the complete experiment lifecycle:
    1. Method preparation (e.g., building SVD features)
    2. Training (if applicable)
    3. Testing/evaluation
    4. Result saving
    5. Memory cleanup

    The runner is agnostic to the specific experiment type (residual stream,
    attention heads, etc.) and delegates to the experiment class for actual work.

    Example:
        >>> runner = BenchmarkRunner(
        ...     experiment_class=PatchResidualStream,
        ...     experiment_kwargs={
        ...         'pipeline': pipeline,
        ...         'causal_model': task,
        ...         'layers': list(range(0, 12)),
        ...         'token_positions': token_positions,
        ...         'checker': checker,
        ...         'config': config
        ...     },
        ...     model_dir="models/",
        ...     results_dir="results/"
        ... )
        >>> results = runner.run_benchmark(
        ...     strategies=[DASStrategy(), DBMStrategy()],
        ...     train_data=train_data,
        ...     test_data=test_data,
        ...     target_variables=["position", "answer"]
        ... )
    """

    def __init__(self,
                 experiment_class: type,
                 experiment_kwargs: Dict[str, Any],
                 model_dir: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 generate_heatmaps: bool = False):
        """
        Initialize the BenchmarkRunner.

        Args:
            experiment_class: Class of experiment to instantiate (e.g., PatchResidualStream)
            experiment_kwargs: Keyword arguments to pass to experiment constructor
            model_dir: Directory to save trained models
            results_dir: Directory to save results
            generate_heatmaps: Whether to generate and save heatmaps (only for residual stream)
        """
        self.experiment_class = experiment_class
        self.experiment_kwargs = experiment_kwargs
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.generate_heatmaps = generate_heatmaps

        # Ensure directories exist
        ensure_dir(model_dir)
        ensure_dir(results_dir)

        # Create heatmap directory if needed
        if generate_heatmaps and results_dir:
            ensure_dir(os.path.join(results_dir, "heatmaps"))

    def run_method(self,
                   strategy: MethodStrategy,
                   train_data: Dict,
                   test_data: Dict,
                   target_variables: List[str],
                   verbose: bool = False) -> Dict:
        """
        Run a single intervention method through the complete experiment cycle.

        Args:
            strategy: The method strategy to use
            train_data: Training data dictionary
            test_data: Testing data dictionary
            target_variables: List of target variable names
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing experiment results
        """
        method_name = strategy.get_method_name()

        if verbose:
            print(f"Running {method_name} method...")

        # Update config with method name
        config = self.experiment_kwargs.get('config', {}).copy()
        config['method_name'] = method_name
        experiment_kwargs = self.experiment_kwargs.copy()
        experiment_kwargs['config'] = config

        # Create experiment instance
        experiment = self.experiment_class(**experiment_kwargs)

        # Prepare (e.g., build SVD features)
        strategy.prepare(experiment, train_data, verbose=verbose)

        # Generate model directory name
        model_name = experiment.pipeline.model.__class__.__name__
        method_model_dir = None
        if self.model_dir:
            method_model_dir = os.path.join(
                self.model_dir,
                generate_model_dir_name(method_name, model_name, target_variables)
            )

        # Train if needed
        if strategy.requires_training() or method_name == "full_vector":
            strategy.train(experiment, train_data, target_variables, method_model_dir, verbose=verbose)

        # Perform interventions on test data
        raw_results = experiment.perform_interventions(
            test_data,
            verbose=verbose,
            target_variables_list=[target_variables],
            save_dir=self.results_dir
        )

        # Generate heatmaps if requested
        if self.generate_heatmaps and self.results_dir and hasattr(experiment, 'plot_heatmaps'):
            self._generate_heatmaps(experiment, raw_results, method_name, model_name, target_variables)

        # Clean up memory
        del experiment, raw_results
        clear_memory()

        return raw_results

    def run_benchmark(self,
                      strategies: List[MethodStrategy],
                      train_data: Dict,
                      test_data: Dict,
                      target_variables: List[str],
                      verbose: bool = False) -> Dict[str, Dict]:
        """
        Run multiple intervention methods and aggregate results.

        Args:
            strategies: List of method strategies to evaluate
            train_data: Training data dictionary
            test_data: Testing data dictionary
            target_variables: List of target variable names
            verbose: Whether to print verbose output

        Returns:
            Dictionary mapping method names to their results
        """
        all_results = {}

        for strategy in strategies:
            method_name = strategy.get_method_name()
            try:
                results = self.run_method(
                    strategy,
                    train_data,
                    test_data,
                    target_variables,
                    verbose=verbose
                )
                all_results[method_name] = results
            except Exception as e:
                if verbose:
                    print(f"Error running {method_name}: {e}")
                all_results[method_name] = {"error": str(e)}

        return all_results

    def _generate_heatmaps(self, experiment, results: Dict, method_name: str,
                          model_name: str, target_variables: List[str]) -> None:
        """
        Generate and save heatmaps for experiment results.

        Args:
            experiment: The experiment instance
            results: Results dictionary
            method_name: Name of the method
            model_name: Name of the model
            target_variables: List of target variable names
        """
        heatmap_path = os.path.join(
            self.results_dir,
            "heatmaps",
            method_name,
            model_name,
            "-".join(target_variables)
        )

        ensure_dir(heatmap_path)

        # Create standard and average heatmaps
        experiment.plot_heatmaps(results, target_variables, save_path=heatmap_path)
        experiment.plot_heatmaps(results, target_variables,
                                average_counterfactuals=True, save_path=heatmap_path)


# ========== Convenience Functions ==========

def residual_stream_baselines(
    pipeline=None,
    task=None,
    token_positions=None,
    train_data=None,
    test_data=None,
    config=None,
    target_variables=None,
    checker=None,
    start=None,
    end=None,
    verbose=False,
    model_dir=None,
    results_dir=None,
    methods=["full_vector", "DAS", "DBM+SVD", "DBM+PCA", "DBM", "DBM+SAE"]
):
    """
    Run different residual stream intervention methods on language models.

    This is a convenience function that wraps BenchmarkRunner for residual
    stream experiments. It maintains backward compatibility with the original
    aggregate_experiments API.

    Parameters:
    -----------
    pipeline : LMPipeline
        Language model pipeline to use for interventions
    task : CausalModel
        Causal model that defines the task
    token_positions : list
        List of token positions to intervene on
    train_data : dict
        Dictionary mapping dataset names to CounterfactualDataset objects for training
    test_data : dict
        Dictionary mapping dataset names to CounterfactualDataset objects for testing
    config : dict
        Configuration dictionary for experiments
    target_variables : list
        List of variable names to target for interventions
    checker : function
        Function that checks if model output matches expected output
    start : int
        Starting layer index for interventions
    end : int
        Ending layer index for interventions
    verbose : bool
        Whether to print verbose output
    model_dir : str
        Directory to save trained models
    results_dir : str
        Directory to save results
    methods : list
        List of methods to run (options: "full_vector", "DAS", "DBM+SVD", "DBM+PCA", "DBM", "DBM+SAE")

    Returns:
    --------
    dict
        Dictionary mapping method names to their results
    """
    from .LM_experiments.residual_stream_experiment import PatchResidualStream

    # Map method names to strategies
    strategy_map = {
        "full_vector": FullVectorStrategy(),
        "DAS": DASStrategy(),
        "DBM": DBMStrategy(),
        "DBM+SVD": DBMWithSVDStrategy(),
        "DBM+PCA": DBMWithPCAStrategy(),
        "DBM+SAE": DBMWithSAEStrategy()
    }

    # Build list of strategies to run
    strategies = []
    for method in methods:
        if method in strategy_map:
            strategies.append(strategy_map[method])
        elif verbose:
            print(f"Warning: Unknown method '{method}', skipping...")

    # Create experiment kwargs
    experiment_kwargs = {
        'pipeline': pipeline,
        'causal_model': task,
        'layers': list(range(start, end)),
        'token_positions': token_positions,
        'checker': checker,
        'config': config
    }

    # Create and run benchmark
    runner = BenchmarkRunner(
        experiment_class=PatchResidualStream,
        experiment_kwargs=experiment_kwargs,
        model_dir=model_dir,
        results_dir=results_dir,
        generate_heatmaps=True  # Generate heatmaps for residual stream
    )

    return runner.run_benchmark(
        strategies=strategies,
        train_data=train_data,
        test_data=test_data,
        target_variables=target_variables,
        verbose=verbose
    )


def attention_head_baselines(
    pipeline=None,
    task=None,
    token_positions=None,
    train_data=None,
    test_data=None,
    config=None,
    target_variables=None,
    checker=None,
    verbose=False,
    model_dir=None,
    results_dir=None,
    heads_list=None,
    skip=[]
):
    """
    Run different intervention methods on attention head outputs.

    This is a convenience function that wraps BenchmarkRunner for attention
    head experiments. It maintains backward compatibility with the original
    aggregate_experiments API.

    Parameters:
    -----------
    pipeline : LMPipeline
        Language model pipeline to use for interventions
    task : CausalModel
        Causal model that defines the task
    token_positions : TokenPosition
        Token positions to intervene on
    train_data : dict
        Dictionary mapping dataset names to CounterfactualDataset objects for training
    test_data : dict
        Dictionary mapping dataset names to CounterfactualDataset objects for testing
    config : dict
        Configuration dictionary for experiments
    target_variables : list
        List of variable names to target for interventions
    checker : function
        Function that checks if model output matches expected output
    verbose : bool
        Whether to print verbose output
    model_dir : str
        Directory to save trained models
    results_dir : str
        Directory to save results
    heads_list : list
        List of (layer, head) tuples to intervene on
    skip : list
        List of methods to skip

    Returns:
    --------
    dict
        Dictionary mapping method names to their results
    """
    from .LM_experiments.attention_head_experiment import PatchAttentionHeads

    # Map method names to strategies (attention heads don't support SAE or SVD/PCA)
    strategy_map = {
        "full_vector": FullVectorStrategy(),
        "DAS": DASStrategy(),
        "DBM": DBMStrategy(),
        "DBM+SVD": DBMWithSVDStrategy(),
        "DBM+PCA": DBMWithPCAStrategy()
    }

    # Build list of strategies to run (exclude skipped methods)
    strategies = []
    for method_name, strategy in strategy_map.items():
        if method_name not in skip:
            strategies.append(strategy)

    # Create experiment kwargs
    experiment_kwargs = {
        'pipeline': pipeline,
        'causal_model': task,
        'layer_head_lists': heads_list,
        'token_position': token_positions,
        'checker': checker,
        'config': config
    }

    # Create and run benchmark
    runner = BenchmarkRunner(
        experiment_class=PatchAttentionHeads,
        experiment_kwargs=experiment_kwargs,
        model_dir=model_dir,
        results_dir=results_dir,
        generate_heatmaps=False  # Don't generate heatmaps for attention heads
    )

    return runner.run_benchmark(
        strategies=strategies,
        train_data=train_data,
        test_data=test_data,
        target_variables=target_variables,
        verbose=verbose
    )
