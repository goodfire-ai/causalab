from dataclasses import dataclass
from typing import Dict, Callable
from causal.causal_model import CounterfactualDataset


@dataclass
class Task:
    """
    A task abstraction that holds functions for generating causal models,
    datasets, and token positions.

    Attributes:
        name: Name of the task
        causal_model_fns: Dictionary of functions that generate CausalModel instances
        dataset_fns: Dictionary of functions that generate CounterfactualDataset instances
        token_position_fns: Dictionary of functions that generate TokenPosition lists
    """
    name: str
    causal_models: Dict[str, Callable]
    dataset_generators: Dict[str, Callable]
    token_positions: Dict[str, Callable]

    def create_token_positions(self, pipeline):
        """
        Update the token_positions dictionary by calling each token position factory
        function with the provided pipeline.

        Args:
            pipeline: The pipeline object to pass to each token position factory
        """
        return {
            name: factory(pipeline)
            for name, factory in self.token_positions.items()
        }

    def create_datasets(self, num_samples, suffix=""):
        """
        Create CounterfactualDataset objects from the dataset generators.

        Args:
            num_samples: Number of samples to generate for each dataset

        Returns:
            Dictionary mapping dataset names to CounterfactualDataset objects
        """

        return {
            name + suffix: CounterfactualDataset.from_sampler(num_samples, generator, id=name)
            for name, generator in self.dataset_generators.items()
        }