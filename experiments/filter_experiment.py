import collections
import gc
from typing import Dict, Callable, List, Any, Union

import torch
from tqdm import tqdm
from causal.counterfactual_dataset import CounterfactualDataset
from neural.pipeline import Pipeline
from causal.causal_model import CausalModel


class FilterExperiment:
    """
    Filters CounterfactualDatasets based on agreement between neural and causal models.
    
    This class processes datasets in batches, filtering out examples where
    the neural pipeline and causal model outputs disagree on either the original input
    or any of the counterfactual inputs. It uses batched processing for efficiency
    and can provide verbose output regarding the filtering process.
    
    Attributes:
        pipeline: Neural model pipeline that processes inputs
        causal_model: Causal model that generates expected outputs
        checker: Function that compares neural output with causal output
    """
    def __init__(
        self, 
        pipeline: Pipeline, 
        causal_model: CausalModel, 
        checker: Callable[[Any, Any], bool]
    ) -> None:
        """
        Initialize a FilterExperiment instance.
        
        Args:
            pipeline: Neural model pipeline that processes inputs
            causal_model: Causal model that generates expected outputs
            checker: Function that compares neural output with causal output,
                     returning True if they match
        """
        self.pipeline = pipeline
        self.causal_model = causal_model
        self.checker = checker

    def filter(
        self, 
        counterfactual_datasets: Dict[str, CounterfactualDataset], 
        batch_size: int = 32, 
        verbose: bool = False
    ) -> Dict[str, CounterfactualDataset]:
        """
        Filter datasets based on agreement between pipeline and causal model outputs.
        
        For each example in each dataset, checks if both:
        1. The pipeline's prediction on the original input matches the causal model's output
        2. The pipeline's predictions on all counterfactual inputs match the causal model's outputs
        
        Only examples where both conditions are met are kept in the filtered datasets.
        
        Args:
            counterfactual_datasets: Dictionary mapping dataset names to CounterfactualDataset objects
            batch_size: Size of batches for processing
            verbose: Whether to print filtering statistics
            
        Returns:
            Dictionary mapping dataset names to filtered CounterfactualDataset objects
        """
        filtered_datasets = {}
        total_original = 0
        total_kept = 0
        
        # Process each counterfactual dataset
        for dataset_name, counterfactual_dataset in counterfactual_datasets.items():
            dataset = counterfactual_dataset.dataset
            filtered_data = collections.defaultdict(list)
            dataset_original = len(dataset["input"])
            total_original += dataset_original
            
            # Process dataset in batches
            for b_i in tqdm(range(0, len(dataset["input"]), batch_size), 
                           desc=f"Filtering {dataset_name}", 
                           disable=not verbose):
                # Get batch of original inputs and their counterfactuals
                orig_inputs = dataset["input"][b_i:b_i + batch_size]
                
                # Process original inputs
                orig_valid = self._validate_original_inputs(dataset, orig_inputs, b_i, batch_size)

                # Process counterfactual inputs
                all_cf_inputs = dataset["counterfactual_inputs"][b_i:b_i + batch_size]
                cf_valid = self._validate_counterfactual_inputs(dataset, all_cf_inputs, b_i, batch_size)

                # Filter valid original and counterfactual input pairs
                for idx, is_orig_valid in enumerate(orig_valid):
                    if not is_orig_valid or not cf_valid[idx]:
                        continue
                    # If both pass, add to filtered data
                    filtered_data["input"].append(orig_inputs[idx])
                    filtered_data["counterfactual_inputs"].append(all_cf_inputs[idx])

                # Clean up to free memory
                self._cleanup_memory([orig_inputs, all_cf_inputs])

            # Skip empty datasets
            if "input" not in filtered_data or not filtered_data["input"]:
                if verbose:
                    print(f"Dataset '{dataset_name}' has no valid examples after filtering.")
                continue
                
            # Create filtered dataset
            filtered_datasets[dataset_name] = CounterfactualDataset.from_dict(
                filtered_data, id=dataset_name
            )
            dataset_kept = len(filtered_data["input"])
            total_kept += dataset_kept
            
            if verbose:
                keep_rate = (dataset_kept / dataset_original) * 100
                print(f"Dataset '{dataset_name}': kept {dataset_kept}/{dataset_original} examples "
                      f"({keep_rate:.1f}%)")
        
        # Report overall filtering results
        if verbose and total_original > 0:
            overall_keep_rate = (total_kept / total_original) * 100
            print(f"\nTotal filtering results:")
            print(f"Original examples: {total_original}")
            print(f"Kept examples: {total_kept}")
            print(f"Overall keep rate: {overall_keep_rate:.1f}%")

        return filtered_datasets
    
    def _validate_original_inputs(
        self,
        dataset: Any,
        orig_inputs: List[Any],
        batch_idx: int,
        batch_size: int
    ) -> List[bool]:
        """
        Validate original inputs against causal model expectations.

        Args:
            dataset: The dataset being processed
            orig_inputs: Original inputs for the current batch
            batch_idx: Starting index of the current batch
            batch_size: Size of the batch

        Returns:
            List of booleans indicating whether each original input is valid
        """
        # Get predictions and expected outputs
        batch_output_dict = self.pipeline.generate(orig_inputs)
        decoded_strings = self.pipeline.dump(batch_output_dict["sequences"])
        orig_expected = [
            self.causal_model.run_forward(x)["raw_output"]
            for x in orig_inputs
        ]

        # Split batch outputs into individual example dicts and check validity
        results = []
        for idx, exp in enumerate(orig_expected):
            example_dict = {"sequences": batch_output_dict["sequences"][idx:idx+1]}
            if "scores" in batch_output_dict:
                # Each score in the list corresponds to a token position
                example_dict["scores"] = [score[idx:idx+1] for score in batch_output_dict["scores"]]
            example_dict["string"] = decoded_strings[idx] if isinstance(decoded_strings, list) else decoded_strings
            results.append(self.checker(example_dict, exp))
        return results
    
    def _validate_counterfactual_inputs(
        self,
        dataset: Any,
        all_cf_inputs: List[List[Any]],
        batch_idx: int,
        batch_size: int
    ) -> List[bool]:
        """
        Validate counterfactual inputs against causal model expectations.

        Args:
            dataset: The dataset being processed
            all_cf_inputs: Counterfactual inputs for the current batch
            batch_idx: Starting index of the current batch
            batch_size: Size of the batch

        Returns:
            List of booleans indicating whether each example's counterfactuals are all valid
        """
        # Flatten the counterfactual inputs for batch processing
        cf_flattened_inputs = [
            cf_input for cf_inputs in all_cf_inputs for cf_input in cf_inputs
        ]

        # Skip processing if there are no counterfactual inputs
        if not cf_flattened_inputs:
            return [True for _ in range(len(all_cf_inputs))]

        # Get pipeline predictions for counterfactuals
        batch_output_dict = self.pipeline.generate(cf_flattened_inputs)
        decoded_strings = self.pipeline.dump(batch_output_dict["sequences"])

        # Create individual output dicts for each flattened example
        individual_dicts = []
        for idx in range(len(cf_flattened_inputs)):
            example_dict = {"sequences": batch_output_dict["sequences"][idx:idx+1]}
            if "scores" in batch_output_dict:
                example_dict["scores"] = [score[idx:idx+1] for score in batch_output_dict["scores"]]
            example_dict["string"] = decoded_strings[idx] if isinstance(decoded_strings, list) else decoded_strings
            individual_dicts.append(example_dict)

        # Restructure individual dicts to match original grouping
        cf_output_dicts = []
        i = 0
        for cf_inputs in all_cf_inputs:
            cf_output_dicts.append(individual_dicts[i:i + len(cf_inputs)])
            i += len(cf_inputs)

        cf_expected = [
            [self.causal_model.run_forward(x)["raw_output"] for x in cf_inputs]
            for cf_inputs in all_cf_inputs
        ]

        # Check if all counterfactuals for each example are valid
        return [
            all(self.checker(output_dict, exp) for output_dict, exp in zip(cf_output_dicts[i], cf_expected[i]))
            for i in range(len(cf_output_dicts))
        ]
    
    def _cleanup_memory(self, objects_to_delete: List[Any]) -> None:
        """
        Clean up memory by deleting objects and running garbage collection.
        
        Args:
            objects_to_delete: List of objects to delete
        """
        for obj in objects_to_delete:
            del obj
        
        # Ensure CUDA operations are completed before freeing memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Run garbage collection to free memory
        gc.collect()