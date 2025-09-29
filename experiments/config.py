"""
Configuration settings for causal abstraction experiments.

This module provides default configurations and preset configurations for various
experiments that extend intervention_experiment.py
"""

# Default configuration with all available parameters
DEFAULT_CONFIG = {
    # ========== General Experiment Parameters ==========
    # Batch size for training and general processing
    "batch_size": 32,
    
    # Batch size specifically for evaluation (defaults to batch_size if not set)
    "evaluation_batch_size": 32,
    
    # Name of the method/experiment being run
    "method_name": "InterventionExperiment",
    
    # Whether to output raw scores/logits instead of decoded text
    "output_scores": False,
    
    # Whether to compare raw LM outputs to the causal model output
    # instead of processed/decoded outputs
    "check_raw": False,
    
    # ========== Training Parameters ==========
    # Number of training epochs
    "training_epoch": 3,
    
    # Initial learning rate for optimization
    "init_lr": 1e-2,
    
    # L1/L2 regularization coefficient for sparsity
    "regularization_coefficient": 1e-4,
    
    # Maximum number of tokens to generate for output
    "max_output_tokens": 1,
    
    # Directory for TensorBoard logs
    "log_dir": "logs",
    
    # Number of features for DAS/DBM methods
    "n_features": 32,
    
    # Temperature annealing schedule for mask-based methods (start, end)
    "temperature_schedule": (1.0, 0.01),
    
    # ========== Optional Training Parameters ==========
    # Early stopping patience (None to disable)
    "patience": None,
    
    # Learning rate scheduler type ("constant", "linear", "cosine")
    "scheduler_type": "constant",
    
    # Frequency of memory cleanup during training (in batches)
    "memory_cleanup_freq": 50,
    
    # Whether to shuffle training data
    "shuffle": True,

    # Whether to save raw_outputs in the results
    "raw_outputs": True,

    # Keyword arguments to pass to Featurizer constructors
    # This allows flexible configuration of featurizer behavior
    "featurizer_kwargs": {
        # Whether to tie mask weights within each atomic model unit (DBM only)
        "tie_masks": False
    }
}

# Preset configurations for common experiments