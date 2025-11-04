"""
Configuration settings for causal abstraction experiments.

This module provides default configurations and preset configurations for various
experiments that extend intervention_experiment.py
"""

# Default configuration with all available parameters
DEFAULT_CONFIG = {
    # ========== General Experiment Parameters ==========
    # Batch size for training and general processing
    "train_batch_size": 32,

    # Batch size specifically for evaluation
    "evaluation_batch_size": 32,

    # Name of the method/experiment being run
    "method_name": "InterventionExperiment",

    # Unique identifier for this experiment configuration
    "id": "default_experiment",

    # Whether to include raw scores/logits in output dictionary and keep in memory
    "output_scores": True,

    # Number of top-k logits to extract and keep in memory (applied immediately after generation)
    # Set to None or 0 to keep full vocabulary logits (uses ~256K floats per token, can cause OOM)
    # Recommended: 10-50 (reduces memory by ~5000-25000x)
    "top_k_logits": 10,

    # ========== General Training Parameters ==========
    # Number of training epochs
    "training_epoch": 3,

    # Initial learning rate for optimization
    "init_lr": 1e-2,

    # Maximum number of tokens to generate for output
    "max_output_tokens": 1,

    # Directory for TensorBoard logs
    "log_dir": "logs",

    # ========== Optional Training Parameters ==========
    # Early stopping patience (None to disable)
    "patience": None,

    # Learning rate scheduler type ("constant", "linear", "cosine")
    "scheduler_type": "constant",

    # Frequency of memory cleanup during training (in batches)
    "memory_cleanup_freq": 50,

    # Whether to shuffle training data
    "shuffle": True,


    # ========== DAS-specific Parameters ==========
    # Used in train_interventions() when method="DAS"
    "DAS": {
        # Number of features/dimensions for the learned subspace
        "n_features": 32,
    },

    # ========== DBM/Masking-specific Parameters ==========
    # Used in train_interventions() when method="DBM"
    "masking": {
        # L1 regularization coefficient for sparsity
        "regularization_coefficient": 1e-4,

        "temperature_annealing_fraction": 0.5,  # Fraction of training steps to anneal temperature

        # Temperature annealing schedule for Gumbel-Softmax (start, end)
        "temperature_schedule": (1.0, 0.001),
    },

    # Keyword arguments to pass to Featurizer constructors
    # This allows flexible configuration of featurizer behavior
    "featurizer_kwargs": {
        # Whether to tie mask weights within each atomic model unit (DBM only)
        "tie_masks": False
    }
}

# Preset configurations for common experiments