from experiments.residual_stream_experiment import PatchResidualStream
from experiments.attention_head_experiment import PatchAttentionHeads
import os
import gc
import torch

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
    """
    def heatmaps(experiment, results, config):
        """Generate and save heatmaps for experiment results."""
        heatmap_path = os.path.join(results_dir, "heatmaps", config["method_name"], 
                         pipeline.model.__class__.__name__, "-".join(target_variables))

        # Create directory if it doesn't exist
        if not os.path.exists(heatmap_path):
            os.makedirs(heatmap_path)
            
        # Create standard and average heatmaps
        experiment.plot_heatmaps(results, target_variables, save_path=heatmap_path)
        experiment.plot_heatmaps(results, target_variables, average_counterfactuals=True, save_path=heatmap_path)
    
    def clear_memory():
        """Free memory between experiments to prevent OOM errors."""
        # Clear Python garbage collector
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force a synchronization point to ensure memory is freed
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Make sure results directory exists
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Model directory for saving trained interventions
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Run full vector intervention method
    if "full_vector" in methods:
        if verbose:
            print("Running full vector method...")
            
        config["method_name"] = "full_vector"
        experiment = PatchResidualStream(pipeline, task, list(range(start, end)), token_positions, checker, config=config)
        method_model_dir = os.path.join(model_dir, f"full_vector_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.save_featurizers(None, method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        heatmaps(experiment, raw_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    # Run DAS (Direct Attribution with Subspace) method
    if "DAS" in methods:
        if verbose:
            print("Running DAS method with alignment map"+config["alignment_map"]+"...")
            
        config["method_name"] = "DAS"
        experiment = PatchResidualStream(pipeline, task, list(range(start, end)), token_positions, checker, config=config)
        method_model_dir = os.path.join(model_dir, f"DAS_{config['alignment_map']}_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DAS", verbose=verbose, model_dir=method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        heatmaps(experiment, raw_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    # Run DBM+SVD method (Differential Binary Masking with SVD)
    if "DBM+SVD" in methods:
        if verbose:
            print("Running DBM+SVD method...")
            
        config["method_name"] = "DBM+SVD"
        experiment = PatchResidualStream(pipeline, task, list(range(start, end)), token_positions, checker, config=config)
        experiment.build_SVD_feature_interventions(train_data, verbose=verbose)  # No PCA=True here
        method_model_dir = os.path.join(model_dir, f"DBM+SVD_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        heatmaps(experiment, raw_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    # Run DBM+PCA method (Differential Binary Masking with PCA)
    if "DBM+PCA" in methods:
        if verbose:
            print("Running DBM+PCA method...")
            
        config["method_name"] = "DBM+PCA"
        experiment = PatchResidualStream(pipeline, task, list(range(start, end)), token_positions, checker, config=config)
        experiment.build_SVD_feature_interventions(train_data, verbose=verbose, PCA=True)  # With PCA=True
        method_model_dir = os.path.join(model_dir, f"DBM+PCA_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        heatmaps(experiment, raw_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    # Run standard DBM method
    if "DBM" in methods:
        if verbose:
            print("Running DBM method...")
            
        config["method_name"] = "DBM"
        experiment = PatchResidualStream(pipeline, task, list(range(start, end)), token_positions, checker, config=config)
        method_model_dir = os.path.join(model_dir, f"DBM_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        heatmaps(experiment, raw_results, config)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    # Run Gemma SAE method (specific to Gemma models)
    if "DBM+SAE" in methods and hasattr(pipeline.model, 'config') and hasattr(pipeline.model.config, '_name_or_path'):
        model_path = pipeline.model.config._name_or_path
        
        # For Gemma 2B model
        if model_path == "google/gemma-2-2b":
            if verbose:
                print("Running DBM+SAE method for Gemma 2B...")
                
            config["method_name"] = "DBM+SAE"
            from sae_lens import SAE

            def sae_loader(layer):
                sae, _, _ = SAE.from_pretrained(
                    release = "gemma-scope-2b-pt-res-canonical",
                    sae_id = f"layer_{layer}/width_16k/canonical",
                    device = "cpu",
                )
                return sae

            experiment = PatchResidualStream(pipeline, task, list(range(start, end)), token_positions, checker, config=config)
            experiment.build_SAE_feature_intervention(sae_loader)
            method_model_dir = os.path.join(model_dir, f"DBM+SAE_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
            experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
            raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
            heatmaps(experiment, raw_results, config)
            
            # Final memory cleanup
            del experiment, raw_results, sae_loader
            clear_memory()

        # For Llama 3.1 8B model
        elif model_path == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            if verbose:
                print("Running DBM+SAE method for Llama 3.1 8B...")
                
            config["method_name"] = "DBM+SAE"
            from sae_lens import SAE

            def sae_loader(layer):
                sae, _, _ = SAE.from_pretrained(
                    release = "llama_scope_lxr_8x",
                    sae_id = f"l{layer}r_8x",
                    device = "cpu",
                )
                return sae

            experiment = PatchResidualStream(pipeline, task, list(range(start, end)), token_positions, checker, config=config)
            experiment.build_SAE_feature_intervention(sae_loader)
            method_model_dir = os.path.join(model_dir, f"DBM+SAE_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
            experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
            raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
            heatmaps(experiment, raw_results, config)
            
            # Final memory cleanup
            del experiment, raw_results, sae_loader
            clear_memory()
        
        elif verbose:
            print(f"No SAE available for model: {model_path}")

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
    
    Parameters are similar to residual_stream_baselines, with the addition of:
    
    heads_list : list
        List of (layer, head) tuples to intervene on
    skip : list
        List of methods to skip
    """
    # Import the generic attention head class
    
    def clear_memory():
        # Clear Python garbage collector
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force a synchronization point to ensure memory is freed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
    # Make sure directories exist
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if "full_vector" not in skip:
        if verbose:
            print("Running full_vector method...")
            
        # Full vector method
        config["method_name"] = "full_vector"
        experiment = PatchAttentionHeads(pipeline, task,  heads_list, token_positions, checker, config=config)
        method_model_dir = os.path.join(model_dir, f"full_vector_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.save_featurizers(None, method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    if "DAS" not in skip:
        if verbose:
            print("Running DAS method...")
            
        # DAS method
        config["method_name"] = "DAS"
        experiment = PatchAttentionHeads(pipeline, task, heads_list, token_positions, checker, config=config)
        method_model_dir = os.path.join(model_dir, f"DAS_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DAS", verbose=verbose, model_dir=method_model_dir)
        clear_memory()  # Clear memory after training
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    if "DBM" not in skip:
        if verbose:
            print("Running DBM method...")
            
        # DBM method
        config["method_name"] = "DBM"
        experiment = PatchAttentionHeads(pipeline, task, heads_list, token_positions, checker, config=config)
        method_model_dir = os.path.join(model_dir, f"DBM_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    if "DBM+SVD" not in skip:
        if verbose:
            print("Running DBM+SVD method...")
            
        # DBM+SVD method
        config["method_name"] = "DBM+SVD"
        experiment = PatchAttentionHeads(pipeline, task, heads_list, token_positions, checker, config=config)
        experiment.build_SVD_feature_interventions(train_data, verbose=verbose, PCA=False)
        method_model_dir = os.path.join(model_dir, f"DBM+SVD_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()

    if "DBM+PCA" not in skip:
        if verbose:
            print("Running DBM+PCA method...")
            
        # DBM+PCA method
        config["method_name"] = "DBM+PCA"
        experiment = PatchAttentionHeads(pipeline, task, heads_list, token_positions, checker, config=config)
        experiment.build_SVD_feature_interventions(train_data, verbose=verbose, PCA=True)
        method_model_dir = os.path.join(model_dir, f"DBM+PCA_{pipeline.model.__class__.__name__}_{"-".join(target_variables)}")
        experiment.train_interventions(train_data, target_variables, method="DBM", verbose=verbose, model_dir=method_model_dir)
        raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir)
        
        # Release memory before next experiment
        del experiment, raw_results
        clear_memory()
