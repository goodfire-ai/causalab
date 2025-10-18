from datasets import Dataset

class CounterfactualDataset():
    """
    A Dataset class for managing counterfactual data.
    
    Counterfactuals are alternative inputs to a neural network or causal model.
    These counterfactual inputs are used for interchange interventions, where 
    the original inputs are run on the neural network or causal model and then 
    features or variables are fixed to take on values they would have if the 
    counterfactual were provided.
    
    Attributes:
        id (str): Identifier for the dataset.
    """
    
    def __init__(self, dataset=None, id="null"):
        """
        Initialize a CounterfactualDataset instance.
        
        Args:
            dataset (Dataset, optional): An existing HuggingFace dataset to use.
                                        If provided, it must contain "input" and
                                        "counterfactual_inputs" features.
                                        Defaults to None.
            id (str, optional): Identifier for the dataset. Defaults to "null".
            **kwargs: Additional keyword arguments passed to the parent Dataset class
                     when creating a new dataset (if dataset is None).
        
        Raises:
            AssertionError: If required features "input" or "counterfactual_inputs" 
                            are missing in the provided dataset.
        """
        self.id = id
        
        if dataset is not None:
            # Use the provided dataset
            # Verify it has the required features
            assert "input" in dataset.features, "Provided dataset missing 'input' feature"
            assert "counterfactual_inputs" in dataset.features, "Provided dataset missing 'counterfactual_inputs' feature"
            
            # Initialize with the provided dataset
            self.dataset = dataset
        else:
            # Create a new empty dataset with the required features
            empty_data = {"input": [], "counterfactual_inputs": []}
            assert "input" in self.features
            assert "counterfactual_inputs" in self.features

    @classmethod
    def from_dict(cls, data_dict, id="null"):
        """
        Create a CounterfactualDataset from a dictionary.
        
        Args:
            data_dict (dict): Dictionary containing "input" and "counterfactual_inputs".
        
        Returns:
            CounterfactualDataset: A new CounterfactualDataset instance.
        """
        dataset = Dataset.from_dict(data_dict)
        return cls(dataset=dataset, id=id)

    @classmethod
    def from_sampler(cls, size, counterfactual_sampler, filter=None, id="null"):
        """
        Generate a dataset of counterfactual examples.
        
        Creates a new dataset by repeatedly sampling inputs and their counterfactuals
        using the provided sampling function, optionally filtering the samples.
        
        Args:
            size (int): Number of examples to generate.
            counterfactual_sampler (callable): Function that returns a dictionary 
                                            with keys "input" and "counterfactual_inputs".
            filter (callable, optional): Function that takes a sample and returns a boolean 
                                        indicating whether to include it. Defaults to None.
        
        Returns:
            CounterfactualDataset: A new CounterfactualDataset containing the generated examples.
        """
        inputs = []
        counterfactuals = []
        while len(inputs) < size:
            sample = counterfactual_sampler()  # sample is a dict with keys "input" and "counterfactual_inputs"
            if filter is None or filter(sample):
                inputs.append(sample["input"])
                counterfactuals.append(sample["counterfactual_inputs"])
        
        # Create and return a CounterfactualDataset with the generated data
        dataset = Dataset.from_dict({
            "input": inputs,
            "counterfactual_inputs": counterfactuals
        })
        return cls(dataset=dataset, id=id)

    def display_counterfactual_data(self, num_examples=1, verbose=True):
        """
        Display examples from the dataset, showing both the original inputs
        and their corresponding counterfactual inputs.
        
        Args:
            num_examples (int, optional): Number of examples to display. Defaults to 1.
            verbose (bool, optional): Whether to print additional information such as
                                    dataset ID and formatting. Defaults to True.
        
        Returns:
            dict: A dictionary containing the displayed examples for programmatic access.
        """
        if verbose:
            print(f"Dataset '{self.id}':")
        
        displayed_examples = {}
        
        for i in range(min(num_examples, len(self))):
            example = self.dataset[i]
            
            if verbose:
                print(f"\nExample {i+1}:")
                print(f"Input: {example['input']}")
                print(f"Counterfactual Inputs ({len(example['counterfactual_inputs'])} alternatives):")
                
                for j, counterfactual_input in enumerate(example["counterfactual_inputs"]):
                    print(f"  [{j+1}] {counterfactual_input}")
            
            # Store for programmatic access
            displayed_examples[i] = {
                "input": example["input"],
                "counterfactual_inputs": example["counterfactual_inputs"]
            }
        
        if verbose and len(self) > num_examples:
            print(f"\n... {len(self) - num_examples} more examples not shown")
        
        return displayed_examples
    
    def add_column(self, column_name, column_data):
        """
        Add a new column to the dataset.
        
        Args:
            column_name (str): Name of the new column.
            column_data (list): Data for the new column.
        
        Raises:
            ValueError: If the length of column_data does not match the number of examples in the dataset.
        """
        if len(column_data) != len(self.dataset):
            raise ValueError(f"Length of {column_name} must match number of examples in dataset.")
        
        self.dataset = self.dataset.add_column(column_name, column_data)
    
    def remove_column(self, column_name):
        """
        Remove a column from the dataset.
        
        Args:
            column_name (str): Name of the column to remove.
        """
        self.dataset = self.dataset.remove_columns(column_name)

    def __getitem__(self, idx):
        """
        Get an example from the dataset by index.
        
        Args:
            idx (int): Index of the example to retrieve.
        
        Returns:
            dict: The example at the specified index, containing "input" and 
                    "counterfactual_inputs".
        """
        return self.dataset[idx]

    def __len__(self):
        """
        Return the number of examples in the dataset.
        
        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.dataset)

