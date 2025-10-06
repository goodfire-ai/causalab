import random
import copy
import itertools
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from datasets import Dataset, load_dataset

# visualization libraries
from dash import Dash, html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto

from causal.counterfactual_dataset import CounterfactualDataset

class DEFAULT_COLORS:
    """default colors for displaying causal model with dash"""
    BASE_INPUT = "#68AF9C"
    BASE_OUTPUT = "#A0CCC0"
    SOURCE_INPUT = "#EC3B82"
    SOURCE_OUTPUT = "#F276A8"
    COUNTERFACTUAL_OUTPUT = "#A59FD9"

class CausalModel:
    """
    A class to represent a causal model with variables, values, parents, and mechanisms.
    Attributes:
    -----------
    variables : list
        A list of variables in the causal model.
    values : dict
        A dictionary mapping each variable to its possible values.
    parents : dict
        A dictionary mapping each variable to its parent variables.
    mechanisms : dict
        A dictionary mapping each variable to its causal mechanism.
    print_pos : dict, optional
        A dictionary specifying positions for plotting (default is None).
    """
    def __init__(
        self,
        variables,
        values,
        parents,
        mechanisms,
        print_pos=None,
        id="null"
    ):
        """
        Initialize a CausalModel instance with the given parameters.
        
        Parameters:
        -----------
        variables : list
            A list of variables in the causal model.
        values : dict
            A dictionary mapping each variable to its possible values.
        parents : dict
            A dictionary mapping each variable to its parent variables.
        mechanisms : dict
            A dictionary mapping each variable to its causal mechanism.
        print_pos : dict, optional
            A dictionary specifying positions for plotting (default is None).
        """
        self.variables = variables
        self.values = values
        self.parents = parents
        self.mechanisms = mechanisms
        self.id = id
        assert "raw_input" in self.variables, "Variable 'raw_input' must be present in the model variables."
        assert "raw_output" in self.variables, "Variable 'raw_output' must be present in the model variables."

        # Create children and verify model integrity
        self.children = {var: [] for var in variables}
        for variable in variables:
            assert variable in self.parents
            for parent in self.parents[variable]:
                self.children[parent].append(variable)

        # Find inputs and outputs
        self.inputs = [var for var in self.variables if len(parents[var]) == 0]
        self.outputs = copy.deepcopy(variables)
        for child in variables:
            for parent in parents[child]:
                if parent in self.outputs:
                    self.outputs.remove(parent)

        # Generate timesteps
        self.timesteps = {input_var: 0 for input_var in self.inputs}
        step = 1
        change = True
        while change:
            change = False
            copytimesteps = copy.deepcopy(self.timesteps)
            for parent in self.timesteps:
                if self.timesteps[parent] == step - 1:
                    for child in self.children[parent]:
                        copytimesteps[child] = step
                        change = True
            self.timesteps = copytimesteps
            step += 1
        self.end_time = step - 2
        for output in self.outputs:
            self.timesteps[output] = self.end_time

        # Verify that the model is valid
        for variable in self.variables:
            try:
                assert variable in self.values
            except AssertionError:
                raise ValueError(f"Variable {variable} not in values")
            try:
                assert variable in self.children
            except AssertionError:
                raise ValueError(f"Variable {variable} not in children")
            try:
                assert variable in self.mechanisms
            except AssertionError:
                raise ValueError(f"Variable {variable} not in mechanisms")
            try:
                assert variable in self.timesteps
            except AssertionError:
                raise ValueError(f"Variable {variable} not in timesteps")

            for variable2 in copy.copy(self.variables):
                if variable2 in self.parents[variable]:
                    try:
                        assert variable in self.children[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in children of {variable2}")
                    try:
                        assert self.timesteps[variable2] < self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has a later timestep than {variable}")
                if variable2 in self.children[variable]:
                    try:
                        assert variable in parents[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in parents of {variable2}")
                    try:
                        assert self.timesteps[variable2] > self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has an earlier timestep than {variable}")
        
        # Sort variables by timestep
        self.variables.sort(key=lambda x: self.timesteps[x])

        # Set positions for plotting
        self.print_pos = print_pos
        width = {_: 0 for _ in range(len(self.variables))}
        if self.print_pos is None:
            self.print_pos = dict()
        if "raw_input" not in self.print_pos:
            self.print_pos["raw_input"] =  (0, -2)
        for var in self.variables:
            if var not in self.print_pos:
                self.print_pos[var] = (width[self.timesteps[var]], self.timesteps[var])
                width[self.timesteps[var]] += 1

        # Initializing the equivalence classes of children values
        # that produce a given parent value is expensive
        self.equiv_classes = {}

    # FUNCTIONS FOR RUNNING THE MODEL

    def run_forward(self, intervention=None):
        """
        Run the causal model forward with optional interventions.
        
        Parameters:
        -----------
        intervention : dict, optional
            A dictionary mapping variables to their intervened values (default is None).
            
        Returns:
        --------
        dict
            A dictionary mapping each variable to its computed value.
        """
        total_setting = defaultdict(None)
        length = len(list(total_setting.keys()))
        while length != len(self.variables):
            for variable in self.variables:
                for variable2 in self.parents[variable]:
                    if variable2 not in total_setting:
                        continue
                if intervention is not None and variable in intervention:
                    total_setting[variable] = intervention[variable]
                else:
                    total_setting[variable] = self.mechanisms[variable](
                        *[total_setting[parent] for parent in self.parents[variable]]
                    )
            length = len(list(total_setting.keys()))
        return total_setting

    def run_interchange(self, input_setting, counterfactual_inputs):
        """
        Run the model with interchange interventions.
        
        Parameters:
        -----------
        input_setting : dict
            A dictionary mapping input variables to their values.
        counterfactual_inputs : dict
            A dictionary mapping variables to their counterfactual input settings.
            
        Returns:
        --------
        dict
            A dictionary mapping each variable to its computed value after
            interchange interventions.
        """
        interchange_intervention = copy.deepcopy(input_setting)
        for var in counterfactual_inputs:
            setting = self.run_forward(counterfactual_inputs[var])
            interchange_intervention[var] = setting[var]
        return self.run_forward(interchange_intervention)

    # FUNCTIONS FOR SAMPLING INPUTS AND GENERATING DATASETS

    def sample_intervention(self, filter_func=None):
        """
        Sample a random intervention that satisfies an optional filter.
        
        Parameters:
        -----------
        filter_func : function, optional
            A function that takes an intervention and returns a boolean indicating
            whether it satisfies the filter (default is None).
            
        Returns:
        --------
        dict
            A dictionary mapping variables to their sampled intervention values.
        """
        filter_func = filter_func if filter_func is not None else lambda x: len(x.keys()) >0
        intervention = {}
        while not filter_func(intervention):
            intervention = {}
            while len(intervention.keys()) == 0:
                for var in self.variables:
                    if var in self.inputs or var in self.outputs:
                        continue
                    if random.choice([0, 1]) == 0:
                        intervention[var] = random.choice(self.values[var])
        return intervention

    def sample_input(self, filter_func=None):
        """
        Sample a random input that satisfies an optional filter when run through the model.
        
        Parameters:
        -----------
        filter_func : function, optional
            A function that takes a setting and returns a boolean indicating
            whether it satisfies the filter (default is None).
            
        Returns:
        --------
        dict
            A dictionary mapping input variables to their sampled values.
        """
        filter_func = filter_func if filter_func is not None else lambda x: True
        input_setting = {var: random.choice(self.values[var]) for var in self.inputs}
        total = self.run_forward(intervention=input_setting)
        while not filter_func(total):
            input_setting = {var: random.choice(self.values[var]) for var in self.inputs}
            total = self.run_forward(intervention=input_setting)
        return input_setting

    def generate_dataset(self, size, input_sampler=None, filter_func=None):
        """
        Generate a dataset of inputs.
        
        Parameters:
        -----------
        size : int
            The number of samples to generate.
        input_sampler : function, optional
            A function that samples inputs (default is self.sample_input).
        filter_func : function, optional
            A function that takes an input and returns a boolean indicating
            whether it satisfies the filter (default is None).
            
        Returns:
        --------
        Dataset
            A Hugging Face Dataset with an "input" field.
        """
        if input_sampler is None:
            input_sampler = self.sample_input
        inputs = []
        while len(inputs) < size:
            inp = input_sampler()
            if filter_func is None or filter_func(inp):
                inputs.append(inp)
        # Create and return a Hugging Face Dataset with a single "input" field.
        return Dataset.from_dict({"input": inputs})

    def label_counterfactual_data(self, dataset, target_variables):
        """
        Labels a dataset with results from running interchange interventions.
        
        Takes a dataset containing inputs and counterfactual inputs, runs interchange
        interventions using the specified target variables, and returns a new dataset
        with labeled outputs.
        
        Parameters:
        -----------
        dataset : Dataset
            Dataset containing "input" and "counterfactual_inputs" fields.
        target_variables : list
            List of variable names to use for interchange.
            
        Returns:
        --------
        CounterfactualDataset
            A new dataset with labeled results from interchange interventions.
        """
        labels = []
        settings = []
        
        for example in dataset:
            input = example["input"] 
            counterfactual_inputs = example["counterfactual_inputs"]
            
            setting = self.run_interchange(
                input, 
                dict(zip(target_variables, counterfactual_inputs))
            )
            labels.append(setting["raw_output"])
            settings.append(setting)
        
        if "label" in dataset.dataset.features:
            dataset.remove_column("label")
        dataset.add_column("label", labels)
        if "setting" in dataset.dataset.features:
            dataset.remove_column("setting")
        dataset.add_column("setting", settings)
            
        return dataset

    def label_data_with_variables(self, dataset, target_variables):
        """
        Labels a dataset based on variable settings from running the forward model.
        
        Takes a dataset of inputs, runs the forward model on each input, and assigns
        a unique label ID based on the values of the specified target variables.
        
        Parameters:
        -----------
        dataset : Dataset
            Dataset containing "input" field.
        target_variables : list
            List of variable names to use for labeling.
            
        Returns:
        --------
        tuple
            A tuple containing:
                - Dataset: A new dataset with inputs and corresponding labels.
                - dict: A mapping from concatenated target variable values to label IDs.
        """
        inputs = []
        labels = []
        label_to_setting = {}
        
        new_id = 0
        for example in dataset:
            # Store input
            inputs.append(example["input"])
            
            # Run forward model and get target variable values
            setting = self.run_forward(example["input"])
            target_labels = [str(setting[var]) for var in target_variables]
            
            # Assign or create a label ID
            label_key = "".join(target_labels)
            if label_key in label_to_setting:
                id_value = label_to_setting[label_key]
            else:
                id_value = new_id
                label_to_setting[label_key] = new_id
                new_id += 1
                
            labels.append(id_value)
            
        return Dataset.from_dict({"input": inputs, "label": labels}), label_to_setting
    
    def can_distinguish_with_dataset(self, dataset, target_variables1, target_variables2):
        """
        Check if the model can distinguish between two sets of target variables 
        using interchange interventions on a counterfactual dataset.
        """

        count = 0
        for example in dataset:
            input = example["input"]
            counterfactual_inputs = example["counterfactual_inputs"]
            assert len(counterfactual_inputs) == 1 
            
            setting1 = self.run_interchange(
                input, 
                {var: counterfactual_inputs[0] for var in target_variables1}
            )
            if target_variables2 is not None:
                setting2 = self.run_interchange(
                    input, 
                    {var: counterfactual_inputs[0] for var in target_variables2}
                )
                if setting1["raw_output"] != setting2["raw_output"]:
                    count += 1
            else:
                if setting1["raw_output"] != self.run_forward(input)["raw_output"]:
                    count += 1
        
        proportion = count / len(dataset)
        print(f"Can distinguish between {target_variables1} and {target_variables2}: {count} out of {len(dataset)} examples")
        print(f"Proportion of distinguishable examples: {proportion:.2f}")
        return {"proportion": proportion, "count": count}

    # FUNCTIONS FOR PRINTING OUT THE MODEL AND SETTINGS

    def display_structure(self, colors=DEFAULT_COLORS):
        """Visualize the structure of the causal model without any inputs."""
        variables = [
            {
                'data': {
                    'id': var, 
                    'label': var
                },
                'position': {'x': 0, 'y': 0}, # updated by automatic layout
                'classes': 'variable' # as opposed to their inner values
            }
            for var in self.variables
        ]
        edges = [
            {'data': {'id': f'{parent}->{child}', 'source': parent, 'target': child}}
            for child in self.variables
            for parent in self.parents[child]
        ]

        app = Dash()

        app.layout = html.Div([
            cyto.Cytoscape(
                id='causal-graph-visualization',
                elements=variables + edges,
                layout={
                    'name': 'breadthfirst',
                    'roots': '#raw_input'
                },
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'content': 'data(label)',
                            'width': 50,
                            'height': 50
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'curve-style': 'straight',
                            'target-arrow-shape': 'triangle',
                        }
                    },
                    {
                        'selector': '.variable',
                        'style': {
                            'text-valign': 'top'
                        }
                    },
                    {
                        'selector': '.base_input',
                        'style': {
                            'background-color': colors.BASE_INPUT,
                            'color': 'white',
                            'text-valign': 'center'
                        }
                    },
                    {
                        'selector': '.base_value',
                        'style': {
                            'background-color': colors.BASE_OUTPUT,
                            'text-valign': 'center'
                        }
                    },
                    {
                        'selector': '.source_input',
                        'style': {
                            'background-color': colors.SOURCE_INPUT,
                            'color': 'white',
                            'text-valign': 'center',
                        }
                    },
                    {
                        'selector': '.source_value',
                        'style': {
                            'background-color': colors.SOURCE_OUTPUT,
                            'text-valign': 'center',
                        }
                    },
                    {
                        'selector': '.counterfactual_value',
                        'style': {
                            'background-color': colors.COUNTERFACTUAL_OUTPUT,
                            'text-valign': 'center',
                        }
                    }
                ]
            )
        ])

        app.run()

    def display_forward_pass(self, inputs, intervention=None, colors=DEFAULT_COLORS):
        """Visualize a forward pass of the model."""
        if intervention is None:
            intervention = {}
        outputs = self.run_forward({**inputs, **intervention})

        intervention_only = self._get_descendants(intervention)
        counterfactual = self._get_descendants(intervention, strict=False)

        variables = [
            {
                'data': {
                    'id': var, 
                    'label': var
                },
                'position': {'x': 0, 'y': 0}, # updated by automatic layout
                'classes': 'variable' # as opposed to their inner values
            }
            for var in self.variables
        ]
        edges = [
            {'data': {'id': f'{parent}->{child}', 'source': parent, 'target': child}}
            for child in self.variables
            for parent in self.parents[child]
        ]

        app = Dash()

        app.layout = html.Div([
            cyto.Cytoscape(
                id='causal-graph-visualization',
                elements=variables + edges,
                layout={
                    'name': 'breadthfirst',
                    'roots': '#raw_input'
                },
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'content': 'data(label)',
                            'width': 50,
                            'height': 50
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'curve-style': 'straight',
                            'target-arrow-shape': 'triangle',
                        }
                    },
                    {
                        'selector': '.variable',
                        'style': {
                            'text-valign': 'top',
                            # don't show variable when moving value!
                            'background-color': 'white'
                        }
                    },
                    {
                        'selector': '.base_input',
                        'style': {
                            'background-color': colors.BASE_INPUT,
                            'color': 'white',
                            'text-valign': 'center'
                        }
                    },
                    {
                        'selector': '.base_value',
                        'style': {
                            'background-color': colors.BASE_OUTPUT,
                            'text-valign': 'center'
                        }
                    },
                    {
                        'selector': '.source_input',
                        'style': {
                            'background-color': colors.SOURCE_INPUT,
                            'color': 'white',
                            'text-valign': 'center',
                        }
                    },
                    {
                        'selector': '.source_value',
                        'style': {
                            'background-color': colors.SOURCE_OUTPUT,
                            'text-valign': 'center',
                        }
                    },
                    {
                        'selector': '.counterfactual_value',
                        'style': {
                            'background-color': colors.COUNTERFACTUAL_OUTPUT,
                            'text-valign': 'center',
                        }
                    }
                ]
            ),
            # hidden trigger for onload so we can add nodes with values
            html.Div(id='hidden-onload-trigger', style={'display': 'none'})
        ])

        @app.callback(
            [
                Output('causal-graph-visualization', 'layout'), 
                Output('causal-graph-visualization', 'elements')
            ],
            [Input('hidden-onload-trigger', 'children')],
            [
                State('causal-graph-visualization', 'elements')
            ]
        )
        def onload(_, elements):
            """
            Add values as inner text of variables nodes.
            Needs to be done after positions have been rendered.
            """
            # update layout to preset to fix node positions
            layout = {'name': 'preset'}
            for variable, value in outputs.items():
                if variable in intervention: # intervention first bc it overrides base
                    classes = 'source_input'
                elif variable in inputs:
                    classes = 'base_input'
                elif variable in intervention_only:
                    classes = 'source_value'
                elif variable in counterfactual:
                    classes = 'counterfactual_value'
                else: # variable wasn't affected by intervention, so it's a base value
                    classes = 'base_value'

                # get original variable node
                variable_node = [e for e in elements if e['data']['id'] == variable][0]
                elements.append({
                    'data': {'id': f'{variable}-value', 'label': value},
                    'position': variable_node['position'],
                    'classes': classes
                })
            return layout, elements

        @app.callback(
            Output('causal-graph-visualization', 'elements', allow_duplicate=True),
            Input('causal-graph-visualization', 'elements'),
            prevent_initial_call=True
        )
        def on_moving_nodes(elements):
            for var in self.variables:
                variable_node = [e for e in elements if e['data']['id'] == var][0]
                value_node = [e for e in elements if e['data']['id'] == f'{var}-value'][0]
                variable_node['position'] = value_node['position']
            return elements

        app.run()

    def display_interchange(self, inputs, counterfactual_inputs, colors=DEFAULT_COLORS):
        """Visualize an interchange intervention between a base input and counterfactual input(s)"""
        outputs = self.run_interchange(inputs, counterfactual_inputs)

        intervention_only = self._get_descendants(counterfactual_inputs)
        counterfactual = self._get_descendants(counterfactual_inputs, strict=False)

        # CODE
        # set up base causal graph
        variables = [
            {
                'data': {
                    'id': var, 
                    'label': var
                },
                'position': {'x': 0, 'y': 0}, # updated by automatic layout
                'classes': 'variable' # as opposed to their inner values
            }
            for var in self.variables
        ]
        edges = [
            {'data': {'id': f'{parent}->{child}', 'source': parent, 'target': child}}
            for child in self.variables
            for parent in self.parents[child]
        ]

        # set up source causal graphs
        for i, source in enumerate(counterfactual_inputs.keys()):
            source_variables = [
                {
                    'data': {
                        'id': f'{var}-source-{i}', 
                        'label': var
                    },
                    'position': {'x': 0, 'y': 0}, # updated by automatic layout
                    'classes': 'variable' # as opposed to their inner values
                }
                for var in self.variables
            ]
            source_edges = [
                {'data': {'id': f'{parent}->{child}-source-{i}', 'source': f'{parent}-source-{i}', 'target': f'{child}-source-{i}'}}
                for child in self.variables
                for parent in self.parents[child]
            ]
            intervention_edge = [{
                'data': {'id': f'interchange-{i}', 'source': f'{source}-source-{i}', 'target': source},
                'classes': 'interchange_edge'
            }]
            variables += source_variables
            edges += source_edges + intervention_edge
            
        # set raw input as source for each graph
        roots = ['#raw_input'] + [f'#raw_input-source-{i}' for i in range(len(counterfactual_inputs))]

        app = Dash()

        app.layout = html.Div([
            cyto.Cytoscape(
                id='causal-graph-visualization',
                elements=variables + edges,
                layout={
                    'name': 'breadthfirst',
                    'roots': ','.join(roots)
                },
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'content': 'data(label)',
                            'width': 50,
                            'height': 50
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'curve-style': 'straight',
                            'target-arrow-shape': 'triangle',
                        }
                    },
                    {
                        'selector': '.variable',
                        'style': {
                            'text-valign': 'top',
                            'background-color': 'white'
                        }
                    },
                    # color variables by value
                    {
                        'selector': '.base_input',
                        'style': {
                            'background-color': colors.BASE_INPUT,
                            'color': 'white',
                            'text-valign': 'center'
                        }
                    },
                    {
                        'selector': '.base_value',
                        'style': {
                            'background-color': colors.BASE_OUTPUT,
                            'text-valign': 'center'
                        }
                    },
                    {
                        'selector': '.source_input',
                        'style': {
                            'background-color': colors.SOURCE_INPUT,
                            'color': 'white',
                            'text-valign': 'center',
                        }
                    },
                    {
                        'selector': '.source_value',
                        'style': {
                            'background-color': colors.SOURCE_OUTPUT,
                            'text-valign': 'center',
                        }
                    },
                    {
                        'selector': '.counterfactual_value',
                        'style': {
                            'background-color': colors.COUNTERFACTUAL_OUTPUT,
                            'text-valign': 'center',
                        }
                    },
                    # color interchange arrows
                    {
                        'selector': '.interchange_edge',
                        'style': {
                            'curve-style': 'unbundled-bezier',
                            'line-color': colors.SOURCE_INPUT,
                            'target-arrow-color': colors.SOURCE_INPUT,
                            'control-point-distances': '80 -80 80'
                        }
                    },
                ]
            ),
            # hidden trigger for onload so we can add nodes with values
            html.Div(id='hidden-onload-trigger', style={'display': 'none'})
        ])

        @app.callback(
            [
                Output('causal-graph-visualization', 'layout'), 
                Output('causal-graph-visualization', 'elements')
            ],
            [Input('hidden-onload-trigger', 'children')],
            [State('causal-graph-visualization', 'elements')]
        )
        def onload(_, elements):
            """
            Add values as inner text of variables nodes.
            Needs to be done after positions have been rendered.
            """
            # update layout to preset to fix node positions
            layout = {'name': 'preset'}

            # set up source graphs
            for i, source_inputs in enumerate(counterfactual_inputs.values()):
                source_outputs = self.run_forward(source_inputs)
                for variable, value in source_outputs.items():
                    if variable in source_inputs:
                        classes = 'source_input'
                    else:
                        classes = 'source_value'
                    
                    variable_node = [e for e in elements if e['data']['id'] == f'{variable}-source-{i}'][0]
                    elements.append({
                        'data': {'id': f'{variable}-source-{i}-value', 'label': value},
                        'position': variable_node['position'],
                        'classes': classes
                    })
                    
            # set up base graphs
            for variable, value in outputs.items():
                if variable in counterfactual_inputs: # intervention first bc it overrides base
                    classes = 'source_input'
                elif variable in inputs:
                    classes = 'base_input'
                elif variable in intervention_only:
                    classes = 'source_value'
                elif variable in counterfactual:
                    classes = 'counterfactual_value'
                else: # variable wasn't affected by intervention, so it's a base value
                    classes = 'base_value'

                # get original variable node
                variable_node = [e for e in elements if e['data']['id'] == variable][0]
                elements.append({
                    'data': {'id': f'{variable}-value', 'label': value},
                    'position': variable_node['position'],
                    'classes': classes
                })
            return layout, elements

        @app.callback(
            Output('causal-graph-visualization', 'elements', allow_duplicate=True),
            Input('causal-graph-visualization', 'elements'),
            prevent_initial_call=True
        )
        def on_moving_nodes(elements):
            """make sure variable nodes follow their respective value nodes"""
            for variable_node in elements:
                var_id = variable_node['data']['id']
                if not var_id.endswith('value') and 'label' in variable_node['data']:
                    value_node = [e for e in elements if e['data']['id'] == f'{var_id}-value'][0]
                    variable_node['position'] = value_node['position']
            return elements

        app.run()

    def print_structure(self, font=12, node_size=1000):
        """
        Print the graph structure of the causal model.
        
        Parameters:
        -----------
        font : int, optional
            Font size for node labels (default is 12).
        node_size : int, optional
            Size of nodes in the graph (default is 1000).
        """
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (parent, child)
                for child in self.variables
                for parent in self.parents[child]
            ]
        )
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(
            graph, 
            with_labels=True, 
            node_color="green", 
            pos=self.print_pos, 
            font_size=font, 
            node_size=node_size
        )
        plt.show()

    def print_setting(self, total_setting, font=12, node_size=1000, var_names=False):
        """
        Print the graph with variable values.
        
        Parameters:
        -----------
        total_setting : dict
            A dictionary mapping variables to their values.
        font : int, optional
            Font size for node labels (default is 12).
        node_size : int, optional
            Size of nodes in the graph (default is 1000).
        var_names : bool, optional
            Whether to include variable names in labels (default is False).
        """
        relabeler = {
            var: var + ":\n " + str(total_setting[var]) for var in self.variables
        }
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (parent, child)
                for child in self.variables
                for parent in self.parents[child]
            ]
        )
        plt.figure(figsize=(10, 10))
        graph = nx.relabel_nodes(graph, relabeler)
        newpos = dict()
        if self.print_pos is not None:
            for var in self.print_pos:
                newpos[relabeler[var]] = self.print_pos[var]
        nx.draw_networkx(
            graph, 
            with_labels=True, 
            node_color="green", 
            pos=newpos, 
            font_size=font, 
            node_size=node_size
        )
        plt.show()

    def generate_equiv_classes(self):
        """
        Generate equivalence classes for each variable.
        
        This method computes, for each non-input variable, the sets of parent values
        that produce each possible value of the variable.
        """
        for var in self.variables:
            if var in self.inputs or var in self.equiv_classes:
                continue
            self.equiv_classes[var] = {val: [] for val in self.values[var]}
            for parent_values in itertools.product(
                *[self.values[par] for par in self.parents[var]]
            ):
                value = self.mechanisms[var](*parent_values)
                self.equiv_classes[var][value].append(
                    {par: parent_values[i] for i, par in enumerate(self.parents[var])}
                )

    def find_live_paths(self, intervention):
        """
        Find all live causal paths in the model given an intervention.
        
        A live path is a sequence of variables where changing the value of one
        variable can affect the value of the next variable in the sequence.
        
        Parameters:
        -----------
        intervention : dict
            A dictionary mapping variables to their intervened values.
            
        Returns:
        --------
        dict
            A dictionary mapping path lengths to lists of paths.
        """
        actual_setting = self.run_forward(intervention)
        paths = {1: [[variable] for variable in self.variables]}
        step = 2
        while True:
            paths[step] = []
            for path in paths[step - 1]:
                for child in self.children[path[-1]]:
                    actual_cause = False
                    for value in self.values[path[-1]]:
                        newintervention = copy.deepcopy(intervention)
                        newintervention[path[-1]] = value
                        counterfactual_setting = self.run_forward(newintervention)
                        if counterfactual_setting[child] != actual_setting[child]:
                            actual_cause = True
                    if actual_cause:
                        paths[step].append(copy.deepcopy(path) + [child])
            if len(paths[step]) == 0:
                break
            step += 1
        del paths[1]
        return paths

    def sample_input_tree_balanced(self, output_var=None, output_var_value=None):
        """
        Sample an input that leads to a specific output value using a balanced tree approach.
        
        Parameters:
        -----------
        output_var : str, optional
            The output variable to target (default is the first output variable).
        output_var_value : any, optional
            The desired value for the output variable (default is a random choice).
            
        Returns:
        --------
        dict
            A dictionary mapping input variables to their sampled values.
        """
        assert output_var is not None or len(self.outputs) == 1
        self.generate_equiv_classes()

        if output_var is None:
            output_var = self.outputs[0]
        if output_var_value is None:
            output_var_value = random.choice(self.values[output_var])

        def create_input(var, value, input_dict={}):
            """
            Recursively create an input that leads to the specified value for a variable.
            
            Parameters:
            -----------
            var : str
                The variable to target.
            value : any
                The desired value for the variable.
            input_dict : dict, optional
                The input dictionary to build upon (default is an empty dictionary).
                
            Returns:
            --------
            dict
                The updated input dictionary.
            """
            parent_values = random.choice(self.equiv_classes[var][value])
            for parent in parent_values:
                if parent in self.inputs:
                    input_dict[parent] = parent_values[parent]
                else:
                    create_input(parent, parent_values[parent], input_dict)
            return input_dict

        input_setting = create_input(output_var, output_var_value)
        for input_var in self.inputs:
            if input_var not in input_setting:
                input_setting[input_var] = random.choice(self.values[input_var])
        return input_setting

    def get_path_maxlen_filter(self, lengths):
        """
        Get a filter function that checks if the maximum length of any live path
        is in a given set of lengths.
        
        Parameters:
        -----------
        lengths : list or set
            A list or set of path lengths to check against.
            
        Returns:
        --------
        function
            A filter function that takes a setting and returns a boolean.
        """
        def check_path(total_setting):
            """
            Check if the maximum length of any live path is in the specified lengths.
            
            Parameters:
            -----------
            total_setting : dict
                A dictionary mapping variables to their values.
                
            Returns:
            --------
            bool
                True if the maximum length is in the specified lengths, False otherwise.
            """
            input_setting = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input_setting)
            max_len = max([l for l in paths.keys() if len(paths[l]) != 0])
            if max_len in lengths:
                return True
            return False

        return check_path

    def get_partial_filter(self, partial_setting):
        """
        Get a filter function that checks if a setting matches a partial setting.
        
        Parameters:
        -----------
        partial_setting : dict
            A dictionary mapping variables to their desired values.
            
        Returns:
        --------
        function
            A filter function that takes a setting and returns a boolean.
        """
        def compare(total_setting):
            """
            Check if a setting matches the partial setting.
            
            Parameters:
            -----------
            total_setting : dict
                A dictionary mapping variables to their values.
                
            Returns:
            --------
            bool
                True if the setting matches the partial setting, False otherwise.
            """
            for var in partial_setting:
                if total_setting[var] != partial_setting[var]:
                    return False
            return True

        return compare

    def get_specific_path_filter(self, start, end):
        """
        Get a filter function that checks if there is a live path from a start
        variable to an end variable.
        
        Parameters:
        -----------
        start : str
            The start variable of the path.
        end : str
            The end variable of the path.
            
        Returns:
        --------
        function
            A filter function that takes a setting and returns a boolean.
        """
        def check_path(total_setting):
            """
            Check if there is a live path from the start variable to the end variable.
            
            Parameters:
            -----------
            total_setting : dict
                A dictionary mapping variables to their values.
                
            Returns:
            --------
            bool
                True if there is a live path from start to end, False otherwise.
            """
            input_setting = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input_setting)
            for k in paths:
                for path in paths[k]:
                    if path[0] == start and path[-1] == end:
                        return True
            return False

        return check_path

    def _get_descendants(self, intervention, strict=True):
        """breadth-first search, include only variables directly affected by intervention"""
        descendants = [v for v in intervention]
        current_paths = [v for v in intervention]
        covered = [v for v in intervention]
        while current_paths:
            variable = current_paths.pop(0)
            for c in self.children[variable]:
                if c in covered: # don't loop
                    continue
                covered.append(c)
                if all(p in descendants for p in self.parents[c]) or not strict:
                    descendants.append(c)
                    current_paths.append(c)
        return descendants

def simple_example():
    """
    Run a simple example of a causal model.
    
    This creates a small causal model with three variables plus raw_input/raw_output
    and runs it with and without interventions.
    """
    variables = ["A", "B", "C", "raw_input", "raw_output"]
    values = {
        "A": [True, False],
        "B": [True, False], 
        "C": [True, False],
        "raw_input": None,
        "raw_output": None
    }
    parents = {
        "A": [], 
        "B": [], 
        "C": ["A", "B"],
        "raw_input": ["A", "B"],  # raw_input depends on input variables
        "raw_output": ["C"]       # raw_output depends on output variables
    }

    def A():
        return True

    def B():
        return False

    def C(a, b):
        return a and b
    
    def raw_input(a, b):
        return f"Input: A={a}, B={b}"
    
    def raw_output(c):
        return f"Output: C={c}"

    mechanisms = {
        "A": A, 
        "B": B, 
        "C": C,
        "raw_input": raw_input,
        "raw_output": raw_output
    }
    
    model = CausalModel(variables, values, parents, mechanisms)
    model.print_structure()
    
    print("No intervention:")
    result = model.run_forward()
    print(result)
    print(f"Raw input: {result['raw_input']}")
    print(f"Raw output: {result['raw_output']}")
    print()
    
    model.print_setting(result)
    
    print("Intervention setting A and B to TRUE:")
    intervention_result = model.run_forward({"A": True, "B": True})
    print(intervention_result)
    print(f"Raw input: {intervention_result['raw_input']}")
    print(f"Raw output: {intervention_result['raw_output']}")
    
    print("Timesteps:", model.timesteps)


if __name__ == "__main__":
    simple_example()