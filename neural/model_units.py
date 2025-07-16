"""
model_units.py
==============
Abstractions for locating *components* and *features* inside a transformer
model.  A **Component** specifies *where* in the network (layer, tensor-slot)
to intervene; a **Featurizer** then specifies *how* to access a particular
representation space at that location.

This file introduces:

* `ComponentIndexer` – a callable that yields dynamic indices, e.g., token 
  positions in a transformer.
* `Component` / `StaticComponent` – address a tensor slice inside the model.
* `AtomicModelUnit` – pairs a Component with a Featurizer (+ optional
  feature-subset).
"""

from typing import List, Union, Optional

import pyvene as pv

from neural.featurizers import Featurizer, SubspaceFeaturizer, SubspaceFeaturizerRevNet 


class ComponentIndexer:
    """Callable wrapper that returns location indices for a given *input*.
    This is used to specify the *where* in the model to intervene, e.g.,
    the *input* might be a batch of tokenized text with indices that are 
    the positions of the tokens to be intervened upon.
    """

    def __init__(self, indexer, id: str = "null"):
        """
        Parameters
        ----------
        indexer :
            A function `input -> List[int]` returning the indices.
        id :
            Human-readable identifier for diagnostics / printing.
        """
        self.indexer = indexer
        self.id = id

    # ------------------------------------------------------------------ #
    def index(self, input, batch=False):
        """Return indices for *input* by delegating to wrapped function."""
        if batch:
            return [self.indexer(i) for i in input]
        return self.indexer(input)

    # ------------------------------------------------------------------ #
    def __repr__(self):
        return f"ComponentIndexer(id='{self.id}')"


class Component:
    """Dynamic component inside a model (layer + tensor location)."""

    def __init__(
        self,
        layer: int,
        component_type: str,
        indices_func: Union[ComponentIndexer, List[int]],
        unit: str = "pos",
    ):
        """
        Parameters
        ----------
        layer :
            Layer number in the *IntervenableModel*.
        component_type :
            E.g. 'block_input', 'mlp', 'head_attention_value_output', etc.
        indices_func :
            Either a `ComponentIndexer` **or** a static list of indices.
        unit :
            String describing the dimension being indexed (e.g. 'pos' or
            'h.pos' for attention head · token position).
        """
        self.component_type = component_type
        self.unit = unit
        self.layer = layer

        # Normalise to an indexer
        if isinstance(indices_func, list):
            constant = indices_func

            def _const(_):
                return constant

            self._indices_func = ComponentIndexer(_const, id=f"constant_{constant}")
        else:
            self._indices_func = indices_func

    def get_layer(self) -> int:
        return self.layer

    def set_layer(self, layer: int):
        self.layer = layer

    def get_index_id(self) -> str:
        return self._indices_func.id

    def index(self, input, **kwargs) -> List:
        return self._indices_func.index(input, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"layer={self.layer}, "
            f"component_type='{self.component_type}', "
            f"indices={self._indices_func}, "
            f"unit='{self.unit}')"
        )

    def __eq__(self, other):
        return (
            isinstance(other, Component)
            and self.layer == other.layer
            and self.component_type == other.component_type
            and self.unit == other.unit
            and self._indices_func.id == other._indices_func.id
        )

    def __hash__(self):
        return hash(
            (self.layer, self.component_type, self.unit, self._indices_func.id)
        )


class StaticComponent(Component):
    """Component whose indices are constants."""

    def __init__(
        self,
        layer: int,
        component_type: str,
        component_indices: List[int],
        unit: str = "pos",
    ):
        super().__init__(layer, component_type, component_indices, unit)


class AtomicModelUnit:
    """A (Component, Featurizer, feature indices) triple.
    
    This is the basic unit of intervention in this library.
    It specifies a *location* in the model (Component) 
    and a *feature space* (Featurizer) to be used for intervention.
    The `feature_indices` are an optional subset of the features
    returned by the featurizer.  If not specified, all features
    are used.  
    
    The `shape` is reserved for downstream sub-space
    creation helpers.  The `id` is a human-readable identifier
    for the unit, used for diagnostics and printing.
    """

    def __init__(
        self,
        component: Component,
        featurizer: Optional[Featurizer] = None,
        feature_indices: Optional[List[int]] = None,
        shape=None,
        *,
        id: str = "null",
    ):
        """
        Parameters
        ----------
        component :
            A `Component` instance specifying *where* in the network.
        featurizer :
            A `Featurizer` (defaults to identity featurizer).
        feature_indices :
            Optional subset of indices inside the featurizer’s feature vector.
            Will be bounds-checked against `featurizer.n_features`.
        shape :
            Reserved for downstream sub-space creation helpers.
        id :
            Diagnostic identifier.
        """
        self.id = id
        self.component = component
        self.featurizer = featurizer or Featurizer()
        self.shape = shape

        # Bounds-check feature indices
        self.feature_indices: Optional[List[int]] = None
        if feature_indices is not None:
            self.set_feature_indices(feature_indices)

    # ------------------------ Feature helpers -------------------------- #
    def get_shape(self):
        return self.shape

    def index_component(self, input, **kwargs):
        return self.component.index(input, **kwargs)

    def get_feature_indices(self):
        return self.feature_indices

    def set_feature_indices(self, feature_indices: List[int]):
        """Assign `feature_indices` after validating bounds."""
        if (
            self.featurizer.n_features is not None
            and feature_indices is not None
            and len(feature_indices) > 0
            and max(feature_indices) >= self.featurizer.n_features
        ):
            raise ValueError(
                f"Feature index {max(feature_indices)} exceeds "
                f"featurizer dimensionality {self.featurizer.n_features}"
            )
        self.feature_indices = feature_indices

    def set_featurizer(self, featurizer: Featurizer):
        """Swap in a new featurizer (re-checking feature bounds)."""
        self.featurizer = featurizer
        if self.feature_indices is not None:
            self.set_feature_indices(self.feature_indices)

    # ------------------------ PyVENE helpers -------------------------- #
    def is_static(self):
        return isinstance(self.component, StaticComponent)

    def create_intervention_config(self, group_key, intervention_type):
        """Return PyVENE config dict for this unit + featurizer."""
        config = {
            "component": self.component.component_type,
            "unit": self.component.unit,
            "layer": self.component.layer,
            "group_key": group_key,
        }
        if intervention_type == "interchange":
            config["intervention_type"] = self.featurizer.get_interchange_intervention()
        elif intervention_type == "collect":
            config["intervention_type"] = self.featurizer.get_collect_intervention()
        elif intervention_type == "mask":
            config["intervention_type"] = self.featurizer.get_mask_intervention()
        else:
            raise ValueError(f"Unknown intervention type '{intervention_type}'.")

        return config

    def __repr__(self):
        return f"AtomicModelUnit(id='{self.id}')"

    # ---------------- Utility & misc ----------------------------------- #
    def set_layer(self, layer: int):
        self.component.layer = layer

    def get_layer(self):
        return self.component.layer
