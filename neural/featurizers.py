"""
featurizers.py
==============
Utility classes for defining *invertible* feature spaces on top of a model’s
hidden-state tensors, together with intervention helpers that operate inside
those spaces.

Key ideas
---------

* **Featurizer** – a lightweight wrapper holding:
    • a forward `featurizer` module that maps a tensor **x → (f, error)**  
      where *error* is the reconstruction residual (useful for lossy
      featurizers such as sparse auto-encoders);  
    • an `inverse_featurizer` that re-assembles the original space  
      **(f, error) → x̂**.

* **Interventions** – three higher-order factory functions build PyVENE
  interventions that work in the featurized space:
    - *interchange*
    - *collect*
    - *mask* (differential binary masking)

All public classes / functions below carry PEP-257-style doc-strings.
"""

from typing import Optional, Tuple

import torch
import pyvene as pv


# --------------------------------------------------------------------------- #
#  Basic identity featurizers                                                 #
# --------------------------------------------------------------------------- #
class IdentityFeaturizerModule(torch.nn.Module):
    """A no-op featurizer: *x → (x, None)*."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return x, None


class IdentityInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`IdentityFeaturizerModule`."""

    def forward(self, x: torch.Tensor, error: None) -> torch.Tensor:  # noqa: D401
        return x


# --------------------------------------------------------------------------- #
#  High-level Featurizer wrapper                                              #
# --------------------------------------------------------------------------- #
class Featurizer:
    """Container object holding paired featurizer and inverse modules.

    Parameters
    ----------
    featurizer :
        A `torch.nn.Module` mapping **x → (features, error)**.
    inverse_featurizer :
        A `torch.nn.Module` mapping **(features, error) → x̂**.
    n_features :
        Dimensionality of the feature space.  **Required** when you intend to
        build a *mask* intervention; optional otherwise.
    id :
        Human-readable identifier used by `__str__` methods of the generated
        interventions.
    """

    # --------------------------------------------------------------------- #
    #  Construction / public accessors                                      #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        featurizer: torch.nn.Module = IdentityFeaturizerModule(),
        inverse_featurizer: torch.nn.Module = IdentityInverseFeaturizerModule(),
        *,
        n_features: Optional[int] = None,
        id: str = "null",
        tie_masks: bool = False,
    ):
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer
        self.n_features = n_features
        self.id = id
        self.tie_masks = tie_masks

    # -------------------- Intervention builders -------------------------- #
    def get_interchange_intervention(self):
        if not hasattr(self, "_interchange_intervention"):
            self._interchange_intervention = build_feature_interchange_intervention(
                self.featurizer, self.inverse_featurizer, self.id
            )
        return self._interchange_intervention

    def get_collect_intervention(self):
        if not hasattr(self, "_collect_intervention"):
            self._collect_intervention = build_feature_collect_intervention(
                self.featurizer, self.id
            )
        return self._collect_intervention

    def get_mask_intervention(self):
        if self.n_features is None:
            raise ValueError(
                "`n_features` must be provided on the Featurizer "
                "to construct a mask intervention."
            )
        if not hasattr(self, "_mask_intervention"):
            self._mask_intervention = build_feature_mask_intervention(
                self.featurizer,
                self.inverse_featurizer,
                self.n_features,
                self.id,
                self.tie_masks,
            )
        return self._mask_intervention

    # ------------------------- Convenience I/O --------------------------- #
    def featurize(self, x: torch.Tensor):
        return self.featurizer(x)

    def inverse_featurize(self, x: torch.Tensor, error):
        return self.inverse_featurizer(x, error)

    # --------------------------------------------------------------------- #
    #  (De)serialisation helpers                                            #
    # --------------------------------------------------------------------- #
    def save_modules(self, path: str) -> Tuple[str, str]:
        """Serialise featurizer & inverse to `<path>_{featurizer, inverse}`.

        Notes
        -----
        * **SAE featurizers** are *not* serialisable: a
          :class:`NotImplementedError` is raised.
        * Existing files will be *silently overwritten*.
        """
        featurizer_class = self.featurizer.__class__.__name__

        if featurizer_class == "SAEFeaturizerModule":
            #SAE featurizers are to be loaded from sae_lens
            return None, None

        inverse_featurizer_class = self.inverse_featurizer.__class__.__name__

        # Extra config needed for Subspace featurizers
        additional_config = {}
        if featurizer_class == "SubspaceFeaturizerModule":
            additional_config["rotation_matrix"] = (
                self.featurizer.rotate.weight.detach().clone()
            )
            additional_config["requires_grad"] = (
                self.featurizer.rotate.weight.requires_grad
            )

        model_info = {
            "featurizer_class": featurizer_class,
            "inverse_featurizer_class": inverse_featurizer_class,
            "n_features": self.n_features,
            "additional_config": additional_config,
        }

        torch.save(
            {"model_info": model_info, "state_dict": self.featurizer.state_dict()},
            f"{path}_featurizer",
        )
        torch.save(
            {
                "model_info": model_info,
                "state_dict": self.inverse_featurizer.state_dict(),
            },
            f"{path}_inverse_featurizer",
        )
        return f"{path}_featurizer", f"{path}_inverse_featurizer"

    @classmethod
    def load_modules(cls, path: str) -> "Featurizer":
        """Inverse of :meth:`save_modules`.

        Returns
        -------
        Featurizer
            A *new* instance with reconstructed modules and metadata.
        """
        featurizer_data = torch.load(f"{path}_featurizer")
        inverse_data = torch.load(f"{path}_inverse_featurizer")

        model_info = featurizer_data["model_info"]
        featurizer_class = model_info["featurizer_class"]

        if featurizer_class == "SubspaceFeaturizerModule":
            rot = model_info["additional_config"]["rotation_matrix"]
            requires_grad = model_info["additional_config"]["requires_grad"]

            # Re-build a parametrised orthogonal layer with identical shape.
            in_dim, out_dim = rot.shape
            rotate_layer = pv.models.layers.LowRankRotateLayer(
                in_dim, out_dim, init_orth=False
            )
            rotate_layer.weight.data.copy_(rot)
            rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            rotate_layer.requires_grad_(requires_grad)

            featurizer = SubspaceFeaturizerModule(rotate_layer)
            inverse = SubspaceInverseFeaturizerModule(rotate_layer)

            # Sanity-check weight shape
            assert (
                featurizer.rotate.weight.shape == rot.shape
            ), "Rotation-matrix shape mismatch after deserialisation."
        elif featurizer_class == "IdentityFeaturizerModule":
            featurizer = IdentityFeaturizerModule()
            inverse = IdentityInverseFeaturizerModule()
        else:
            raise ValueError(f"Unknown featurizer class '{featurizer_class}'.")

        featurizer.load_state_dict(featurizer_data["state_dict"])
        inverse.load_state_dict(inverse_data["state_dict"])

        return cls(
            featurizer,
            inverse,
            n_features=model_info["n_features"],
            id=model_info.get("featurizer_id", "loaded"),
        )


# --------------------------------------------------------------------------- #
#  Intervention factory helpers                                               #
# --------------------------------------------------------------------------- #
def build_feature_interchange_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    featurizer_id: str,
):
    """Return a class implementing PyVENE’s TrainableIntervention."""

    class FeatureInterchangeIntervention(
        pv.TrainableIntervention, pv.DistributedRepresentationIntervention
    ):
        """Swap features between *base* and *source* in the featurized space."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer

        def forward(self, base, source, subspaces=None):
            f_base, base_err = self._featurizer(base)
            f_src, _ = self._featurizer(source)

            if subspaces is None or _subspace_is_all_none(subspaces):
                f_out = f_src
            else:
                f_out = pv.models.intervention_utils._do_intervention_by_swap(
                    f_base,
                    f_src,
                    "interchange",
                    self.interchange_dim,
                    subspaces,
                    subspace_partition=self.subspace_partition,
                    use_fast=self.use_fast,
                )
            return self._inverse(f_out, base_err).to(base.dtype)

        def __str__(self):  # noqa: D401
            return f"FeatureInterchangeIntervention(id={featurizer_id})"

    return FeatureInterchangeIntervention


def build_feature_collect_intervention(
    featurizer: torch.nn.Module, featurizer_id: str
):
    """Return a `CollectIntervention` operating in feature space."""

    class FeatureCollectIntervention(pv.CollectIntervention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer

        def forward(self, base, source=None, subspaces=None):
            f_base, _ = self._featurizer(base)
            return pv.models.intervention_utils._do_intervention_by_swap(
                f_base,
                source,
                "collect",
                self.interchange_dim,
                subspaces,
                subspace_partition=self.subspace_partition,
                use_fast=self.use_fast,
            )

        def __str__(self):  # noqa: D401
            return f"FeatureCollectIntervention(id={featurizer_id})"

    return FeatureCollectIntervention


def build_feature_mask_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    n_features: int,
    featurizer_id: str,
    tie_masks: bool = False,
):
    """Return a trainable mask intervention.

    Args:
        featurizer: The featurizer module
        inverse_featurizer: The inverse featurizer module
        n_features: Number of features in the featurized space
        featurizer_id: Human-readable identifier for the intervention
        tie_masks: If True, use a single scalar mask weight for all features.
                  If False, use per-feature mask weights (default behavior).
    """

    class FeatureMaskIntervention(pv.TrainableIntervention):
        """Differential-binary masking in the featurized space."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer
            self._tie_masks = tie_masks

            # Learnable parameters
            if tie_masks:
                # Single scalar mask weight for all features
                self.mask = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            else:
                # Per-feature mask weights
                self.mask = torch.nn.Parameter(torch.zeros(n_features), requires_grad=True)
            self.temperature: Optional[torch.Tensor] = None  # must be set by user

        # -------------------- API helpers -------------------- #
        def get_temperature(self) -> torch.Tensor:
            if self.temperature is None:
                raise ValueError("Temperature has not been set.")
            return self.temperature

        def set_temperature(self, temp: float | torch.Tensor):
            self.temperature = (
                torch.as_tensor(temp, dtype=self.mask.dtype).to(self.mask.device)
            )

        # ------------------------- forward ------------------- #
        def forward(self, base, source, subspaces=None):
            if self.temperature is None:
                raise ValueError("Cannot run forward without a temperature.")

            f_base, base_err = self._featurizer(base)
            f_src, _ = self._featurizer(source)

            # Align devices / dtypes
            mask = self.mask.to(f_base.device)
            temp = self.temperature.to(f_base.device)

            f_base = f_base.to(mask.dtype)
            f_src = f_src.to(mask.dtype)

            if self.training:
                gate = torch.sigmoid(mask / temp)
            else:
                gate = (torch.sigmoid(mask) > 0.5).float()

            f_out = (1.0 - gate) * f_base + gate * f_src
            return self._inverse(f_out.to(base.dtype), base_err).to(base.dtype)

        # ---------------- Sparsity regulariser --------------- #
        def get_sparsity_loss(self) -> torch.Tensor:
            if self.temperature is None:
                raise ValueError("Temperature has not been set.")
            gate = torch.sigmoid(self.mask / self.temperature)
            return torch.norm(gate, p=1)

        def __str__(self):  # noqa: D401
            tied_str = ",tied" if self._tie_masks else ""
            return f"FeatureMaskIntervention(id={featurizer_id}{tied_str})"

    return FeatureMaskIntervention


# --------------------------------------------------------------------------- #
#  Concrete featurizer implementations                                        #
# --------------------------------------------------------------------------- #
class SubspaceFeaturizerModule(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, rotate_layer: pv.models.layers.LowRankRotateLayer):
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, x: torch.Tensor):
        r = self.rotate.weight.T  # (out, in)ᵀ
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        return f, error


class SubspaceInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, rotate_layer: pv.models.layers.LowRankRotateLayer):
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, f, error):
        r = self.rotate.weight.T
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)


class SubspaceFeaturizer(Featurizer):
    """Orthogonal linear sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
        **kwargs,
    ):
        assert (
            shape is not None or rotation_subspace is not None
        ), "Provide either `shape` or `rotation_subspace`."

        if shape is not None:
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)
        else:
            shape = rotation_subspace.shape
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=False)
            rotate.weight.data.copy_(rotation_subspace)

        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)

        super().__init__(
            SubspaceFeaturizerModule(rotate),
            SubspaceInverseFeaturizerModule(rotate),
            n_features=rotate.weight.shape[1],
            id=id,
            **kwargs,
        )


class SAEFeaturizerModule(torch.nn.Module):
    """Wrapper around a *Sparse Autoencoder*’s encode() / decode() pair."""

    def __init__(self, sae):
        super().__init__()
        self.sae = sae

    def forward(self, x):
        features = self.sae.encode(x.to(self.sae.dtype))
        error = x - self.sae.decode(features).to(x.dtype)
        return features.to(x.dtype), error


class SAEInverseFeaturizerModule(torch.nn.Module):
    """Inverse for :class:`SAEFeaturizerModule`."""

    def __init__(self, sae):
        super().__init__()
        self.sae = sae

    def forward(self, features, error):
        return (
            self.sae.decode(features.to(self.sae.dtype)).to(features.dtype)
            + error.to(features.dtype)
        )


class SAEFeaturizer(Featurizer):
    """Featurizer backed by a pre-trained sparse auto-encoder.

    Notes
    -----
    Serialisation is *disabled* for SAE featurizers – saving will raise
    ``NotImplementedError``.
    """

    def __init__(self, sae, *, trainable: bool = False, **kwargs):
        sae.requires_grad_(trainable)
        super().__init__(
            SAEFeaturizerModule(sae),
            SAEInverseFeaturizerModule(sae),
            n_features=sae.cfg.to_dict()["d_sae"],
            id="sae",
            **kwargs,
        )


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def _subspace_is_all_none(subspaces) -> bool:
    """Return ``True`` if *every* element of *subspaces* is ``None``."""
    return subspaces is None or all(
        inner is None or all(elem is None for elem in inner) for inner in subspaces
    )
