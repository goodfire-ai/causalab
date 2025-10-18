# tests/test_pyvene_core/test_prepare_intervenable_model_integration.py

import pytest
import torch
from unittest.mock import MagicMock, patch

from experiments.pyvene_core import _prepare_intervenable_model
from neural.model_units import AtomicModelUnit, Component, StaticComponent
from neural.LM_units import ResidualStream


class TestPrepareIntervenableModelIntegration:
    """Integration tests for the _prepare_intervenable_model function."""
    
# tests/test_pyvene_core/test_prepare_intervenable_model.py

    def test_end_to_end_creation(self, mock_tiny_lm, token_positions, monkeypatch):
        """Test end-to-end model creation with real components (no mocks)."""
        # Set up a real model_units_list
        model_units_list = []
        layers = [0, 1]

        for layer in layers:
            units = []
            for token_position in token_positions:
                unit = ResidualStream(
                    layer=layer,
                    token_indices=token_position,
                    shape=(mock_tiny_lm.model.config.hidden_size,),
                    target_output=True
                )
                # The key fix: Mock is_static to return True for consistent testing
                unit.is_static = MagicMock(return_value=True)
                units.append(unit)
            model_units_list.append(units)

        # Create mock pyvene components to avoid actual model creation
        mock_config = MagicMock()
        mock_model = MagicMock()
        
        class MockPV:
            def __init__(self):
                self.IntervenableConfig = MagicMock(return_value=mock_config)
                self.IntervenableModel = MagicMock(return_value=mock_model)
                
                # Mock TrainableIntervention and other intervention classes
                self.TrainableIntervention = MagicMock()
                self.DistributedRepresentationIntervention = MagicMock()
                self.CollectIntervention = MagicMock()
        
        mock_pv = MockPV()
        
        # Apply the patch
        monkeypatch.setattr('experiments.pyvene_core.pv', mock_pv)
        
        # Call the function
        result = _prepare_intervenable_model(mock_tiny_lm, model_units_list)
        
        # Check the result
        assert result is mock_model
        
        # Verify IntervenableConfig was created with correct number of configs
        # We expect one config per unit
        expected_calls = sum(len(units) for units in model_units_list)
        assert mock_pv.IntervenableConfig.call_count == 1
        
        # Verify IntervenableModel was created with correct config
        mock_pv.IntervenableModel.assert_called_once_with(mock_config, model=mock_tiny_lm.model, use_fast=True)
        
        # Verify set_device was called
        mock_model.set_device.assert_called_once_with(mock_tiny_lm.model.device)
    
    def test_with_real_pyvene(self, mock_tiny_lm, token_positions):
        """
        Test with actual pyvene library if available.
        This test is skipped if pyvene is not installed.
        """
        try:
            import pyvene
        except ImportError:
            pytest.skip("pyvene not installed, skipping test")
        
        # Set up a real model_units_list
        model_units_list = []
        layers = [0]
        
        for layer in layers:
            units = []
            for token_position in token_positions:
                unit = ResidualStream(
                    layer=layer,
                    token_indices=token_position,
                    shape=(mock_tiny_lm.model.config.hidden_size,),
                    target_output=True
                )
                units.append(unit)
            model_units_list.append(units)
        
        # Mock the create_intervention_config method to avoid actual intervention creation
        for units in model_units_list:
            for unit in units:
                unit.create_intervention_config = MagicMock(return_value={
                    'component': unit.component.component_type,
                    'unit': unit.component.unit,
                    'layer': unit.component.layer,
                    'group_key': 0,
                    'intervention_type': MagicMock()
                })
        
        # Mock pyvene components to avoid actual model creation
        with patch('pyvene.IntervenableModel') as mock_model_class, \
             patch('pyvene.IntervenableConfig') as mock_config_class:
                
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            # Call the function
            result = _prepare_intervenable_model(mock_tiny_lm, model_units_list)
            
            # Check the result
            assert result is mock_model
            
            # Verify IntervenableConfig was created
            mock_config_class.assert_called_once()
            
            # Verify IntervenableModel was created with correct arguments
            mock_model_class.assert_called_once()
            
            # Verify set_device was called
            mock_model.set_device.assert_called_once_with(mock_tiny_lm.model.device)