import pytest
import datetime
import numpy as np
from ..simoc_abm.agents import LampAgent

class MockModel:
    agents = {}
    time = datetime.datetime(2020, 1, 1, 0)

class MockAgent:
    def __init__(self, lamp_id, photoperiod, par_baseline, active):
        self.flows = {'in': {'par': {'connections': [lamp_id]}}}
        self.properties = {'photoperiod': {'value': photoperiod},
                           'par_baseline': {'value': par_baseline}}
        self.active = active

@pytest.fixture
def mock_model():
    model = MockModel()
    model.agents = {
        'lamp': LampAgent(model, 'lamp'),
        'test_plant_1': MockAgent('lamp', 13, 1.5, 2),
        'test_plant_2': MockAgent('lamp', 15, 1.2, 2),
    }
    return model

class TestAgentLamp:
    def test_agent_lamp_init(self):
        """Initialize attributes properly"""
        model = object()
        lamp = LampAgent(model, 'lamp')
        for k, v in LampAgent.default_attributes.items():
            assert lamp.attributes[k] == v

    def test_agent_lamp_register_full(self, mock_model):
        """Register lamp with model"""
        lamp = mock_model.agents['lamp']
        lamp.register(mock_model)
        assert len(lamp.connected_plants) == 2
        assert all(isinstance(p, str) for p in lamp.connected_plants)
        assert lamp.active == 4

    def test_agent_lamp_register_empty(self, mock_model):
        del mock_model.agents['test_plant_1']
        del mock_model.agents['test_plant_2']
        lamp = mock_model.agents['lamp']
        lamp.register(mock_model)
        assert len(lamp.connected_plants) == 0
        assert lamp.active == 0

    def test_agent_lamp_update_attributes(self, mock_model):
        lamp = mock_model.agents['lamp']
        lamp.register(mock_model)

        assert lamp.connected_plants == ['test_plant_1', 'test_plant_2']
        assert lamp.lamp_configuration == {'test_plant_1': 2, 'test_plant_2': 2}
        assert lamp.attributes['photoperiod'] == 15
        expected_par_rate = 1.5 * 24/13  # Highest instantaneous of any connected plant
        assert lamp.attributes['par_rate'] == expected_par_rate
        expected_daily_growth = np.zeros(24)
        expected_daily_growth[4:19] = expected_par_rate
        assert str(lamp.daily_growth) == str(expected_daily_growth)
        assert lamp.active == 4

    def test_agent_lamp_step(self, mock_model):
        lamp = mock_model.agents['lamp']
        # lamp.register()
        # Update daily_growth_factor and par storage
        lamp.step()
        assert lamp.attributes['daily_growth_factor'] == 0
        assert lamp.storage['par'] == 0
        # Daily_growth_factor updates with model time, storage reset
        mock_model.time = datetime.datetime(2020, 1, 1, 4)
        lamp.storage['par'] = 1
        lamp.step()
        assert lamp.attributes['daily_growth_factor'] == 1.5 * 24/13
        assert lamp.storage['par'] == 0
        # Update variables when connected plants change
        mock_model.agents['test_plant_1'].active = 0
        lamp.step()
        assert lamp.active == 2
        assert lamp.lamp_configuration == {'test_plant_1': 0, 'test_plant_2': 2}
        assert lamp.attributes['photoperiod'] == 15
        assert lamp.attributes['par_rate'] == 1.2 * 24/15
        