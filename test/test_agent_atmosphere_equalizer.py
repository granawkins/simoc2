import pytest
from simoc_abm.agents import AtmosphereEqualizerAgent
from simoc_abm.util import get_default_currency_data

class MockAgent:
    def __init__(self, amount, volume, o2, co2):
        self.amount = amount
        self.properties = {'volume': {'value': volume}}
        self.storage = {'o2': o2, 'co2': co2}
        self.capacity = {'o2': 100, 'co2': 100}

    def view(self, *args, **kwargs):
        return {k: v for k, v in self.storage.items()}
    
    def increment(self, currency, amount):
        self.storage[currency] += amount

class MockModel:
    agents = {}
    currencies = None

@pytest.fixture
def mock_model(default_currency_dict):
    model = MockModel()
    model.currencies = default_currency_dict
    model.agents = {
        'test_agent_1': MockAgent(1, 20, 3, 3),
        'test_agent_2': MockAgent(1, 10, 3, 3),
    }
    return model

class TestAgentAtmosphereEqualizer:
    def test_agent_atmosphere_equalizer_step(self, mock_model):
        conns = ['test_agent_1', 'test_agent_2']
        kwargs = {'agent_id': 'atmosphere_equalizer',
                  'flows': {'in': {'atmosphere': {'connections': conns}},
                            'out': {'atmosphere': {'connections': conns}}}}
        agent = AtmosphereEqualizerAgent(mock_model, **kwargs)
        agent.step()
        test_agent_1 = mock_model.agents['test_agent_1']
        test_agent_2 = mock_model.agents['test_agent_2']
        assert test_agent_1.storage['o2'] == 4
        assert test_agent_1.storage['co2'] == 4
        assert test_agent_2.storage['o2'] == 2
        assert test_agent_2.storage['co2'] == 2
        