from copy import deepcopy
import pytest
from ..simoc_abm.agents import BaseAgent, ConcreteAgent

@pytest.fixture
def minimum_kwargs():
    return {
        'flows': {
            'in': {
                'co2': {'value': 0},
                'caoh2': {'value': 0},
            }, 
            'out': {
                'caco3': {'value': 0},
                'moisture': {'value': 0},
            }
        },
        'capacity': {
            'caoh2': 0,
            'caco3': 0,
            'moisture': 0
        },
    }

@pytest.fixture
def concrete_kwargs():
    return {
        "flows": {
            "in": {
                "co2": {
                    "value": 44.01,
                    "weighted": ["carbonation_rate"],
                    "connections": ["test_greenhouse"],
                },
                "caoh2": {
                    "value": 74.1,
                    "weighted": ["carbonation_rate"],
                    "requires": ["co2"],
                    "connections": ["concrete"],
                }
            },
            "out": {
                "caco3": {
                    "value": 100.09,
                    "weighted": ["carbonation_rate"],
                    "requires": ["co2", "caoh2"],
                    "connections": ["concrete"],
                },
                "moisture": {
                    "value": 18.02,
                    "weighted": ["carbonation_rate"],
                    "requires": ["co2", "caoh2"],
                    "connections": ["concrete"],
                }
            }
        },
        "capacity": {
            "caoh2": 1000,
            "caco3": 1000,
            "moisture": 1000
        },
    }

class MockModel:
    floating_point_precision = 6
    agents = {}
    currencies = {
        'o2': {'currency_type': 'currency'},
        'co2': {'currency_type': 'currency'},
        'caoh2': {'currency_type': 'currency'},
        'caco3': {'currency_type': 'currency'},
        'moisture': {'currency_type': 'currency'},
        'atmosphere': {'currency_type': 'category', 'currencies': ['o2', 'co2']}
    }

@pytest.fixture
def mock_model(concrete_kwargs):
    model = MockModel()
    model.agents = {
        'concrete': ConcreteAgent(model, 'concrete', **concrete_kwargs),
        'test_greenhouse': BaseAgent(model, 'test_greenhouse',
                                 storage={'o2': 999.65, 'co2': 0.35},
                                 capacity={'o2': 1000, 'co2': 1000})
    }
    return model

class TestConcreteAgent:
    def test_agent_concrete_calc_max(self):
        lowest, highest = ConcreteAgent.ppm_range
        vals = [ConcreteAgent.calc_max_carbonation(ppm)
                for ppm in range(lowest, highest+50, 50)]
        for i, carbonation in enumerate(vals[1:]):
            assert carbonation > vals[i]

    def test_agent_concrete_init(self, minimum_kwargs, concrete_kwargs):
        """Initialize attributes properly"""

        # Missing kwarg
        model = object()
        incomplete_kwargs = deepcopy(minimum_kwargs)
        del incomplete_kwargs['flows']['in']['co2']
        with pytest.raises(ValueError):
            concrete = ConcreteAgent(model, 'concrete', **incomplete_kwargs)

        # Minimum kwargs
        model = object()
        concrete = ConcreteAgent(model, 'concrete', **minimum_kwargs)
        for k, v in ConcreteAgent.default_attributes.items():
            assert concrete.attributes[k] == v

        # All kwargs
        model = object()
        concrete = ConcreteAgent(model, 'concrete', **concrete_kwargs)
        assert True

        # Initial carbonation
        half_carbonated = ConcreteAgent.calc_max_carbonation(3000) / 2
        carbo_kwargs = {**concrete_kwargs, 
                        'attributes': {'carbonation': half_carbonated}}
        concrete = ConcreteAgent(model, 'concrete', **carbo_kwargs)
        assert concrete.attributes['carbonation'] == pytest.approx(0.078267, 6)
        assert concrete.attributes['carbonation_rate'] == 0
        assert concrete.storage == {
            'caoh2': pytest.approx(5.799572, 6),
            'caco3': pytest.approx(7.783372, 6),
            'moisture': pytest.approx(1.410358, 6),
        }

    def test_agent_concrete_step(self, mock_model):
        concrete_agent = mock_model.agents['concrete']
        greenhouse_agent = mock_model.agents['test_greenhouse']
        concrete_agent.register()
        greenhouse_agent.register()
        assert concrete_agent.attributes['carbonation'] == 0
        concrete_agent.step()
        carbonation_rate = concrete_agent.attributes['carbonation_rate']
        assert carbonation_rate != 0
        assert concrete_agent.attributes['carbonation'] == carbonation_rate
        