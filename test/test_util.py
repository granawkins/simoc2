import pytest

from ..agent_model.Agent import Agent
from ..agent_model.Model import Model
from ..agent_model.util import (load_data_file, merge_json, get_default_agent_data,
                                evaluate_reference, evaluate_growth)

def test_load_data_files():
    agent_desc = load_data_file('agent_desc.json')
    assert 'wheat' in agent_desc, 'Failed to load agent_desc'

def test_get_default_agent_data():
    wheat_data = get_default_agent_data('wheat')
    assert all([k in wheat_data for k in ['amount', 'storage', 'properties', 'storage', 'flows']])

def test_merge_config():
    c1 = {'a': 'red', 'b': 2, 'c': {'d': 3, 'e': 4}, 'f': [1, 2, 3]}
    c2 = {'a': 'blue', 'c': {'d': 6}, 'f': [3, 4, 5]}
    c3 = merge_json(c1, c2)
    assert c3 == {'a': 'blue', 'b': 2, 'c': {'d': 6, 'e': 4}, 'f': [1, 2, 3, 4, 5]}


class MockAgent:
    def __init__(self, model, other_agent=None, storage=None):
        self.model = model
        self.attributes = {'grown': False}
        self.storage = storage
        self.flows = {'in': {'co2': {'connections': [other_agent]}}}
    def view(self, *args, **kwargs):
        return self.storage
class MockModel:
    def __init__(self):
        self.step_num = 0
        self.floating_point_accuracy = 6
        self.currency_dict = Model.build_currency_dict()
        self.agents = {
            'a': MockAgent(self, 'b'),
            'b': MockAgent(self, 'a', storage={'co2': 0.02, 'o2': .2, 'n2': 0.78})}

def test_evaluate_reference():
    model = MockModel()
    agent = model.agents['a']
    reference = {'path': 'grown', 'limit': '=', 'value': False}
    assert evaluate_reference(agent, reference)
    reference = {'path': 'in_co2_ratio', 'limit': '>', 'value': 0.01}
    assert evaluate_reference(agent, reference)
    reference = {'path': 'in_co2_ratio', 'limit': '>', 'value': 0.03}
    assert not evaluate_reference(agent, reference)

def test_evaluate_growth():
    model = MockModel()
    agent = model.agents['a']
    assert evaluate_growth(agent, 'daily', {'type': 'clipped'}) > 0


