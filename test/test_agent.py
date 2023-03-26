import copy
import pytest
from pytest import approx

from ..agent_model.Agent import Agent
from ..agent_model.Model import Model
from ..agent_model.util import load_data_file

class MockModel:
    def __init__(self):
        self.floating_point_accuracy = 6
        self.step_num = 0
        self.currency_dict = Model.build_currency_dict()
        self.agents = {}

    def register_agents(self):
        for agent in self.agents.values():
            agent.register()

    def step(self, n_steps=1):
        for _ in range(n_steps):
            for agent in self.agents.values():
                agent.step()

@pytest.fixture
def default_kwargs():
    return dict(
        amount=1,
        description='',
        agent_class='',
        properties={},
        capacity={},
        thresholds={},
        flows={'in': {}, 'out': {}},
        active=1,
        storage={},
        attributes={},
    )

@pytest.fixture
def agent_kwargs():
    return dict(
        amount=10,
        description='Behaves like a human',
        agent_class='test',
        properties={'activity_level': {'value': 0.5}, 'mass_ratio': {'value': 0.5, 'unit': 'kg'}},
        capacity={'ch4': .01},
        thresholds={'o2': {'path': 'in_o2_ratio', 'limit': '<', 'value': .14, 'connections': 'all'},
                    'ch4': {'path': 'ch4', 'limit': '>', 'value': .009}},
        flows={'in': {
            'o2': {
                'value': .03,
                'flow_rate': {'unit': 'kg', 'time': 'hour'},
                'deprive': {'value': 2, 'unit': 'hour'},
                'weighted': ['activity_level', 'mass_ratio'],
                'connections': ['test_habitat', 'test_o2_mask'],
            },
            'ch4': {
                'value': .001,
                'flow_rate': {'unit': 'kg', 'time': 'hour'},
                'criteria': {'path': 'in_ch4_ratio', 'limit': '>', 'value': .001, 'buffer': 2},
                'connections': ['test_habitat'],
            }
        }, 'out': {
            'co2': {
                'value': .03,
                'flow_rate': {'unit': 'kg', 'time': 'hour'},
                'requires': ['o2'],
                'connections': ['test_habitat'],
            },
            'ch4': {
                'value': .001,
                'flow_rate': {'unit': 'kg', 'time': 'hour'},
                'connections': ['test_agent'],
                'requires': ['ch4'],
            }
        }},
        storage={'ch4': 0.0},
        attributes={},
    )

@pytest.fixture
def habitat_kwargs():
    return dict(
        capacity={'n2': 100, 'o2': 100, 'co2': 100, 'ch4': 100},
        storage={'n2': 80, 'o2': 19.5, 'co2': 0.38, 'ch4': 0.12},
    )

@pytest.fixture
def o2_mask_kwargs():
    return dict(
        storage={'o2': 5},
        capacity={'o2': 5},
    )

def test_agent_register(default_kwargs, agent_kwargs, habitat_kwargs, o2_mask_kwargs):

    # __init__
    # Default (empty) agent
    model = MockModel()
    agent = Agent('test_agent', model)
    for k, v in default_kwargs.items():
        assert getattr(agent, k) == v

    # All kwargs
    model = MockModel()
    model.agents = {
        'test_agent': Agent('test_agent', model, **agent_kwargs),
        'test_habitat': Agent('test_habitat', model, **habitat_kwargs),
        'test_o2_mask': Agent('test_o2_mask', model, **o2_mask_kwargs),
    }
    kwargs = (agent_kwargs, habitat_kwargs, o2_mask_kwargs)
    for agent, _kwargs in zip(model.agents.values(), kwargs):
        for k, v in _kwargs.items():
            assert str(getattr(agent, k)) == str(v)

    # register, register_flow
    for agent in model.agents.values():
        agent.register()
    agent = model.agents['test_agent']
    assert agent.registered
    assert str(agent.records) == str({
        'step_num': [0],
        'active': [10],
        'cause_of_death': None,
        'storage': {'ch4': [0.0]},
        'attributes': {
            'in_o2_deprive': [2],
            'in_ch4_criteria_buffer': [2]},
        'flows': {
            'in': {'o2': {'test_habitat': [0], 'test_o2_mask': [0]},
                   'ch4': {'test_habitat': [0]}},
            'out': {'co2': {'test_habitat': [0]},
                    'ch4': {'test_agent': [0]}}}})

@pytest.fixture
def dummy_model(agent_kwargs, habitat_kwargs, o2_mask_kwargs):
    model = MockModel()
    model.agents = {
        'test_agent': Agent('test_agent', model, **agent_kwargs),
        'test_habitat': Agent('test_habitat', model, **habitat_kwargs),
        'test_o2_mask': Agent('test_o2_mask', model, **o2_mask_kwargs),
    }
    model.register_agents()
    return model

def test_agent_storage(dummy_model):

    # view
    agent = dummy_model.agents['test_agent']
    habitat = dummy_model.agents['test_habitat']
    assert str(agent.view('co2')) == str({'co2': 0})
    assert str(habitat.view('co2')) == str({'co2': 0.38})
    assert str(habitat.view('atmosphere')) == str({'o2': 19.5, 'co2': 0.38, 'n2': 80, 'ch4': 0.12})
    assert str(habitat.view('food')) == str({})

    # increment
    assert str(habitat.increment('co2', -.001)) == str({'co2': -0.001})
    assert str(habitat.view('co2')) == str({'co2': 0.379})
    # if not enough in storage, do as much as is available
    assert str(habitat.increment('co2', -10)) == str({'co2': -0.379})
    assert str(habitat.view('co2')) == str({'co2': 0.0})
    # if over-capacity, take as much as possible
    assert str(habitat.increment('n2', 30)) == str({'n2': 20})
    assert str(habitat.view('n2')) == str({'n2': 100})
    # for groups, return proportional amount
    group_increment = habitat.increment('atmosphere', -1)
    assert sum(group_increment.values()) == approx(-1)

def test_agent_get_step_value(dummy_model):
    agent = dummy_model.agents['test_agent']

    # weighted
    step_value = agent.get_step_value(
        dT=1,
        direction='in',
        currency='o2',
        flow=agent.flows['in']['o2'],
        influx={})
    assert step_value == (
        0.03   # value
        * 0.5  # properties.activity_level
        * 0.5  # properties.mass_ratio
    )

    # requires
    kwargs = dict(
        dT=1,
        direction='out',
        currency='co2',
        flow=agent.flows['out']['co2'],
        influx={'o2': 0.25})
    step_value = agent.get_step_value(**kwargs)
    assert step_value == (
        0.03   # value
        * 0.25  # influx[o2]
    )
    kwargs['influx']['o2'] = 0.0
    step_value = agent.get_step_value(**kwargs)
    assert step_value == 0.0  # Return 0 if requirements not met

    # criteria
    kwargs = dict(dT=1,
                  direction='in',
                  currency='ch4',
                  flow=agent.flows['in']['ch4'],
                  influx={})
    assert agent.attributes['in_ch4_criteria_buffer'] == 2
    step_value = agent.get_step_value(**kwargs)
    assert step_value == (0.0)  # Return 0 and increment buffer
    assert agent.attributes['in_ch4_criteria_buffer'] == 1
    step_value2 = agent.get_step_value(**kwargs)
    assert step_value2 == (0.0)
    step_value3 = agent.get_step_value(**kwargs)
    assert step_value3 == (0.001)  # When buffer  is empty, return value
    dummy_model.agents['test_habitat'].storage['ch4'] = 0
    step_value4 = agent.get_step_value(**kwargs)
    assert step_value4 == (0.0)  # If criteria false, reset buffer and return 0
    assert agent.attributes['in_ch4_criteria_buffer'] == 2

def test_agent_step_requires(dummy_model):
    agent = Agent('test_agent', dummy_model, flows={
            'in': {'n2': {'value': .001, 'connections': ['test_habitat'], 'requires': ['o2']},
                   'o2': {'value': .001, 'connections': ['test_habitat']},
                   'ch4': {'value': .001, 'connections': ['test_habitat'], 'requires': ['o2']}},
            'out': {}})
    dummy_model.agents['test_agent'] = agent
    dummy_model.step(2)
    dummy_model.agents['test_habitat'].storage['o2'] = 0.0005
    dummy_model.step(2)
    assert agent.records['flows']['in']['n2']['test_habitat'] == [0, 0, 0, 0, 0]  # before  step, no flow
    assert agent.records['flows']['in']['ch4']['test_habitat'] == [0, -0.001, -0.001, -0.0005, 0]  # dependent and weighted

def test_agent_step_threshold(dummy_model):
    agent = dummy_model.agents['test_agent']
    assert agent.attributes['in_ch4_criteria_buffer'] == 2
    assert agent.storage['ch4'] == 0
    dummy_model.step()
    assert agent.attributes['in_ch4_criteria_buffer'] == 1
    dummy_model.step()
    assert agent.attributes['in_ch4_criteria_buffer'] == 0
    assert agent.storage['ch4'] == 0
    dummy_model.step()
    assert agent.storage['ch4'] == 0.01
    dummy_model.step()
    assert agent.active == 0
    assert agent.cause_of_death == 'test_agent passed ch4 threshold'

def test_agent_step_connections(dummy_model):
    # Force the habitat to run out of O2 and use o2 mask. have to remove threshold to do this.
    habitat = dummy_model.agents['test_habitat']
    habitat.storage['ch4'] = 0  # Don't kill agent
    habitat.storage['o2'] = 0   # Force agent to use secondary ocnnection for o2
    dummy_model.step(1)
    o2_record = dummy_model.agents['test_agent'].records['flows']['in']['o2']
    assert o2_record['test_o2_mask'][1] == -0.075
    assert dummy_model.agents['test_agent'].active == 10

def test_agent_step_deprive(dummy_model):
    dummy_model.agents['test_habitat'].storage['o2'] = 0
    dummy_model.agents['test_o2_mask'].storage['o2'] = 0
    agent = dummy_model.agents['test_agent']
    del agent.thresholds['o2']
    assert agent.attributes['in_o2_deprive'] == 2
    dummy_model.step()
    assert agent.attributes['in_o2_deprive'] == 1
    dummy_model.step()
    assert agent.attributes['in_o2_deprive'] == 0
    assert agent.cause_of_death is None
    dummy_model.step()
    assert agent.cause_of_death == 'test_agent deprived of o2'

