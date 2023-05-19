import json
import time
import datetime
from unittest.mock import Mock

import pytest
import numpy as np

from ..simoc_abm.agent_model import (DEFAULT_START_TIME,
                                 DEFAULT_LOCATION,
                                 DEFAULT_PRIORITIES,
                                 AgentModel, 
                                 Scheduler)
from ..simoc_abm.agents import BaseAgent
from ..simoc_abm.util import get_default_agent_data

class MockModel:
    def __init__(self, seed=1000):
        self.rng = np.random.RandomState(seed)
        self.priorities = ['test_agent_class_2', 'test_agent_class_1']
        self.agents = {}

class MockAgent():
    def __init__(self, agent_id, agent_class):
        self.agent_id = agent_id
        self.agent_class = agent_class
        self.dT = None
        self.stepped_at = None
    def step(self, dT):
        time.sleep(.001)
        self.dT = dT
        self.stepped_at = datetime.datetime.now()
        

@pytest.fixture
def mock_model():
    model = MockModel()
    model.agents = {
        'a': MockAgent('a', 'test_agent_class_1'),
        'b': MockAgent('b', 'test_agent_class_1'),
        'c': MockAgent('c', 'test_agent_class_2'),
        'd': MockAgent('d', 'test_agent_class_3'),
        'e': MockAgent('e', 'test_agent_class_4'),
    }
    return model

class TestScheduler:
    def test_scheduler_init(self, mock_model):
        test_scheduler = Scheduler(mock_model)
        assert test_scheduler.model == mock_model
        assert test_scheduler.priorities == [*mock_model.priorities, 'other']
        assert test_scheduler.class_agents == {
            'test_agent_class_1': ['a', 'b'],
            'test_agent_class_2': ['c'],
            'other': ['d', 'e']
        }

    def test_scheduler_step(self, mock_model):
        test_scheduler = Scheduler(mock_model)
        
        # dT is passed appropriately
        test_scheduler.step(0.42)
        for agent in mock_model.agents.values():
            assert agent.dT == 0.42
            
        # Expected class order: c, [a and b], [d and e]
        stepped_at = {k: v.stepped_at for k, v in mock_model.agents.items()}
        assert stepped_at['c'] == min(stepped_at.values())
        assert any(stepped_at[i] == max(stepped_at.values()) for i in ['d', 'e'])

        # Within classes, order is random
        samples = []
        for _ in range(5):
            test_scheduler.step(0.42)
            samples.append((
                mock_model.agents['a'].stepped_at.timestamp(), 
                mock_model.agents['b'].stepped_at.timestamp()))
        assert any(a < b for a, b in samples)
        assert any(b < a for a, b in samples)        
        
@pytest.fixture
def model_kwargs():
    return {
        'termination': [{
            "condition": "time",
            "value": 10,
            "unit": "day"
        }],
        'location': 'test_location',
        'priorities': ['test_agent_class_1', 'test_agent_class_2'],
        'start_time': '2020-01-01T00:00:00',
        'elapsed_time': 0,
        'step_num': 0,
        'seed': 1000,
        'is_terminated': False,
        'termination_reason': 'test_termination',
    }

class TestModel:
    def test_model_init_basic(self):
        test_model = AgentModel()
        assert test_model.termination == []
        assert test_model.location == DEFAULT_LOCATION
        assert test_model.priorities == DEFAULT_PRIORITIES
        assert test_model.start_time == datetime.datetime.fromisoformat(DEFAULT_START_TIME)
        assert test_model.elapsed_time == datetime.timedelta()
        assert test_model.step_num == 0
        assert 0 <= test_model.seed <= 2**32 - 1
        assert test_model.is_terminated is None
        assert test_model.termination_reason == ''
        assert test_model.agents == {}
        assert test_model.currencies == {}
        assert test_model.rng == None
        assert test_model.scheduler == None
        assert test_model.records == {'time': [], 'step_num': []}

    def test_model_init_complex(self, model_kwargs):
        model_kwargs['elapsed_time'] = 42
        model_kwargs['step_num'] = 100
        test_model = AgentModel(**model_kwargs)
        assert test_model.termination == model_kwargs['termination']
        assert test_model.location == model_kwargs['location']
        assert test_model.priorities == model_kwargs['priorities']
        assert test_model.start_time == datetime.datetime.fromisoformat(model_kwargs['start_time'])
        assert test_model.elapsed_time == datetime.timedelta(seconds=42)
        assert test_model.step_num == 100
        assert test_model.seed == model_kwargs['seed']
        assert test_model.is_terminated is False
        assert test_model.termination_reason == 'test_termination'

    def test_model_add_agent(self, model_kwargs):
        model = AgentModel(**model_kwargs)
        test_agent = object()
        model.add_agent('test_agent_id', test_agent)
        with pytest.raises(ValueError):
            model.add_agent('test_agent_id', object())
        assert model.agents == {'test_agent_id': test_agent}

    def test_model_add_currency(self, model_kwargs):
        model = AgentModel(**model_kwargs)
        model.add_currency('test_currency_id')
        with pytest.raises(ValueError):
            model.add_currency('test_currency_id')
        assert model.currencies == {'test_currency_id': {'currency_type': 'currency'}}

    def test_model_register(self, model_kwargs):
        # With record initial state
        model = AgentModel(**model_kwargs)
        test_agent = Mock()
        model.add_agent('test_agent', test_agent)
        model.register(record_initial_state=True)
        assert isinstance(model.rng, np.random.RandomState)
        assert isinstance(model.scheduler, Scheduler)
        assert model.records == {'time': [model_kwargs['start_time']], 'step_num': [0]}
        test_agent.register.assert_called_once_with(True)
        assert model.registered

        # Without record initial state
        model = AgentModel(**model_kwargs)
        test_agent = Mock()
        model.add_agent('test_agent', test_agent)
        model.register(record_initial_state=False)
        assert model.records == {'time': [], 'step_num': []}
        test_agent.register.assert_called_once_with(False)
    
    def test_model_from_config(self, model_kwargs):
        agents = {'o2_storage': {'description': 'test_description'},
                  'test_agent': {'capacity': {'test_currency': 0}, 'storage': {'test_currency': 0}, 
                                 'flows': {'in': {'o2': {'value': 0, 'connections': ['o2_storage']}}}}}
        currencies = {'test_currency': {'description': 'test_description'}}
        model = AgentModel.from_config(agents, currencies, **model_kwargs)
        assert list(model.agents.keys()) == ['o2_storage', 'test_agent']
        assert 'test_currency' in model.currencies
        assert model.registered
        assert len(model.records['time']) == 1
        
        # Check that agent is merged with default agent data
        default_o2_storage_agent = get_default_agent_data('o2_storage')
        o2_storage_agent = model.agents['o2_storage']
        assert isinstance(o2_storage_agent, BaseAgent)
        assert o2_storage_agent.description == 'test_description'
        assert o2_storage_agent.capacity == default_o2_storage_agent['capacity']

        # Check that loaded models (step_num != 0) don't record initial state
        model_kwargs['step_num'] = 1
        model = AgentModel.from_config(agents, currencies, **model_kwargs)
        assert model.registered
        assert len(model.records['time']) == 0

    def test_model_time(self, model_kwargs):
        model = AgentModel(**model_kwargs)
        assert model.time == model.start_time
        model.elapsed_time = datetime.timedelta(seconds=42)
        assert model.time == model.start_time + model.elapsed_time

    def test_model_step(self, model_kwargs):
        model = AgentModel(**model_kwargs)
        model.step()
        assert model.registered
        assert model.step_num == 1
        assert model.elapsed_time == datetime.timedelta(hours=1)
        assert model.records['time'] == ['2020-01-01T01:00:00']
        assert model.records['step_num'] == [1]

    def test_model_terminate(self, model_kwargs):
        for (unit, value) in (('day', 2), ('hour', 30)):
            model_kwargs['termination'][0]['unit'] = unit
            model_kwargs['termination'][0]['value'] = value
            model = AgentModel(**model_kwargs)
            expected_steps = value * 24 if unit == 'day' else value
            for _ in range(expected_steps - 1):
                model.step()
            assert not model.is_terminated
            model.step()
            assert model.is_terminated
            assert model.termination_reason == 'time'

    def test_model_get_records(self, model_kwargs):
        model = AgentModel.from_config({'o2_storage': {}}, {}, **model_kwargs)
        model.step()
        # Basic records
        records = model.get_records()        
        assert list(records.keys()) == ['time', 'step_num', 'agents']
        assert len(records['time']) == 2
        assert len(records['agents']['o2_storage']['active']) == 2
        # Include static records
        records = model.get_records(static=True)
        assert 'static' in records
        assert 'static' in records['agents']['o2_storage']
        assert list(records['static'].keys()) == ['currencies', 'termination', 
                                                  'location', 'priorities', 
                                                  'start_time', 'seed']
        # Check that everything in records is serializable
        json.dumps(records)
        assert True
        # Clear cache
        records = model.get_records(clear_cache=True)
        assert len(model.records['time']) == 0
        assert len(model.agents['o2_storage'].records['active']) == 0

    def test_model_save(self, model_kwargs):
        model = AgentModel.from_config({'o2_storage': {}}, {}, **model_kwargs)
        model.step()
        saved = model.save()
        assert 'records' not in saved
        json.dumps(saved)
        assert True
        # Re-load
        # TODO: Test save/load in depth
        model = AgentModel.from_config(**saved)
        model.step()
        assert True
        # Include records
        saved = model.save(records=True)
        assert 'records' in saved