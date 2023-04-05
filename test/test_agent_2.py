import copy
import datetime
import pytest
from ..agent_model.Agent import Agent

@pytest.fixture
def kwargs():
    return {
        'model': object(),
        'agent_id': 'test_agent',
        'amount': 10,
        'description': 'test_description',
        'agent_class': 'test_agent_class',
        'properties': {'test_property': 1},
        'capacity': {'test_currency': 2},
        'thresholds': {'test_currency': {
            'path': 'test_currency',
            'limit': '<',
            'value': 0.5,
        }},
        'flows': {
            'in': {
                'test_currency': {
                    'value': 1,
                    'connections': ['test_agent_2']
                }
            },
            'out': {
                'test_currency': {
                    'value': 1,
                    'connections': ['test_agent_2']
                }
            }
        },
        'cause_of_death': 'test_death',
        'active': 5,
        'storage': {'test_currency': 1},
        'attributes': {'test_attribute': 1},
    }

class TestAgentInit:
    def test_agent_init_empty(self):
        """Confirm that all attributes are set correctly when no kwargs are passed"""
        model = object()
        test_agent = Agent(model, 'test_agent')
        assert test_agent.agent_id == 'test_agent'
        assert test_agent.amount == 1
        assert test_agent.model == model
        assert test_agent.registered == False
        assert test_agent.cause_of_death == None
        assert str(test_agent.flows) == str({'in': {}, 'out': {}})
        empty_strings = {'description', 'agent_class'}
        empty_dicts = {'properties', 'capacity', 'thresholds', 'storage', 'attributes', 'records'}
        for k in empty_strings:
            assert getattr(test_agent, k) == ''
        for k in empty_dicts:
            assert str(getattr(test_agent, k)) == str({})

    def test_agent_init_full(self, kwargs):
        """Confirm that all kwargs are set correctly"""
        test_agent = Agent(**kwargs)
        for k, v in kwargs.items():
            assert str(getattr(test_agent, k)) == str(v)
        
    def test_agent_init_kwargs_immutable(self, kwargs):
        """Test that the kwargs passed to Agent.__init__() are immutable.

        We pass a set of kwargs to the Agent class and modify them outside of 
        the class. Then, we confirm that the Agent's internal attributes are 
        not modified by the external changes to the kwargs object. This test 
        ensures that the Agent's initialization process correctly creates
        a copy of the kwargs object to ensure immutability."""
        test_agent = Agent(**kwargs)
        # Confirm that class is not 
        def recursively_modify_kwargs(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    obj[k] = recursively_modify_kwargs(v)
            elif isinstance(obj, object):
                return object()
            elif isinstance(obj, list):
                return [recursively_modify_kwargs(i) for i in obj]
            elif isinstance(obj, (int, float)):
                return obj + 1
            else:
                return f'{obj}_modified'
        recursively_modify_kwargs(kwargs)
        for k, v in kwargs.items():
            assert str(getattr(test_agent, k)) != str(v)

@pytest.fixture
def mock_model():
    class MockModel:
        agents = {}
        time = datetime.datetime(2020, 1, 1)
    return MockModel()

@pytest.fixture
def basic_model(mock_model, kwargs):
    test_agent = Agent(**{**kwargs, 'model': mock_model})
    test_agent_2 = Agent(mock_model, 'test_agent_2', capacity={'test_currency': 2})
    mock_model.agents = {
        'test_agent': test_agent, 
        'test_agent_2': test_agent_2
    }
    return mock_model

class TestAgentRegister:
    def test_agent_register_empty(self):
        test_agent = Agent(object(), 'test_agent')
        assert test_agent.registered == False
        test_agent.register()
        assert test_agent.registered == True
        assert test_agent.attributes == {'age': 0}
        assert test_agent.records['active'] == []
        assert test_agent.records['cause_of_death'] == None
        assert test_agent.records['storage'] == {}
        assert test_agent.records['attributes'] == {'age': []}
        assert test_agent.records['flows'] == {'in': {}, 'out': {}}

    def test_agent_register_full_missing_connection(self, basic_model):
        test_agent = basic_model.agents['test_agent']
        basic_model.agents.pop('test_agent_2')
        with pytest.raises(ValueError):
            test_agent.register()
        assert test_agent.registered == False

    def test_agent_register_full_missing_currency(self, basic_model):
        test_agent = basic_model.agents['test_agent']
        basic_model.agents['test_agent_2'].capacity.pop('test_currency')
        with pytest.raises(ValueError):
            test_agent.register()
        assert test_agent.registered == False

    def test_agent_register_full(self, basic_model):
        test_agent = basic_model.agents['test_agent']
        test_agent.register()
        assert test_agent.registered == True
        assert test_agent.attributes == {'age': 0, 'test_attribute': 1}
        assert test_agent.records['active'] == []
        assert test_agent.records['storage'] == {'test_currency': []}
        assert test_agent.records['attributes'] == {'age': [], 'test_attribute': []}
        assert test_agent.records['flows'] == {
            'in': {'test_currency': {'test_agent_2': []}}, 
            'out': {'test_currency': {'test_agent_2': []}}
        }

    def test_agent_register_record_initial_state(self, basic_model):
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        assert test_agent.records['active'] == [5]
        assert test_agent.records['storage'] == {'test_currency': [1]}
        assert test_agent.records['attributes'] == {'age': [0], 'test_attribute': [1]}
        assert test_agent.records['flows'] == {
            'in': {'test_currency': {'test_agent_2': [0]}},
            'out': {'test_currency': {'test_agent_2': [0]}}
        }

@pytest.fixture
def flow():
    return {
        'value': 1,
        'criteria': {
            'buffer': 1,
        },
        'deprive': {
            'value': 2,
        },
        'growth': {
            'daily': {
                'type': 'clipped',
            },
            'lifetime': {
                'type': 'sigmoid'
            }
        },
        'connections': ['test_agent_2'] 
    }

class TestAgentRegisterFlow:
    def test_agent_register_flow(self, basic_model, flow):
        test_agent = basic_model.agents['test_agent']
        test_agent.properties['lifetime'] = {'value': 100}
        test_agent.flows['in']['test_currency'] = flow
        test_agent.register(record_initial_state=True)
        
        for attr in [
            'in_test_currency_criteria_buffer',
            'in_test_currency_deprive',
            'in_test_currency_daily_growth_factor',
            'in_test_currency_lifetime_growth_factor',
        ]:
            assert attr in test_agent.attributes
            assert len(test_agent.records['attributes'][attr]) == 1

@pytest.fixture
def mock_model_with_currencies(mock_model):
    mock_model.currencies = {
        'test_currency_1': {
            'currency_type': 'currency', 
            'class': 'test_currency_class'},
        'test_currency_2': {
            'currency_type': 'currency', 
            'class': 'test_currency_class'},
        'test_currency_class': {
            'currency_type': 'class', 
            'currencies': ['test_currency_1', 'test_currency_2']},
    }
    return mock_model

class TestAgentView:
    def test_agent_view_empty(self, mock_model_with_currencies):
        test_agent = Agent(mock_model_with_currencies, 'test_agent')
        test_agent.register()
        assert test_agent.view('test_currency_1') == {'test_currency_1': 0}
        assert test_agent.view('test_currency_2') == {'test_currency_2': 0}
        assert test_agent.view('test_currency_class') == {}

    def test_agent_view_full(self, mock_model_with_currencies):
        test_agent = Agent(
            mock_model_with_currencies, 
            'test_agent',
            storage={'test_currency_1': 1, 'test_currency_2': 2},
            capacity={'test_currency_1': 1, 'test_currency_2': 2},
        )
        assert test_agent.view('test_currency_1') == {'test_currency_1': 1}
        assert test_agent.view('test_currency_2') == {'test_currency_2': 2}
        assert test_agent.view('test_currency_class') == {'test_currency_1': 1, 'test_currency_2': 2}

    def test_agent_view_error(self, mock_model_with_currencies):
        test_agent = Agent(mock_model_with_currencies, 'test_agent')
        with pytest.raises(KeyError):
            test_agent.view('test_currency_3')

class TestAgentSerialize:
    def test_agent_serialize(self, basic_model, kwargs):
        test_agent = basic_model.agents['test_agent']
        test_agent.register()
        serialized = test_agent.serialize()
        serializable = {'agent_id', 'amount', 'description', 'agent_class', 
                        'properties', 'capacity', 'thresholds', 'flows',
                        'cause_of_death', 'active', 'storage', 'attributes'}
        assert set(serialized.keys()) == serializable
        for key in serializable:
            if key == 'attributes': 
                assert serialized[key] == {'age': 0, 'test_attribute': 1}
            else:
                assert serialized[key] == kwargs[key]

class TestAgentGetRecords:
    def test_agent_get_records_basic(self, basic_model):
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        records = test_agent.get_records()

        assert records['active'] == [5]
        assert records['cause_of_death'] == 'test_death'
        assert records['storage'] == {'test_currency': [1]}
        assert records['attributes'] == {'age': [0], 'test_attribute': [1]}
        assert records['flows'] == {
            'in': {'test_currency': {'test_agent_2': [0]}},
            'out': {'test_currency': {'test_agent_2': [0]}}
        }

    def test_agent_get_records_static(self, basic_model, kwargs):
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        records = test_agent.get_records(static=True)
        static_keys = {'agent_id', 'amount', 'agent_class', 'description', 
                       'properties', 'capacity', 'thresholds', 'flows'}
        assert set(records['static'].keys()) == static_keys
        for key in static_keys:
            assert records['static'][key] == kwargs[key]

    def test_agent_get_records_clear_cache(self, basic_model):
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        test_agent.get_records(clear_cache=True)
        def recursively_check_empty(dictionary):
            for key in dictionary:
                if isinstance(dictionary[key], dict):
                    recursively_check_empty(dictionary[key])
                elif isinstance(dictionary[key], list):
                    assert dictionary[key] == []
        recursively_check_empty(test_agent.records)

class TestAgentSave:
    def test_agent_save(self, basic_model, kwargs):
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        expected = copy.deepcopy(kwargs)
        del expected['model']
        expected['attributes'] = {'age': 0, 'test_attribute': 1}

        saved = test_agent.save()        
        assert saved == expected
        assert 'records' not in saved

    def test_agent_save_with_records(self, basic_model, kwargs):
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        expected = copy.deepcopy(kwargs)
        del expected['model']
        expected['attributes'] = {'age': 0, 'test_attribute': 1}
        expected['records'] = test_agent.get_records()

        saved = test_agent.save(records=True)        
        assert saved == expected