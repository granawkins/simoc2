import copy
import datetime
import pytest
from ..agent_model.Agent import Agent
from ..agent_model.util import get_default_currency_data

@pytest.fixture
def kwargs():
    return {
        'model': object(),
        'agent_id': 'test_agent',
        'amount': 10,
        'description': 'test_description',
        'agent_class': 'test_agent_class',
        'properties': {'test_property': {'value': 1}},
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
        floating_point_accuracy = 6
        agents = {}
        time = datetime.datetime(2020, 1, 1)
        currencies = {'test_currency': {'currency_type': 'currency'}}
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
        """Confirm that all attributes set correctly when no kwargs passed"""
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
        """Confirm that an error is raised if connection agent not in model"""
        test_agent = basic_model.agents['test_agent']
        basic_model.agents.pop('test_agent_2')
        with pytest.raises(ValueError):
            test_agent.register()
        assert test_agent.registered == False

    def test_agent_register_full_missing_currency(self, basic_model):
        """Confirm that an error is raised if capacity missing for storage"""
        test_agent = basic_model.agents['test_agent']
        basic_model.agents['test_agent_2'].capacity.pop('test_currency')
        with pytest.raises(ValueError):
            test_agent.register()
        assert test_agent.registered == False

    def test_agent_register_full_missing_capacity(self, basic_model):
        """Confirm that an error is raised if initial storage greater than capacity"""
        test_agent = basic_model.agents['test_agent']
        test_agent.capacity['test_currency'] = 0
        with pytest.raises(ValueError):
            test_agent.register()
        assert test_agent.registered == False

    def test_agent_register_full(self, basic_model):
        """Confirm that all fields set correctly when kwargs passed"""
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
        """Confirm that initial state is recorded when requested"""
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
        """Confirm that all flow attributes are set correctly"""
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
        """Confirm that view returns empty dict if no storage or capacity"""
        test_agent = Agent(mock_model_with_currencies, 'test_agent')
        test_agent.register()
        assert test_agent.view('test_currency_1') == {'test_currency_1': 0}
        assert test_agent.view('test_currency_2') == {'test_currency_2': 0}
        assert test_agent.view('test_currency_class') == {}

    def test_agent_view_full(self, mock_model_with_currencies):
        """Confirm view returns correct values for currency or currency class"""
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
        """Confirm that error is raised if view currency not in model"""
        test_agent = Agent(mock_model_with_currencies, 'test_agent')
        with pytest.raises(KeyError):
            test_agent.view('test_currency_3')

class TestAgentSerialize:
    def test_agent_serialize(self, basic_model, kwargs):
        """Confirm that all fields are serialized correctly"""
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
        """Confirm that all fields are recorded correctly"""
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
        """Confirm that static fields are recorded correctly"""
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        records = test_agent.get_records(static=True)
        static_keys = {'agent_id', 'amount', 'agent_class', 'description', 
                       'properties', 'capacity', 'thresholds', 'flows'}
        assert set(records['static'].keys()) == static_keys
        for key in static_keys:
            assert records['static'][key] == kwargs[key]

    def test_agent_get_records_clear_cache(self, basic_model):
        """Confirm that get_records clears cache when requested"""
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
        """Test that save returns a dictionary matching initialization"""
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        expected = copy.deepcopy(kwargs)
        del expected['model']
        expected['attributes'] = {'age': 0, 'test_attribute': 1}

        saved = test_agent.save()        
        assert saved == expected
        assert 'records' not in saved

    def test_agent_save_with_records(self, basic_model, kwargs):
        """Test that records are included in save if requested"""
        test_agent = basic_model.agents['test_agent']
        test_agent.register(record_initial_state=True)
        expected = copy.deepcopy(kwargs)
        del expected['model']
        expected['attributes'] = {'age': 0, 'test_attribute': 1}
        expected['records'] = test_agent.get_records()

        saved = test_agent.save(records=True)        
        assert saved == expected

class TestAgentIncrement:
    def test_agent_increment_positive(self, mock_model_with_currencies):
        """Test that increment correctly increments currencies"""
        test_agent = Agent(
            mock_model_with_currencies, 
            'test_agent',
            storage={'test_currency_1': 1},
            capacity={'test_currency_1': 2},
        )
        # Test incrementing a single currency
        receipt = test_agent.increment('test_currency_1', 1)
        assert receipt == {'test_currency_1': 1}
        assert test_agent.storage['test_currency_1'] == 2
        # Test incrementing a single currency beyond capacity
        receipt = test_agent.increment('test_currency_1', 1)
        assert receipt == {'test_currency_1': 0}
        assert test_agent.storage['test_currency_1'] == 2
        # Test incrementing a currency without capacity
        with pytest.raises(ValueError):
            test_agent.increment('test_currency_2', 1)
        # Test incrementing a currency class
        with pytest.raises(ValueError):
            test_agent.increment('test_currency_class', 1)

    def test_agent_increment_negative(self, mock_model_with_currencies):
        """Test that increment correctly decrements currencies"""
        test_agent = Agent(
            mock_model_with_currencies, 
            'test_agent',
            storage={'test_currency_1': 2, 'test_currency_2': 1},
            capacity={'test_currency_1': 2, 'test_currency_2': 2},
        )
        # Test decrementing a single currency
        receipt = test_agent.increment('test_currency_1', -1)
        assert receipt == {'test_currency_1': -1}
        assert test_agent.storage['test_currency_1'] == 1
        # Test decrementing a currency class
        receipt = test_agent.increment('test_currency_class', -1)
        assert receipt == {'test_currency_1': -0.5, 'test_currency_2': -0.5}
        assert test_agent.storage['test_currency_1'] == 0.5
        assert test_agent.storage['test_currency_2'] == 0.5
        # Test decrementing a currency class beyond stored
        receipt = test_agent.increment('test_currency_class', -2)
        assert receipt == {'test_currency_1': -0.5, 'test_currency_2': -0.5}
        assert test_agent.storage['test_currency_1'] == 0
        assert test_agent.storage['test_currency_2'] == 0
        # TODO: Test with amount>1, confirm capacity scales with amount

@pytest.fixture
def get_flow_value_kwargs(kwargs):
    return {
        'dT': 1,
        'direction': 'in',
        'currency': 'test_currency',
        'flow': kwargs['flows']['in']['test_currency'],
        'influx': {},
    }

class TestAgentGetFlowValue:
    def test_agent_get_flow_value_basic(self, basic_model, get_flow_value_kwargs):
        """Test that get_flow_value returns the correct value"""
        test_agent = basic_model.agents['test_agent']
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 1
        get_flow_value_kwargs['dT'] = 0.33
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 0.33
        
    def test_agent_get_flow_value_requires(self, basic_model, get_flow_value_kwargs):
        """Test that get_flow_value handles requires correctly"""
        # Single Currency
        test_agent = basic_model.agents['test_agent']
        get_flow_value_kwargs['flow']['requires'] = ['test_currency_2']
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 0    
        get_flow_value_kwargs['influx'] = {'test_currency_2': 0.5}
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 0.5
        get_flow_value_kwargs['influx'] = {'test_currency_2': 1}
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 1

        # Multiple Currencies
        get_flow_value_kwargs['flow']['requires'] = ['test_currency_2', 'test_currency_3']
        get_flow_value_kwargs['influx'] = {'test_currency_2': 0.5, 'test_currency_3': 0.5}
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 0.25

    def test_agent_get_flow_value_criteria_basic(self, basic_model, get_flow_value_kwargs):
        # TODO: Move some of this to the test for evaluate_criteria.
        test_agent = basic_model.agents['test_agent']
        # Equality | Attribute
        test_agent.flows['in']['test_currency']['criteria'] = {
            'path': 'test_attribute',
            'limit': '=',
            'value': 1,
        }
        get_flow_value_kwargs['flow'] = test_agent.flows['in']['test_currency']
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 1
        test_agent.flows['in']['test_currency']['criteria']['value'] = 2
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 0
        
    def test_agent_get_flow_value_growth(self, basic_model, get_flow_value_kwargs):
        test_agent = basic_model.agents['test_agent']
        get_flow_value_kwargs['flow']['growth'] = {'lifetime': {'type': 'sigmoid'}}
        test_agent.properties['lifetime'] = {'value': 10}
        test_agent.attributes['age'] = 0
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert 0 < flow_value < 0.0001
        test_agent.attributes['age'] = 5
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 0.5
        test_agent.attributes['age'] = 10
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert 0.9999 < flow_value < 1
        
    def test_agent_get_flow_value_weighted(self, basic_model, get_flow_value_kwargs):
        test_agent = basic_model.agents['test_agent']
        test_agent.properties['test_property']['value'] = 0.5
        test_agent.attributes['test_attribute'] = 0.5
        test_agent.storage['test_currency'] = 10  # divided by active=5
        get_flow_value_kwargs['flow']['weighted'] = [
            'test_property', 'test_attribute', 'test_currency']
        flow_value = test_agent.get_flow_value(**get_flow_value_kwargs)
        assert flow_value == 1 * 0.5 * 0.5 * 10 / 5
        get_flow_value_kwargs['flow']['weighted'] = ['missing_weight']
        with pytest.raises(ValueError):
            test_agent.get_flow_value(**get_flow_value_kwargs)

class TestAgentProcessFlow:
    def test_process_flow_influx(self, mock_model_with_currencies):
        test_agent = Agent(
            mock_model_with_currencies,
            'test_agent',
            flows={
                'in': {
                    'test_currency_1': {
                        'value': 1,
                        'connections': ['test_agent_2'],
                    },
                },
                'out': {
                    'test_currency_2': {
                        'value': 1,
                        'connections': ['test_agent_2'],
                    },
                }
            }
        )
        test_agent_2 = Agent(
            mock_model_with_currencies,
            'test_agent_2',
            capacity={'test_currency_1': 2, 'test_currency_2': 2},
        )
        mock_model_with_currencies.agents = {
            'test_agent': test_agent,
            'test_agent_2': test_agent_2,
        }
        test_agent.register()
        test_agent_2.register()
        influx = {}
        for direction in ('in', 'out'):
            currency = next(iter(test_agent.flows[direction]))
            kwargs = dict(dT=1, direction=direction, currency=currency, 
                          flow=test_agent.flows[direction][currency], influx=influx, 
                          target=1, actual=1)
            test_agent.process_flow(**kwargs)
        assert influx == {'test_currency_1': 1}

    def test_process_flow_deprive(self, mock_model_with_currencies):
        test_agent = Agent(
            mock_model_with_currencies,
            'test_agent',
            amount=5,
            flows={
                'in': {
                    'test_currency_1': {
                        'value': 1,
                        'connections': ['test_agent_2'],
                        'deprive': {'value': 2}
                    },
                },
            }
        )
        test_agent_2 = Agent(
            mock_model_with_currencies,
            'test_agent_2',
            capacity={'test_currency_1': 2},
        )
        mock_model_with_currencies.agents = {
            'test_agent': test_agent,
            'test_agent_2': test_agent_2,
        }
        test_agent.register()
        test_agent_2.register()
        # Start with deprive buffer full
        assert test_agent.attributes['in_test_currency_1_deprive'] == 2
        # Actual is equal to half of target; half of the active are deprived
        process_kwargs = dict(dT=1, direction='in', currency='test_currency_1',
                              flow=test_agent.flows['in']['test_currency_1'], influx={},
                              target=10, actual=5)
        test_agent.process_flow(**process_kwargs)
        assert test_agent.attributes['in_test_currency_1_deprive'] == 1.5
        test_agent.process_flow(**process_kwargs)
        assert test_agent.attributes['in_test_currency_1_deprive'] == 1
        test_agent.process_flow(**process_kwargs)
        assert test_agent.attributes['in_test_currency_1_deprive'] == 0.5
        test_agent.process_flow(**process_kwargs)
        assert test_agent.attributes['in_test_currency_1_deprive'] == 0
        # Deprive buffer is empty, so half of the active die
        test_agent.process_flow(**process_kwargs)
        assert test_agent.active == 2
        # Actual is equal to target again, buffer resets
        process_kwargs['actual'] = 10
        test_agent.process_flow(**process_kwargs)
        assert test_agent.attributes['in_test_currency_1_deprive'] == 2
        # Actual equal to zero; all of active are deprived
        process_kwargs['actual'] = 0
        test_agent.process_flow(**process_kwargs)
        assert test_agent.active == 2
        assert test_agent.attributes['in_test_currency_1_deprive'] == 1
        # Buffer responds to DT
        process_kwargs['dT'] = 0.5
        test_agent.process_flow(**process_kwargs)
        assert test_agent.active == 2
        assert test_agent.attributes['in_test_currency_1_deprive'] == 0.5
        test_agent.process_flow(**process_kwargs)
        assert test_agent.active == 2
        assert test_agent.attributes['in_test_currency_1_deprive'] == 0
        # When deprive buffer is empty, all active die (per dT)
        process_kwargs['dT'] = 1
        test_agent.process_flow(**process_kwargs)
        assert test_agent.active == 0
        assert test_agent.cause_of_death == 'test_agent deprived of test_currency_1'

class TestAgentStep:
    def test_agent_step_empty(self, basic_model):
        test_agent = Agent(basic_model, 'test_agent')
        assert not test_agent.registered
        test_agent.step()
        assert test_agent.registered
        assert test_agent.records == {
            'active': [1],
            'cause_of_death': None,
            'storage': {},
            'attributes': {'age': [1]},
            'flows': {'in': {}, 'out': {}}
        }
        test_agent.step()
        assert test_agent.attributes['age'] == 2
        assert test_agent.records['attributes']['age'] == [1, 2]
        assert test_agent.records['active'] == [1, 1]

    def test_agent_step_threshold(self, mock_model_with_currencies):
        test_agent = Agent(
            mock_model_with_currencies,
            'test_agent',
            thresholds={'test_currency_1': {
                'path': 'in_test_currency_1_ratio',
                'limit': '<',
                'value': 0.5,
            }},
            flows={'in': {'test_currency_1': {'connections': ['test_structure']}}},
        )
        test_structure = Agent(
            mock_model_with_currencies,
            'test_structure',
            capacity={'test_currency_1': 10, 'test_currency_2': 10},
            storage={'test_currency_1': 5, 'test_currency_2': 5},
        )
        mock_model_with_currencies.agents = {
            'test_agent': test_agent,
            'test_structure': test_structure,
        }
        test_agent.register()
        test_structure.register()
        # Threshold not met
        test_agent.step()
        assert test_agent.active
        assert test_agent.cause_of_death == None
        assert test_agent.records['flows']['in']['test_currency_1']['test_structure'] == [0]
        assert test_agent.attributes['age'] == 1
        # Threshold met
        test_structure.storage['test_currency_1'] = 4.9
        test_agent.step()
        assert test_agent.active == 0
        assert test_agent.cause_of_death == 'test_agent passed test_currency_1 threshold'
        assert test_agent.records['flows']['in']['test_currency_1']['test_structure'] == [0, 0]
        assert test_agent.attributes['age'] == 2
        # Records continue to accumulate even after agent is dead, but age stays same
        test_agent.step()
        assert test_agent.records['flows']['in']['test_currency_1']['test_structure'] == [0, 0, 0]
        assert test_agent.attributes['age'] == 2

    def test_agent_step_flows(self, mock_model_with_currencies):
        test_agent = Agent(
            mock_model_with_currencies,
            'test_agent',
            flows={
                'in': {
                    'test_currency_1': {
                        'value': 1,
                        'connections': ['test_structure'],
                    }
                },
                'out': {
                    'test_currency_2': {
                        'value': 1,
                        'connections': ['test_structure'],
                    }
                }
            }
        )
        test_structure = Agent(
            mock_model_with_currencies,
            'test_structure',
            capacity={'test_currency_1': 10, 'test_currency_2': 10},
            storage={'test_currency_1': 5, 'test_currency_2': 5},
        )
        mock_model_with_currencies.agents = {
            'test_agent': test_agent,
            'test_structure': test_structure,
        }
        test_agent.register()
        test_structure.register()
        test_agent.step()
        assert test_agent.records['flows']['in']['test_currency_1']['test_structure'] == [-1]
        assert test_agent.records['flows']['out']['test_currency_2']['test_structure'] == [1]
        assert test_structure.storage['test_currency_1'] == 4
        assert test_structure.storage['test_currency_2'] == 6
        test_agent.step(dT=0.5)
        assert test_agent.records['flows']['in']['test_currency_1']['test_structure'] == [-1, -0.5]
        assert test_agent.records['flows']['out']['test_currency_2']['test_structure'] == [1, 0.5]
        assert test_structure.storage['test_currency_1'] == 3.5
        assert test_structure.storage['test_currency_2'] == 6.5

    def test_agent_step_multi_connections(self, mock_model_with_currencies):
        test_agent = Agent(
            mock_model_with_currencies,
            'test_agent',
            flows={
                'in': {
                    'test_currency_1': {
                        'value': 2,
                        'connections': ['test_structure_1', 'test_structure_2'],
                    }
                },
            }
        )
        test_structure_1 = Agent(
            mock_model_with_currencies,
            'test_structure_1',
            capacity={'test_currency_1': 10},
            storage={'test_currency_1': 5},
        )
        test_structure_2 = Agent(
            mock_model_with_currencies,
            'test_structure_2',
            capacity={'test_currency_1': 10},
            storage={'test_currency_1': 5},
        )
        mock_model_with_currencies.agents = {
            'test_agent': test_agent,
            'test_structure_1': test_structure_1,
            'test_structure_2': test_structure_2,
        }
        test_agent.register()
        test_structure_1.register()
        test_structure_2.register()

        # If available, use first connection
        test_agent.step()
        assert test_agent.records['flows']['in']['test_currency_1'] == {
            'test_structure_1': [-2],
            'test_structure_2': [0],
        }
        assert test_structure_1.storage['test_currency_1'] == 3
        assert test_structure_2.storage['test_currency_1'] == 5

        # If partially available, split between first and second connections
        test_structure_1.storage['test_currency_1'] = 1
        test_agent.step()
        test_agent.step()
        assert test_agent.records['flows']['in']['test_currency_1'] == {
            'test_structure_1': [-2, -1, 0],
            'test_structure_2': [0, -1, -2],
        }
        assert test_structure_1.storage['test_currency_1'] == 0
        assert test_structure_2.storage['test_currency_1'] == 2

class TestAgentKill:
    def test_agent_kill(self, basic_model):
        test_agent = Agent(basic_model, 'test_agent', amount=2)
        basic_model.agents = {'test_agent': test_agent}
        test_agent.register()
        test_agent.kill('test_reason', 1)
        assert test_agent.active == 1
        assert test_agent.cause_of_death == None
        test_agent.kill('test_reason', 1)
        assert test_agent.active == 0
        assert test_agent.cause_of_death == 'test_reason'