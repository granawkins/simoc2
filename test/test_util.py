import pytest
import datetime
from copy import deepcopy

from ..simoc_abm.util import (load_data_file, 
                                get_default_agent_data,
                                get_default_currency_data,
                                merge_json,
                                recursively_clear_lists,
                                evaluate_reference, 
                                pdf,
                                sample_norm,
                                sample_clipped_norm,
                                sample_sigmoid,
                                evaluate_growth,
                                parse_data)

class TestDataFilesHandling:
    def test_load_data_files(self):
        agent_desc = load_data_file('agent_desc.json')
        assert 'wheat' in agent_desc, 'Failed to load agent_desc'
        with pytest.raises(AssertionError):
            load_data_file('nonexistent_file.json')
        with pytest.raises(AssertionError):
            load_data_file('agent_desc.json', data_dir='nonexistent_dir')
    
    def test_get_default_agent_data(self):
        wheat_data = get_default_agent_data('wheat')
        assert all([k in wheat_data for k in ['amount', 'properties', 'flows']])
    
    def test_get_default_currency_data(self):
        currency_data = get_default_currency_data()
        for k, v in currency_data.items():
            assert 'currency_type' in v
            if v['currency_type'] == 'class':
                assert 'currencies' in v
            else:
                assert 'class' in v
                assert v['class'] in currency_data

    def test_merge_json(self):
        default = {'a': 'red', 'b': 2, 'c': {'d': 3, 'e': 4}, 'f': [1, 2, 3]}
        to_merge = {'a': 'blue', 'c': {'d': 6}, 'f': [3, 4, 5]}
        merged = merge_json(default, to_merge)
        assert merged == {'a': 'blue', 'b': 2, 'c': {'d': 6, 'e': 4}, 'f': [1, 2, 3, 4, 5]}
    
    def test_recursively_clear_lists(self):
        data = {
            'a': 'string', 
            'b': 1,
            'c': 2.1,
            'd': ['e', 'f'], 
            'g': {'h': 'string2',
                  'i': ['j', 'k']}}
        data = recursively_clear_lists(data)
        assert data == {
            'a': 'string',
            'b': 1,
            'c': 2.1,
            'd': [],
            'g': {'h': 'string2',
                  'i': []}}

class MockAgent:
    def __init__(self, model):
        self.model = model
        self.attributes = {'test_attribute': 1}
        self.storage = {'test_currency_1': 1, 'test_currency_2': 2}
        self.flows = {
            'in': {        
                'test_currency_1': {
                    'value': 1,
                    'connections': ['test_agent_2']
                }
            },
            'out': {
                'test_currency_1': {
                    'value': 1,
                    'connections': ['test_agent_2']
                }
            },
        }
    def view(self, view):
        if view in ('test_currency_1', 'test_currency_2'):
            return {view: self.storage[view]}
        elif view == 'test_currency_class':
            return deepcopy(self.storage)

class MockModel:
    floating_point_precision = 6
    agents = {}
    currencies = {
        'test_currency_1': {
            'currency_type': 'currency',
            'class': 'test_currency_class'
        },
        'test_currency_2': {
            'currency_type': 'currency',
            'class': 'test_currency_class'
        },
        'test_currency_class': {
            'currency_type': 'class',
            'currencies': ['test_currency_1', 'test_currency_2']
        }
    }

@pytest.fixture(scope='function')
def test_model():
    model = MockModel()
    test_agent_1 = MockAgent(model)
    test_agent_2 = MockAgent(model)
    test_agent_1.model = model
    test_agent_2.model = model
    model.agents = {
        'test_agent_1': test_agent_1,
        'test_agent_2': test_agent_2,
    }
    return model

class TestEvaluateReference:
    def test_evaluate_reference_attribute(self, test_model):
        reference = {
            'path': 'test_attribute',
            'limit': '>',
            'value': 1
        }
        test_agent_1 = test_model.agents['test_agent_1']
        assert not evaluate_reference(test_agent_1, **reference)
        test_agent_1.attributes['test_attribute'] = 2
        assert evaluate_reference(test_agent_1, **reference)

    def test_evaluate_reference_storage(self, test_model):
        reference = {
            'path': 'test_currency_1',
            'limit': '>',
            'value': 1
        }
        test_agent_1 = test_model.agents['test_agent_1']
        assert not evaluate_reference(test_agent_1, **reference)
        test_agent_1.storage['test_currency_1'] = 2
        assert evaluate_reference(test_agent_1, **reference)

    def test_evaluate_reference_ratio(self, test_model):
        reference = {
            'path': 'test_currency_1_ratio',
            'limit': '>',
            'value': 0.5
        }
        test_agent_1 = test_model.agents['test_agent_1']
        assert not evaluate_reference(test_agent_1, **reference)
        reference['limit'] = '<'
        assert evaluate_reference(test_agent_1, **reference)

    def test_evaluate_reference_connected(self, test_model):
        reference = {
            'path': 'in_test_currency_1',
            'limit': '>',
            'value': 1
        }
        test_agent_1 = test_model.agents['test_agent_1']
        assert not evaluate_reference(test_agent_1, **reference)
        test_agent_2 = test_model.agents['test_agent_2']
        test_agent_2.storage['test_currency_1'] = 2
        assert evaluate_reference(test_agent_1, **reference)

class TestGrowthFuncs:
    def test_growth_pdf(self):
        _cache = {}
        results = [pdf(x, 0.5, _cache) for x in range(-4, 5)]
        # middle value should be highest, symmetrical either side
        assert results[4] == max(results)
        for i in range(4):
            assert results[i] == results[-i-1]
        assert list(_cache.values()) == results

    def test_growth_sample_norm(self):
        # Default: 0 < y < 1, x_center = 0.5
        n_samples = 100
        results = [sample_norm(x/100, n_samples=n_samples) for x in range(1, n_samples)]
        assert sum(results)/len(results) == pytest.approx(1, abs=0.02)
        # middle value should be highest, symmetrical either side
        midpoint = n_samples//2-1
        assert results[midpoint] == max(results)
        for i in range(midpoint):
            assert results[i] == pytest.approx(results[-i-1])

        # Shift center
        x_center = 0.25
        results = [sample_norm(x/1000, center=x_center) for x in range(1000)]
        assert sum(results)/len(results) == pytest.approx(1, abs=0.01)
        assert results[250] == max(results)

        # TODO: Shift stdev

    def test_growth_sample_clippped_norm(self):
        results = [sample_clipped_norm(x/10) for x in range(1, 10)]
        assert max(results) == 1
        assert results[4] == max(results)
        for i in range(4):
            assert results[i] == pytest.approx(results[-i-1])

    def test_growth_sample_sigmoid(self):
        results = [sample_sigmoid(x/1000) for x in range(1000)]
        assert results[-1] == max(results)
        assert all(results[i] <= results[i+1] for i in range(len(results)-1))
        # Derivative (slope) is greatest at center
        derivatives = [results[i+1] - results[i] for i in range(len(results)-1)]
        for i in range(500):
            assert derivatives[i] <= derivatives[i+1]
        assert derivatives[500] == max(derivatives)
        for i in range(500, 998):
            assert derivatives[i] >= derivatives[i+1]

@pytest.fixture
def mock_agent():
    class MockAgent:
        def __init__(self, model):
            self.model = model
            self.attributes = {'age': 10}
            self.properties = {'lifetime': {'value': 20}}
    class MockModel:
        time = datetime.datetime(2019, 1, 1, 12)
    return MockAgent(MockModel())

class TestEvaluateGrowth:
    def test_evaluate_growth_daily(self, mock_agent):
        mode = 'daily'
        params = {'type': 'norm'}
        daily_vals = []
        for hour in range(24):
            mock_agent.model.time = datetime.datetime(2019, 1, 1, hour)
            daily_vals.append(evaluate_growth(mock_agent, mode, params))
        # Max growth (1) at noon
        assert daily_vals[12] == max(daily_vals)
        # Min growth (nearly 0) at midnight
        assert daily_vals[0] == min(daily_vals)

    def test_evaluate_growth_lifetime(self, mock_agent):
        mode = 'lifetime'
        params = {'type': 'sigmoid'}
        # Halfway growth at age 10/20
        assert evaluate_growth(mock_agent, mode, params) == 0.5
        # Max growth (nearly 1) at age 20/20
        mock_agent.attributes['age'] = 20
        assert 0.999 < evaluate_growth(mock_agent, mode, params) < 1.0

@pytest.fixture
def mock_data():
    return {
        'model_string_attribute': 'test',
        'model_int_attribute': 1,
        'test_agent': {
            'agent_string_attribute': 'test',
            'agent_int_attribute': 1,
            'agent_list_attribute': [1, 2, 3],
            'agent_dict_attribute': {
                'dict_attr_1': [2, 3, 4],
                'dict_attr_2': [3, 4, 5],
                'dict_attr_3': [4, 5, 6],
            }
        }
    }

class TestParseData:
    def test_parse_data_static_field(self, mock_data):
        model_string_attr = parse_data(mock_data, ['model_string_attribute'])
        assert model_string_attr == 'test'
        model_int_attr = parse_data(mock_data, ['model_int_attribute'])
        assert model_int_attr == 1
        agent_string_attr = parse_data(mock_data, ['test_agent', 'agent_string_attribute'])
        assert agent_string_attr == 'test'
        agent_int_attr = parse_data(mock_data, ['test_agent', 'agent_int_attribute'])
        assert agent_int_attr == 1

    def test_parse_data_missing_field(self, mock_data):
        missing_value = parse_data(mock_data, ['missing_value'])
        assert missing_value == None
        # But still propagate Zeros
        mock_data['model_int_attribute'] = 0
        zero_value = parse_data(mock_data, ['model_int_attribute'])
        assert zero_value == 0

    def test_parse_data_dict_keys(self, mock_data):
        single_field = parse_data(mock_data, ['test_agent', 'agent_dict_attribute', 'dict_attr_1'])
        assert single_field == [2, 3, 4]
        all_fields = parse_data(mock_data, ['test_agent', 'agent_dict_attribute', '*'])
        assert all_fields == mock_data['test_agent']['agent_dict_attribute']
        selected_fields = parse_data(mock_data, ['test_agent', 'agent_dict_attribute', 'dict_attr_1,dict_attr_2'])
        assert selected_fields == {'dict_attr_1': [2, 3, 4], 'dict_attr_2': [3, 4, 5]}
        summed_fields = parse_data(mock_data, ['test_agent', 'agent_dict_attribute', 'SUM'])
        assert summed_fields == [9, 12, 15]

    def test_parse_data_list(self, mock_data):
        all_items = parse_data(mock_data, ['test_agent', 'agent_list_attribute', '*'])
        assert all_items == [1, 2, 3]
        single_item = parse_data(mock_data, ['test_agent', 'agent_list_attribute', 1])
        assert single_item == 2
        slice_item = parse_data(mock_data, ['test_agent', 'agent_list_attribute', '0:2'])
        assert slice_item == [1, 2]
