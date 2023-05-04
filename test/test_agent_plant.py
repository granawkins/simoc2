import copy
import datetime
import pytest
import numpy as np
from ..agent_model.agents import PlantAgent, BaseAgent
from ..agent_model.util import get_default_currency_data, get_default_agent_data

@pytest.fixture
def basic_kwargs():
    return {
        'flows': {
            'in': {'co2': {'value': 1, 'connections': ['test_greenhouse']}, 
                   'par': {'value': 1, 'connections': ['test_greenhouse']}},
            'out': {'biomass': {'value': 1, 'connections': ['test_plant']}, 
                    'inedible_biomass': {'value': 0.1, 'connections': ['test_greenhouse']}},
        },
        'capacity': {'biomass': 10},
        'properties': {'photoperiod': {'value': 12},
                       'lifetime': {'value': 600},
                       'par_baseline': {'value': 2}}}

@pytest.fixture
def wheat_kwargs():
    return get_default_agent_data('wheat')

@pytest.fixture
def mock_model():
    class MockModel:
        floating_point_precision = 6
        agents = {}
        time = datetime.datetime(2020, 1, 1)
        currencies = get_default_currency_data()
        step_num = 0
        def register(self):
            for agent in self.agents.values():
                agent.register()
        def step(self):
            self.step_num += 1
            self.time += datetime.timedelta(hours=1)
            for agent in self.agents.values():
                agent.step()
    return MockModel()

@pytest.fixture
def basic_model(mock_model, basic_kwargs):
    test_agent = PlantAgent(mock_model, 'test_plant', **basic_kwargs)
    test_greenhouse = BaseAgent(
        mock_model, 'test_greenhouse', 
        capacity={'co2': 10, 'par': 10, 'inedible_biomass': 10},
        storage={'co2': 10, 'par': 10},
    )
    mock_model.agents = {
        'test_plant': test_agent, 
        'test_greenhouse': test_greenhouse,
    }
    return mock_model

@pytest.fixture
def wheat_model(mock_model, wheat_kwargs):
    wheat_kwargs['flows']['in']['par']['connections'] = ['lamp']
    wheat_agent = PlantAgent(mock_model, 'wheat', **wheat_kwargs)
    greenhouse = BaseAgent(
        mock_model, 'greenhouse', 
        capacity={'co2': 1e5, 'o2': 1e5, 'h2o': 1e5},
        storage={'co2': 1e5},
    )
    water_storage = BaseAgent(
        mock_model, 'water_storage',
        capacity={'potable': 1e5},
        storage={'potable': 1e5},
    )
    nutrient_storage = BaseAgent(
        mock_model, 'nutrient_storage',
        capacity={'fertilizer': 1e5, 'inedible_biomass': 10},
        storage={'fertilizer': 1e5},
    )
    food_storage = BaseAgent(
        mock_model, 'food_storage',
        capacity={'wheat': 1e5},
    )
    lamp = BaseAgent(
        mock_model, 'lamp',
        capacity={'par': 1e5},
        storage={'par': 1e5},
    )
    mock_model.agents = {
        'wheat': wheat_agent, 
        'greenhouse': greenhouse,
        'water_storage': water_storage,
        'nutrient_storage': nutrient_storage,
        'food_storage': food_storage,
        'lamp': lamp,
    }
    return mock_model

class TestAgentPlant:
    def test_agent_plant_init_basic(self, basic_kwargs):
        """
        - iniitalize default attributes correctly
        - respond to delay correctly
        - initialize non-serialized vars
        """
        test_model = object()

        # Test required attributes
        kwargs = copy.deepcopy(basic_kwargs)
        del kwargs['flows']['out']['inedible_biomass']
        with pytest.raises(ValueError):
            test_plant = PlantAgent(test_model, 'test_plant', **kwargs)

        # Initialize with default attributes and non-serialized vars
        test_plant = PlantAgent(test_model, 'test_plant', **basic_kwargs)
        for attr, value in test_plant.default_attributes.items():
            assert test_plant.attributes[attr] == value
        assert test_plant.daily_growth == []
        assert test_plant.max_growth == 0
        assert test_plant.active == 1

        # Initialize with custom attributes, check delay_start
        test_attributes = {'delay_start': 10, 'test_attribute': 1}
        kwargs = {**basic_kwargs, 'attributes': test_attributes}
        test_plant = PlantAgent(test_model, 'test_plant', **kwargs)
        assert test_plant.attributes['delay_start'] == 10
        assert test_plant.attributes['test_attribute'] == 1
        assert test_plant.active == 0

    def test_agent_plant_init_wheat(self, wheat_kwargs):
        test_model = object()
        PlantAgent(test_model, 'wheat', **wheat_kwargs)
        assert True

    def test_agent_plant_register(self, basic_model):
        """
        - calculate growth rate array correctly
        - calculate max_growth correctly
        - intitialize cache in Model
        """
        plant_agent = basic_model.agents['test_plant']
        plant_agent.register()
        assert plant_agent.registered

        expected_daily_growth = np.zeros(24)
        expected_daily_growth[6:18] = 2
        assert np.array_equal(plant_agent.daily_growth, expected_daily_growth)
        expected_max_growth = 1 * 600
        assert plant_agent.max_growth == expected_max_growth
        expected_co2_response_cache = dict(step_num=0, cu_factor=1, te_factor=1)
        assert basic_model._co2_response_cache == expected_co2_response_cache

    def test_agent_plant_get_flow_value(self, wheat_model):
        """
        - return 0 for non-grown values when grown
        """
        wheat_model.register()
        wheat_plant = wheat_model.agents['wheat']

        # Check all non-grown flows are positive when plant is not grown
        assert wheat_plant.attributes['grown'] == False
        influx = {}
        wheat_plant.attributes['growth_rate'] = 0.5  # Mid-life so all flows > 0
        wheat_model.time = datetime.datetime(2020, 1, 1, 12)
        for direction, flows in wheat_plant.flows.items():
            for currency, flow in flows.items():
                if currency == 'par': 
                    continue  # Dummy flow
                flow_value = wheat_plant.get_flow_value(1, direction, currency, flow, influx)
                if ('criteria' in flow and flow['criteria'][0]['path'] == 'grown'):
                    assert flow_value == 0, f'Flow value for {currency} should be 0'
                else:
                    assert flow_value > 0, f'Flow value for {currency} should be > 0'
                influx[currency] = 1

        # Check all non-grown flows are when plant is grown, vice versa
        wheat_plant.attributes['grown'] = True
        wheat_plant.storage['biomass'] = 10  # So biomass-wieghted flows > 0
        for direction, flows in wheat_plant.flows.items():
            for currency, flow in flows.items():
                if currency == 'par': 
                    continue  # Dummy flow
                flow_value = wheat_plant.get_flow_value(1, direction, currency, flow, influx)
                if ('criteria' in flow and flow['criteria'][0]['path'] == 'grown'):
                    assert flow_value > 0, f'Flow value for {currency} should be > 0'
                else:
                    assert flow_value == 0, f'Flow value for {currency} should be 0'
                influx[currency] = 1

    def test_agent_plant_calculate_co2_response(self, wheat_model):
        """
        - Return correct values for different co2 levels
        - Return floor/ceiling values outside range
        - Return 1 for c4 plants
        - Manage cache correctly
        """
        wheat_model.register()
        wheat_agent = wheat_model.agents['wheat']
        # test cache
        wheat_model._co2_response_cache = dict(
            step_num=0, cu_factor='test_cu', te_factor='test_te')
        assert wheat_agent._calculate_co2_response() == ('test_cu', 'test_te')

        # test c3 plant
        def expected_cu_factor(co2_actual, t_mean=25):
            """Copied from the body of PlantAgent"""
            tt = (163 - t_mean) / (5 - 0.1 * t_mean)
            t_mean = 25 # Mean temperature for timestep.
            tt = (163 - t_mean) / (5 - 0.1 * t_mean) # co2 compensation point
            numerator = (co2_actual - tt) * (350 + 2 * tt)
            denominator = (co2_actual + 2 * tt) * (350 - tt)
            cu_ratio = numerator/denominator
            # Invert the above to give *decrease* in growth for less than ideal CO2
            crf_ideal = 1.2426059597016264  # At 700ppm, the above equation gives this value
            cu_ratio = cu_ratio / crf_ideal
            return cu_ratio
        def expected_te_factor(co2_actual):
            return np.interp(co2_actual, [350, 700], [1/1.37, 1])
        test_cases = [
            # ppm, cu_factor, te_factor
            (300, 1/1.242605, 1/1.37),
            (500, expected_cu_factor(500), expected_te_factor(500)),
            (600, expected_cu_factor(600), expected_te_factor(600)),
            (750, 1, 1),
        ]
        for ppm, expected_cu, expected_te in test_cases:
            wheat_model._co2_response_cache['step_num'] = -1
            greenhouse_atmosphere = {
                'co2': ppm/1e6,
                'o2': 1 - ppm/1e6,
            }
            wheat_model.agents['greenhouse'].storage = greenhouse_atmosphere
            cu_factor, te_factor = wheat_agent._calculate_co2_response()
            assert cu_factor == pytest.approx(expected_cu), f'cu_factor failed at {ppm} ppm'
            assert te_factor == pytest.approx(expected_te), f'te_factor failed at {ppm} ppm'
            assert wheat_model._co2_response_cache['step_num'] == 0
        
        # test non-c3 plant
        wheat_model._co2_response_cache['step_num'] = -1
        gh_atmo = {'co2': 500/1e6, 'o2': 1 - 500/1e6}
        wheat_model.agents['greenhouse'].storage = gh_atmo
        wheat_agent.properties['carbon_fixation']['value'] = 'c4'
        cu_factor, te_factor = wheat_agent._calculate_co2_response()
        assert cu_factor == 1

    def test_agent_plant_kill(self, basic_model, basic_kwargs):
        """
        - Reduce biomass and add to storage correctly, with or without n_dead spec
        """
        basic_kwargs['amount'] = 10
        basic_kwargs['storage'] = {'biomass': 10}
        basic_model.agents['test_plant'] = PlantAgent(
            basic_model, 'test_agent', **basic_kwargs)
        basic_model.register()
        test_plant = basic_model.agents['test_plant']
        test_greenhouse = basic_model.agents['test_greenhouse']

        test_plant.kill('test_reason', 5)
        assert test_plant.storage['biomass'] == 5
        assert test_plant.cause_of_death == None
        assert test_greenhouse.storage['inedible_biomass'] == 5

        test_plant.kill('test_reason_2')
        assert test_plant.storage['biomass'] == 0
        assert test_plant.cause_of_death == 'test_reason_2'
        assert test_greenhouse.storage['inedible_biomass'] == 10

class TestAgentPlantStep:

    def test_agent_plant_step_delay(self, wheat_model, wheat_kwargs):
        wheat_model.time = datetime.datetime(2020, 1, 1, 12)
        wheat_kwargs['attributes']['delay_start'] = 2
        wheat_model.agents['wheat'] = PlantAgent(wheat_model, 'wheat', **wheat_kwargs)
        wheat_agent = wheat_model.agents['wheat']
        wheat_model.register()
        for i in range(3):
            assert wheat_agent.attributes['delay_start'] == max(0, 2 - i)
            if i < 2:
                assert wheat_agent.active == 0
                assert wheat_agent.attributes['age'] == 0
            else:
                assert wheat_agent.active > 0
            wheat_agent.step()
            wheat_model.time += datetime.timedelta(hours=1)
        assert wheat_agent.attributes['age'] > 0
        wheat_records = wheat_agent.get_records()
        wheat_biomass_out = wheat_records['flows']['out']['biomass']['wheat']
        assert wheat_biomass_out[:2] == [0, 0]
        assert wheat_biomass_out[2] > 0

    def test_agent_plant_step_reproduce(self, wheat_model, wheat_kwargs):
        wheat_model.register()
        wheat_agent = wheat_model.agents['wheat']
        ns_agent = wheat_model.agents['nutrient_storage']
        fs_agent = wheat_model.agents['food_storage']
        lifetime = wheat_agent.properties['lifetime']['value']
        expected_lifetime_biomass = lifetime * wheat_agent.flows['out']['biomass']['value']
        for _ in range(lifetime):
            wheat_model.step()
        assert wheat_agent.attributes['age'] == lifetime
        assert wheat_agent.storage['biomass'] == pytest.approx(expected_lifetime_biomass)
        assert 'inedible_biomass' not in ns_agent.storage
        assert 'wheat' not in fs_agent.storage

        wheat_model.step()
        assert wheat_agent.attributes['age'] == 0
        assert wheat_agent.storage['biomass'] == 0
        harvest_index = wheat_agent.properties['harvest_index']['value']
        expected_wheat = pytest.approx(expected_lifetime_biomass * harvest_index)        
        expected_inedib = pytest.approx(expected_lifetime_biomass * (1 - harvest_index))
        assert fs_agent.storage['wheat'] == expected_wheat
        assert ns_agent.storage['inedible_biomass'] == expected_inedib

    def test_agent_plant_step(self):
        """
        - handle delay start correctly
        - handle grown/reproduce correctly
        - Set daily_growth_factor correctly
        - Set par_factor correctly with both electric and sun light
        - Set growth rate correctly
        - Call/Set _calculate_co2_response correctly
        """

        pass