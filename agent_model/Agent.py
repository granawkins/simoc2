import math
from copy import deepcopy
from collections import defaultdict

import numpy as np

from .util import evaluate_reference, evaluate_growth, recursively_clear_lists

class Agent:
    # ------------- SETUP  ------------- #
    def __init__(self, model, agent_id, amount=1, description=None, 
                 agent_class=None, properties=None, capacity=None, 
                 thresholds=None, flows=None, cause_of_death=None, active=None, 
                 storage=None, attributes=None):
        """Create an agent with the given parameters.
        
        Args:
            model (AgentModel): AgentModel instance
            agent_id (str): A unique string
            amount (int): Starting/Maximum number alive
            description (str): Plaintext description
            agent_class (str): Agent class name
            properties (dict): Static vars, 'volume'
            capacity (dict): Max storage per currency per individual
            thresholds (dict): Env. conditions to die
            flows (dict): Exchanges w/ other agents
            cause_of_death (str): Reason for death
            active (int): Current number alive
            storage (dict): Currencies stored by total amount
            attributes (dict): Dynamic vars, 'te_factor'
        """
        # -- STATIC
        self.agent_id = agent_id
        self.amount = 1 if amount is None else amount 
        self.description = '' if description is None else description 
        self.agent_class = '' if agent_class is None else agent_class 
        self.properties = {} if properties is None else deepcopy(properties)
        self.capacity = {} if capacity is None else deepcopy(capacity)
        self.thresholds = {} if thresholds is None else deepcopy(thresholds)
        self.flows = {'in': {}, 'out': {}}
        for direction in ('in', 'out'):
            if flows is not None and direction in flows:
                self.flows[direction] = deepcopy(flows[direction])
        # -- DYNAMIC
        self.cause_of_death = cause_of_death
        self.active = amount if active is None else deepcopy(active)
        self.storage = {} if storage is None else deepcopy(storage)
        self.attributes = {} if attributes is None else deepcopy(attributes)
        # -- NON-SERIALIZED
        self.model = model
        self.registered = False
        self.records = {}

    def register(self, record_initial_state=False):
        """Check and setup agent after all agents have been added to Model.
        
        Args:
            record_initial_state (bool): Whether to include a value for 
                'step 0'; True for new simulations, false when loading
        """
        if self.registered:
            return
        if 'age' not in self.attributes:
            self.attributes['age'] = 0
        for currency in self.storage:
            if currency not in self.capacity:
                raise ValueError(f'Agent {self.agent_id} has storage for '
                                 f'{currency} but no capacity.')
            elif self.storage[currency] > self.capacity[currency] * self.active:
                raise ValueError(f'Agent {self.agent_id} has more storage '
                                 f'for {currency} than capacity.')
        # Initialize flow attributes and records, check connections
        flow_records = {'in': defaultdict(dict), 'out': defaultdict(dict)}
        for direction, flows in self.flows.items():
            for currency, flow in flows.items():
                self.register_flow(direction, currency, flow)
                for conn in flow['connections']:
                    agent = self.model.agents[conn]
                    for _currency in agent.view(currency):
                        record = [] if not record_initial_state else [0]
                        flow_records[direction][_currency][conn] = record

        # Initialize records skeleton
        self.records = {
            'active': [] if not record_initial_state else [self.active],
            'cause_of_death': self.cause_of_death,
            'storage': {currency: [] if not record_initial_state 
                        else [self.storage.get(currency, 0)] 
                        for currency in self.capacity},
            'attributes': {attr: [] if not record_initial_state 
                           else [self.attributes[attr]] 
                           for attr in self.attributes},
            'flows': flow_records,
        }
        self.registered = True

    def register_flow(self, direction, currency, flow):
        """Check flow, setup attributes and records. Overloadable by subclasses."""
        # Check flow fields
        allowed_fields = {'value', 'flow_rate', 'criteria', 'connections', 
                          'deprive', 'weighted', 'requires', 'growth'}
        for field in flow:
            if field not in allowed_fields:
                raise ValueError(f'Flow field {field} not allowed')
        # Initialize attributes
        if 'criteria' in flow:
            for i, criterion in enumerate(flow['criteria']):
                if 'buffer' in criterion:
                    buffer_attr = f'{direction}_{currency}_criteria_{i}_buffer'
                    self.attributes[buffer_attr] = criterion['buffer']
        if 'deprive' in flow:
            deprive_attr = f'{direction}_{currency}_deprive'
            self.attributes[deprive_attr] = flow['deprive']['value']
        if 'growth' in flow:
            for mode, params in flow['growth'].items():
                growth_attr = f'{direction}_{currency}_{mode}_growth_factor'
                self.attributes[growth_attr] = evaluate_growth(self, mode, params)
        # Check flow connections
        for agent in flow['connections']:
            if agent not in self.model.agents:
                raise ValueError(f'Agent {agent} not registered')
            currency_type = self.model.currencies[currency]['currency_type']
            if currency_type == 'currency':
                if currency not in self.model.agents[agent].capacity:
                    raise ValueError(f'Agent {agent} does not store {currency}')
            else:
                class_currencies = self.model.currencies[currency]['currencies']
                if not any(c in self.model.agents[agent].capacity 
                           for c in class_currencies):
                    raise ValueError(f'Agent {agent} does not store any '
                                     f'currencies of class {currency}')

    # ------------- INSPECT ------------- #
    def view(self, view):
        """Return a dict with storage amount for single currency or all of a class"""
        currency_type = self.model.currencies[view]['currency_type']
        if currency_type == 'currency':
            if view not in self.storage:
                return {view: 0}
            return {view: self.storage[view]}
        elif currency_type == 'class':
            class_currencies = self.model.currencies[view]['currencies']
            return {currency: self.storage.get(currency, 0)
                    for currency in class_currencies
                    if currency in self.capacity}
        
    def serialize(self):
        """Return json-serializable dict of agent attributes"""
        serializable = {'agent_id', 'amount', 'description', 'agent_class', 
                        'properties', 'capacity', 'thresholds', 'flows',
                        'cause_of_death', 'active', 'storage', 'attributes'}
        output = {k: deepcopy(getattr(self, k)) for k in serializable}
        return output

    def get_records(self, static=False, clear_cache=False):
        """Return records dict and optionally clear cache"""
        output = deepcopy(self.records)
        if static:
            static_records = self.serialize()
            non_static = ('cause_of_death', 'active', 'storage', 'attributes')
            for k in non_static:
                del static_records[k]
            output['static'] = static_records
        if clear_cache:
            self.records = recursively_clear_lists(self.records)
        return output

    def save(self, records=False):
        """Return a serializable copy of the agent"""
        output = self.serialize()
        if records:
            output['records'] = self.get_records()
        return output

    # ------------- UPDATE ------------- #
    def increment(self, currency, value):
        """Increment currency in storage as available, return actual receipt"""
        if value == 0:  # If currency_class, return dict of currencies
            available = self.view(currency)
            return {k: 0 for k in available.keys()}
        elif value < 0:  # Can be currency or currency_class
            available = self.view(currency)
            total_available = sum(available.values())
            if total_available == 0:
                return available
            actual = -min(-value, total_available)
            increment = {currency: actual * stored/total_available
                         for currency, stored in available.items()}
            for _currency, amount in increment.items():
                if amount != 0:
                    self.storage[_currency] += amount
            return increment
        elif value > 0:  # Can only be currency
            if self.model.currencies[currency]['currency_type'] != 'currency':
                raise ValueError(f'Cannot increment agent by currency class ({currency})')
            if currency not in self.capacity:
                raise ValueError(f'Agent does not store {currency}')
            if currency not in self.storage:
                self.storage[currency] = 0
            total_capacity = self.capacity[currency] * self.active
            remaining_capacity = total_capacity - self.storage[currency]
            actual = min(value, remaining_capacity)
            self.storage[currency] += actual
            return {currency: actual}
        
    def get_flow_value(self, dT, direction, currency, flow, influx):
        """Update flow state pre-exchange and return target value. 
        
        Overloadable by subclasses."""
        # Baseline
        step_value = flow['value'] * dT
        # Adjust
        requires = flow.get('requires')
        if step_value > 0 and requires:
            if any(_currency not in influx for _currency in requires):
                step_value = 0
            else:
                for _currency in requires:
                    step_value *= influx[_currency]  # Scale flows to requires
        criteria = flow.get('criteria')
        if step_value > 0 and criteria:
            for i, criterion in enumerate(criteria):
                buffer_attr = f'{direction}_{currency}_criteria_{i}_buffer'
                if evaluate_reference(self, criterion):
                    if 'buffer' in criterion and self.attributes[buffer_attr] > 0:
                        self.attributes[buffer_attr] -= dT
                        step_value = 0
                else:
                    step_value = 0
                    if 'buffer' in criterion and self.attributes[buffer_attr] == 0:
                        self.attributes[buffer_attr] = criterion['buffer']
        growth = flow.get('growth')
        if step_value > 0 and growth:
            for mode, params in growth.items():
                growth_attr = f'{direction}_{currency}_{mode}_growth_factor'
                growth_factor = evaluate_growth(self, mode, params)
                self.attributes[growth_attr] = growth_factor
                step_value *= growth_factor
        weighted = flow.get('weighted')
        if step_value > 0 and weighted:
            for field in weighted:
                if field in self.capacity:  # e.g. Biomass
                    weight = self.view(field)[field] / self.active
                elif field in self.properties:  # e.g. 'mass'
                    weight = self.properties[field]['value']
                elif field in self.attributes:  # e.g. 'te_factor'
                    weight = self.attributes[field]
                else:
                    raise ValueError(f'Weighted field {field} not found in '
                                     f'{self.agent_id} storage, properties, or attributes.')
                step_value *= weight
        return step_value
    
    def process_flow(self, dT, direction, currency, flow, influx, target, actual):
        """Update flow state post-exchange. Overloadable by subclasses."""
        available_ratio = round(0 if target == 0 else actual/target, 
                                self.model.floating_point_precision)
        if direction == 'in':
            influx[currency] = available_ratio
        if 'deprive' in flow:
            deprive_attr = f'{direction}_{currency}_deprive'
            if available_ratio < 1:
                deprived_ratio = 1 - available_ratio
                remaining = self.attributes[deprive_attr] - (deprived_ratio * dT)
                self.attributes[deprive_attr] = max(0, remaining)
                if remaining < 0:
                    n_dead = math.ceil(-remaining * self.active)
                    self.kill(f'{self.agent_id} deprived of {currency}', n_dead=n_dead)
            else:
                self.attributes[deprive_attr] = flow['deprive']['value']


    def step(self, dT=1):
        """Update agent for given timedelta."""
        if not self.registered:
            self.register()
        if self.active:
            self.attributes['age'] += dT
            # Check thresholds
            for currency, threshold in self.thresholds.items():
                if evaluate_reference(self, threshold):
                    self.kill(f'{self.agent_id} passed {currency} threshold')

        # Execute flows
        influx = {}  # Which currencies were consumed, and what fraction of baseline
        for direction in ['in', 'out']:
            for currency, flow in self.flows[direction].items():

                # Calculate Target Value
                if self.active and 'value' in flow:
                    target = self.active * self.get_flow_value(dT, direction, currency, flow, influx)
                else:
                    target = 0

                # Process Flow
                remaining = float(target)
                for connection in flow['connections']:
                    agent = self.model.agents[connection]
                    if remaining > 0:
                        multiplier = {'in': -1, 'out': 1}[direction]
                        exchange = agent.increment(currency, multiplier * remaining)
                        exchange_value = sum(exchange.values())
                        remaining -= abs(exchange_value)
                    else:
                        exchange = {k: 0 for k in agent.view(currency).keys()}
                    # NOTE: This must be called regardless of whether the agent is active
                    for _currency, _value in exchange.items():
                        self.records['flows'][direction][_currency][connection].append(abs(_value))
                actual = target - remaining
                # TODO: Handle excess outputs; currently ignored

                # Respond to availability
                if self.active and 'value' in flow:
                    self.process_flow(dT, direction, currency, flow, influx, target, actual)

        # Update remaining records
        self.records['active'].append(self.active)
        for currency in self.capacity:
            self.records['storage'][currency].append(self.storage.get(currency, 0))
        for attribute in self.attributes:
            self.records['attributes'][attribute].append(self.attributes[attribute])
        self.records['cause_of_death'] = self.cause_of_death

    def kill(self, reason, n_dead=None):
        """Kill n_dead agents, or all if n_dead is None. Overloadable by subclasses."""
        n_dead = self.active if n_dead is None else n_dead
        self.active = max(0, self.active - n_dead)
        if self.active <= 0:
            self.cause_of_death = reason

class PlantAgent(Agent):
    """Plant agent with growth and reproduction."""

    default_attributes = {
        # Lifecycle
        'delay_start': 0,
        'grown': False,
        # Growth weights
        'daily_growth_factor': 1,
        'par_factor': 1,
        'growth_rate': 0,
        'cu_factor': 1,
        'te_factor': 1,
    }

    required_kwargs = {
        'flows': {'in': {'co2': 0, 'par': 0},
                  'out': {'biomass': 0, 'inedible_biomass': 0}},
        'capacity': {'biomass': 0},
        'properties': {'photoperiod': {'value': 0},
                       'lifetime': {'value': 0},
                       'par_baseline': {'value': 0}}}

    def __init__(self, *args, attributes=None, **kwargs):
        
        def recursively_check_required_kwargs(given, required):
            for key, value in required.items():
                if key not in given:
                    raise ValueError(f'{key} not found in {given}')
                if isinstance(value, dict):
                    recursively_check_required_kwargs(given[key], value)
        recursively_check_required_kwargs(kwargs, self.required_kwargs)

        attributes = {} if attributes is None else attributes
        attributes = {**self.default_attributes, **attributes}
        super().__init__(*args, attributes=attributes, **kwargs)
        if self.attributes['delay_start'] > 0:
            self.active = 0
        # -- NON_SERIALIZED
        self.daily_growth = []
        self.max_growth = 0

    def register(self, record_initial_state=False):
        super().register(record_initial_state=record_initial_state)
        # Create the `daily_growth` attribute:
        # - Length is equal to the number of steps per day (e.g. 24)
        # - Average value is always equal to 1
        # - `photoperiod` is the number of hours per day of sunlight the plant
        #   requires, which is centered about 12:00 noon. Values outside this
        #   period are 0, and during this period are calculated such that the
        #   mean of all numbers is 1.
        steps_per_day = 24
        photoperiod = self.properties['photoperiod']['value']
        photo_start = (steps_per_day - photoperiod) // 2
        photo_end = photo_start + photoperiod
        photo_rate = steps_per_day / photoperiod
        self.daily_growth = np.zeros(steps_per_day)
        self.daily_growth[photo_start:photo_end] = photo_rate
        
        # Max Growth is used to determine the growth rate (% of ideal)
        lifetime = self.properties['lifetime']['value']
        mean_biomass = self.flows['out']['biomass']['value']
        self.max_growth = mean_biomass * lifetime
        
        # To avoid intra-step fluctuation, we cache the response values in
        # the model each step. TODO: This is a hack, and should be fixed.
        if not hasattr(self.model, '_co2_response_cache'):
            self.model._co2_response_cache = {
                'step_num': 0,
                'cu_factor': 1,
                'te_factor': 1,
            }

    def get_flow_value(self, dT, direction, currency, flow, influx):
        is_grown = self.attributes['grown']
        on_grown = ('criteria' in flow and 
                    any(c['path'] == 'grown' for c in flow['criteria']))
        if ((is_grown and not on_grown) or 
            (not is_grown and on_grown)):
            return 0.
        return super().get_flow_value(dT, direction, currency, flow, influx)
        
    def _calculate_co2_response(self):
        if self.model._co2_response_cache['step_num'] != self.model.step_num:
            ref_agent_name = self.flows['in']['co2']['connections'][0]
            ref_agent = self.model.agents[ref_agent_name]
            ref_atm = ref_agent.view('atmosphere')
            co2_ppm = ref_atm['co2'] / sum(ref_atm.values()) * 1e6
            co2_actual = max(350, min(co2_ppm, 700))
            # CO2 Uptake Factor: Decrease growth if actual < ideal
            if ('carbon_fixation' not in self.properties or 
                self.properties['carbon_fixation']['value'] != 'c3'):
                cu_ratio = 1
            else:
                # Standard equation found in research; gives *increase* in growth for eCO2
                t_mean = 25 # Mean temperature for timestep.
                tt = (163 - t_mean) / (5 - 0.1 * t_mean) # co2 compensation point
                numerator = (co2_actual - tt) * (350 + 2 * tt)
                denominator = (co2_actual + 2 * tt) * (350 - tt)
                cu_ratio = numerator/denominator
                # Invert the above to give *decrease* in growth for less than ideal CO2
                crf_ideal = 1.2426059597016264  # At 700ppm, the above equation gives this value
                cu_ratio = cu_ratio / crf_ideal
            # Transpiration Efficiency Factor: Increase water usage if actual < ideal
            co2_range = [350, 700]
            te_range = [1/1.37, 1]  # Inverse of previously used
            te_factor = np.interp(co2_actual, co2_range, te_range)
            # Cache the values
            self.model._co2_response_cache = {
                'step_num': self.model.step_num,
                'cu_factor': cu_ratio,
                'te_factor': te_factor,
            }
        cached = self.model._co2_response_cache
        return cached['cu_factor'], cached['te_factor']
    
    def step(self, dT=1):
        if not self.registered:
            self.register()
        # --- LIFECYCLE ---
        # Delay start
        if self.attributes['delay_start']:
            super().step(dT)
            self.attributes['delay_start'] -= dT
            if self.attributes['delay_start'] <= 0:
                self.active = self.amount
            return
        # Grown
        if self.attributes['age'] >= self.properties['lifetime']['value']:
            self.attributes['grown'] = True
        
        # --- WEIGHTS ---
        # Daily growth
        hour_of_day = self.model.time.hour
        self.attributes['daily_growth_factor'] = self.daily_growth[hour_of_day]
        # Par Factor
        # 12/22/22: Electric lamps and sunlight work differently.
        # - Lamp.par is multiplied by the lamp amount (to scale kwh consumption)
        # - Sun.par is not, because there's nothing to scale and plants can't
        #   compete over it. Sunlight also can't be incremented.
        # TODO: Implement a grid layout system; add/take par from grid cells
        par_ideal = self.properties['par_baseline']['value'] * self.attributes['daily_growth_factor']
        light_type = self.flows['in']['par']['connections'][0]
        light_agent = self.model.agents[light_type]
        is_electric = ('sun' not in light_type)
        if is_electric:
            par_ideal *= self.active
            exchange = light_agent.increment('par', -par_ideal)
            par_available = abs(sum(exchange.values()))
        else:
            par_available = light_agent.storage['par']
        self.attributes['par_factor'] = (0 if par_ideal == 0 
                                         else min(1, par_available / par_ideal))
        # Growth Rate: *2, because expected to sigmoid so max=2 -> mean=1
        if self.active == 0:
            self.attributes['growth_rate'] = 0
        else:
            stored_biomass = sum(self.view('biomass').values())
            fraction_of_max = stored_biomass / self.active / self.max_growth
            self.attributes['growth_rate'] = fraction_of_max * 2
        # CO2 response
        cu_factor, te_factor = self._calculate_co2_response()
        self.attributes['cu_factor'] = cu_factor
        self.attributes['te_factor'] = te_factor

        super().step(dT)

        # Rproduction
        if self.attributes['grown']:
            self.storage['biomass'] = 0
            if ('reproduce' not in self.properties or 
                not self.properties['reproduce']['value']):
                self.kill(f'{self.agent_id} reached end of life')
            else:
                self.active = self.amount
                self.attributes = {**self.attributes, 
                                   **self.default_attributes, 
                                   'age': 0}

    def kill(self, reason, n_dead=None):
        # Convert dead biomass to inedible biomass
        if n_dead is None:
            n_dead = self.active
        dead_biomass = self.view('biomass')['biomass'] * n_dead / self.active
        if dead_biomass:
            self.storage['biomass'] -= dead_biomass
        ined_bio_str_agent = self.flows['out']['inedible_biomass']['connections'][0]
        ined_bio_str_agent = self.model.agents[ined_bio_str_agent]
        ined_bio_str_agent.increment('inedible_biomass', dead_biomass)
        super().kill(reason, n_dead=n_dead)

class LampAgent(Agent):
    default_attributes = {
        'daily_growth_factor': 1,
        'par_rate': 1,
        'photoperiod': 12,
    }
    default_capacity = {
        'par': 5,
    }
    def __init__(self, *args, attributes=None, capacity=None, **kwargs):
        attributes = {} if attributes is None else attributes
        attributes = {**self.default_attributes, **attributes}
        capacity = {} if capacity is None else capacity
        capacity = {**self.default_capacity, **capacity}
        super().__init__(*args, attributes=attributes, capacity=capacity, **kwargs)
        # -- NON_SERIALIZED
        self.connected_plants = []
        self.daily_growth = []
        self.lamp_configuration = {}

    def _update_lamp_attributes(self):
        # Scale the number of lamps to the number of active plants
        lamp_configuration = {p: self.model.agents[p].active 
                              for p in self.connected_plants}
        if lamp_configuration == self.lamp_configuration:
            return
        self.lamp_configuration = lamp_configuration
        self.active = sum(lamp_configuration.values())
        # Set the photoperiod and par_rate to the max required by any plant
        steps_per_day = 24
        photoperiod = 0
        par_rate = 0
        for plant_id in self.connected_plants:
            plant = self.model.agents[plant_id]
            if plant.active > 0:
                photoperiod = max(photoperiod, plant.properties['photoperiod']['value'])
                par_baseline = plant.properties['par_baseline']['value']                
                par_rate = max(par_rate, par_baseline * steps_per_day / photoperiod)
        self.attributes['photoperiod'] = photoperiod
        self.attributes['par_rate'] = par_rate
        # Update the daily growth
        photo_start = (steps_per_day - photoperiod) // 2
        photo_end = photo_start + photoperiod
        self.daily_growth = np.zeros(steps_per_day)
        self.daily_growth[photo_start:photo_end] = par_rate

    def register(self, record_initial_state=False):
        self.connected_plants = []
        for agent_id, agent in self.model.agents.items():
            if ('par' in agent.flows['in'] and 
                self.agent_id in agent.flows['in']['par']['connections']):
                self.connected_plants.append(agent_id)
        if self.connected_plants:
            self._update_lamp_attributes()
        else:
            self.active = 0
        super().register(record_initial_state)

    def step(self, dT=1):
        if not self.registered:
            self.register()
        self.storage['par'] = 0
        self._update_lamp_attributes()
        hour_of_day = self.model.time.hour
        self.attributes['daily_growth_factor'] = self.daily_growth[hour_of_day]
        super().step(dT)

class SunAgent(Agent):
    default_attributes = {
        'daily_growth_factor': 1,
        'monthly_growth_factor': 1,
    }
    default_capacity = {
        'par': 5,
    }
    hourly_par_fraction = [  # Marino fig. 2a, mean par per hour/day, scaled to mean=1
        0.27330022, 0.06846029, 0.06631662, 0.06631662, 0.48421388, 0.54054486,
        0.5366148, 0.53923484, 0.57853553, 0.96171719, 1.40227785, 1.43849271,
        2.82234256, 3.00993782, 2.82915468, 2.43876788, 1.71301526, 1.01608314,
        0.56958994, 0.54054486, 0.54054486, 0.54316491, 0.54316491, 0.47766377,
    ]
    monthly_par = [  # Maringo fig. 2c & 4, mean hourly par, monthly from Jan91 - Dec95
        0.54950686, 0.63372954, 0.7206446 , 0.92002863, 0.97663421, 0.95983702,
        0.89926235, 0.8211712 , 0.75722611, 0.68654778, 0.57748131, 0.49670542,
        0.53580063, 0.61396126, 0.69077189, 0.86995316, 0.82823278, 0.92457803,
        0.87140854, 0.83036469, 0.79133973, 0.67958089, 0.60519844, 0.49848609,
        0.49649926, 0.57264328, 0.74441785, 0.88318598, 0.93440528, 0.98428221,
        0.91292888, 0.80386089, 0.82544877, 0.67260636, 0.5776829 , 0.5265369,
        0.57708425, 0.6437935 , 0.74417503, 0.87688951, 0.92676186, 0.96316316,
        0.91269064, 0.86154311, 0.75853793, 0.69055809, 0.57138185, 0.51013218,
        0.53643822, 0.63480008, 0.7601048 , 0.87867323, 0.95278919, 1.00872435,
        0.92659387, 0.84716341, 0.81756864, 0.73746165, 0.59808571, 0.55165404,
    ]

    def __init__(self, *args, attributes=None, capacity=None, **kwargs):
        attributes = {} if attributes is None else attributes
        attributes = {**self.default_attributes, **attributes}
        capacity = {} if capacity is None else capacity
        capacity = {**self.default_capacity, **capacity}
        super().__init__(*args, attributes=attributes, capacity=capacity, **kwargs)

    def step(self, dT=1):
        if not self.registered:
            self.register()
        self.storage['par'] = 0
        hour_of_day = self.model.time.hour
        self.attributes['daily_growth_factor'] = self.hourly_par_fraction[hour_of_day]
        reference_year = max(1991, min(1995, self.model.time.year))
        reference_month = self.model.time.month - 1
        reference_i = (reference_year - 1991) * 12 + reference_month
        self.attributes['monthly_growth_factor'] = self.monthly_par[reference_i]
        super().step(dT)

class AtmosphereEqualizerAgent(Agent):
    def __init__(self, *args, **kwargs):
        # -- NON_SERIALIZED
        self.atms = {}
        super().__init__(*args, **kwargs)

    def register(self, record_initial_state=True):
        self.atms = {a: self.model.agents[a] for a in self.flows['in']['atmosphere']['connections']}
        for agent_id in self.atms.keys():
            for direction in ('in', 'out'):
                conns = self.flows[direction]['atmosphere']['connections']
                if agent_id not in conns:
                    conns.append(agent_id)
        super().register(record_initial_state)

    def step(self, dT=1):
        if not self.registered:
            self.register()
        volumes = {}  # agent_type: m3
        current = {}  # agent_type: {atmo_currency: kg}
        total_atm = defaultdict(float)  # atmo_currency: kg
        for agent_id, agent in self.atms.items():
            volumes[agent_id] = agent.properties['volume']['value'] * agent.amount
            current[agent_id] = agent.view(view='atmosphere')
            for currency, amount in current[agent_id].items():
                total_atm[currency] += amount

        total_volume = sum(volumes.values())
        for agent_id, agent in self.atms.items():
            atm_ratio = volumes[agent_id] / total_volume
            targets = {k: v * atm_ratio for k, v in total_atm.items()}
            deltas = {k: v - current[agent_id][k] for k, v in targets.items()}
            for currency, delta in deltas.items():
                if delta != 0:
                    # TODO: Add these to flows records
                    agent.increment(currency, delta)
                inflow = abs(max(0, delta))
                outflow = abs(min(0, delta))
                self.records['flows']['in'][currency][agent_id].append(inflow)
                self.records['flows']['out'][currency][agent_id].append(outflow)
        # super().step(dT)
