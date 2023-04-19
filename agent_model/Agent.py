import math
from copy import deepcopy
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
            capacity (dict): Max storage per currency
            thresholds (dict): Env. conditions to die
            flows (dict): Exchanges w/ other agents
            cause_of_death (str): Reason for death
            active (int): Current number alive
            storage (dict): Currencies stored
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
            elif self.storage[currency] > self.capacity[currency]:
                raise ValueError(f'Agent {self.agent_id} has more storage '
                                 f'for {currency} than capacity.')
        # Initialize flow attributes and records, check connections
        flow_records = {'in': {}, 'out': {}}
        for direction, flows in self.flows.items():
            for currency, flow in flows.items():
                self.register_flow(direction, currency, flow)
                record = {c: [] if not record_initial_state else [0]
                          for c in flow['connections']}
                flow_records[direction][currency] = record
        # Initialize records skeleton
        self.records = {
            'active': [] if not record_initial_state else [self.active],
            'cause_of_death': self.cause_of_death,
            'storage': {currency: [] if not record_initial_state 
                        else [self.storage[currency]] 
                        for currency in self.storage},
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
        if 'criteria' in flow and 'buffer' in flow['criteria']:
            buffer_attr = f'{direction}_{currency}_criteria_buffer'
            self.attributes[buffer_attr] = flow['criteria']['buffer']
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
                    from IPython import embed; embed()
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
            return {currency: self.storage[currency]
                    for currency in class_currencies
                    if currency in self.storage}
        
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
        if value < 0:  # Can be currency or currency_class
            available = self.view(currency)
            total_available = sum(available.values())
            if total_available == 0:
                return {currency: 0}
            actual = -min(-value, total_available)
            increment = {currency: round(actual * stored/total_available, self.model.floating_point_accuracy)
                         for currency, stored in available.items()}
            for currency, amount in increment.items():
                self.storage[currency] += amount
            return increment
        elif value > 0:  # Can only be currency
            if self.model.currencies[currency]['currency_type'] != 'currency':
                raise ValueError(f'Cannot increment agent by currency class ({currency})')
            if currency not in self.capacity:
                raise ValueError(f'Agent does not store {currency}')
            if currency not in self.storage:
                self.storage[currency] = 0
            remaining_capacity = self.capacity[currency] - self.storage[currency]
            actual = round(min(value, remaining_capacity), self.model.floating_point_accuracy)
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
            buffer_attr = f'{direction}_{currency}_criteria_buffer'
            if evaluate_reference(self, criteria):
                if 'buffer' in criteria and self.attributes[buffer_attr] > 0:
                    self.attributes[buffer_attr] -= dT
                    step_value = 0
            else:
                step_value = 0
                if 'buffer' in criteria and self.attributes[buffer_attr] == 0:
                    self.attributes[buffer_attr] = criteria['buffer']
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
                if field in self.storage:  # e.g. Biomass
                    weight = self.storage[field] / self.active
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
        available_ratio = 0 if target == 0 else actual/target
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
                    if remaining > 0:
                        agent = self.model.agents[connection]
                        multiplier = {'in': -1, 'out': 1}[direction]
                        exchange = agent.increment(currency, multiplier * remaining)
                        exchange_value = sum(exchange.values())
                        remaining -= abs(exchange_value)
                    else:
                        exchange_value = 0

                    # NOTE: This should be called regardless of whether the agent is active
                    self.records['flows'][direction][currency][connection].append(exchange_value)
                actual = target - remaining
                # TODO: Handle excess outputs; currently ignored

                # Respond to availability
                self.process_flow(dT, direction, currency, flow, influx, target, actual)

        # Update remaining records
        self.records['active'].append(self.active)
        for currency in self.storage:
            self.records['storage'][currency].append(self.storage[currency])
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
    default_attributes = {
        # Lifecycle
        'delay_start': 0,
        'age': 0,
        'grown': False,
        # Growth weights
        'daily_growth_factor': 1,
        'par_factor': 1,
        'growth_rate': 0,
        'cu_factor': 1,
        'te_factor': 1,
    }

    """Plant agent with growth and reproduction."""
    def __init__(self, *args, attributes=None, **kwargs):
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
        step_value = super().get_flow_value(dT, direction, currency, flow, influx)
        if self.attributes['grown']:
            if 'criteria' in flow and flow['criteria']['path'] != 'grown':
                return 0.
        return step_value
    
    def _calculate_co2_response(self):
        if self.model._co2_response_cache['step_num'] != self.model.step_num:
            ref_agent_name = self.flows['in']['co2']['conections'][0]
            ref_agent = self.model.agents[ref_agent_name]
            ref_atm = ref_agent.view('atmosphere')
            co2_ppm = ref_atm['co2'] / sum(ref_atm.values()) * 1e6
            co2_actual = max(350, min(co2_ppm, 700))
            # CO2 Uptake Factor: Decrease growth if actual < ideal
            if self.properties.get('carbon_fixation') == 'c4':
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

        # --- LIFECYCLE ---
        # Delay start
        if self.attributes['delay_start']:
            self.attributes['delay_start'] -= dT
            if self.attributes['delay_start'] <= 0:
                self.active = self.amount
        # Rproduction
        if self.attributes['grown']:
            if not self.properties['reproduce']['value']:
                self.kill(f'{self.agent_id} reached end of life')
            else:
                self.active = self.amount
                self.attributes = {**self.attributes, **self.default_attributes}
        if self.attributes['age'] >= self.properties['lifetime']['value']:
            self.attributes['grown'] = True
        
        # --- WEIGHTS ---
        # Daily growth
        hour_of_day = self.model.time.hour
        self.attributes['daily_growth_factor'] = self.daily_growth[hour_of_day]
        # Par Factor
        light_agent = self.flows['in']['par']['connections'][0]
        light_agent = self.model.agents[light_agent]
        self.attributes['par_factor'] = light_agent.attributes['par_factor']
        # Growth Rate
        self.attributes['growth_rate'] = self.storage['biomass'] / self.max_growth
        # CO2 response
        cu_factor, te_factor = self._calculate_co2_response()
        self.attributes['cu_factor'] = cu_factor
        self.attributes['te_factor'] = te_factor

        super().step(dT)

    def kill(self, reason, n_dead=None):
        # Convert dead biomass to inedible biomass
        if n_dead is None:
            n_dead = self.active
        dead_biomass = self.storage['biomass'] * n_dead / self.active
        self.storage['biomass'] -= dead_biomass
        ined_bio_str_agent = self.flows['out']['inedible_biomass']['connections'][0]
        ined_bio_str_agent = self.model.agents[ined_bio_str_agent]
        ined_bio_str_agent.increment('inedible_biomass', dead_biomass)
        super().kill(reason, n_dead=n_dead)

class LightAgent(Agent):
    default_attributes = {
        'par_factor': 1,
    }
    def __init__(self, *args, attributes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_attributes = {
            'par_factor': 1,
        }
        self.attributes = {**self.default_attributes, **attributes}
        self.active = self.amount