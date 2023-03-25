import math, copy
from .util import evaluate_reference

def _copy(obj):
    return copy.deepcopy(obj)

class Agent:
    # ------------- SETUP  ------------- #
    def __init__(self, agent_id, model, amount=1, description=None, agent_class=None,
                 properties=None, capacity=None, thresholds=None, flows=None,
                 active=None, storage=None, attributes=None):
        # -- STATIC
        self.agent_id = agent_id                                        # A unique string
        self.amount = 1 if amount is None else amount                   # Starting/Maximum number alive
        self.description = '' if description is None else description   # Plaintext description
        self.agent_class = '' if agent_class is None else agent_class   # Agent class name
        self.properties = {} if properties is None else _copy(properties)      # Static vars, 'volume'
        self.capacity = {} if capacity is None else _copy(capacity)            # Max storage per currency
        self.thresholds = {} if thresholds is None else _copy(thresholds)      # Env. conditions to die
        self.flows = {'in': {}, 'out': {}} if flows is None else _copy(flows)  # Exchanges w/ other agents
        # -- DYNAMIC
        self.cause_of_death = None
        self.active = amount if active is None else _copy(active)              # Current number alive
        self.storage = {} if storage is None else _copy(storage)               # Currencies stored
        self.attributes = {} if attributes is None else _copy(attributes)      # Dynamic vars, 'te_factor'
        # -- NON-SERIALIZED
        self.registered = False                                         # Agent has been registered
        self.records = {}                                               # Container for step records
        self.model = model                                              # AgentModel instance

    def register(self):
        """Check and setup agent after all agents have been added to Model"""
        if self.registered:
            return
        # Initialize flow attributes and records, check connections
        flow_records = {'in': {}, 'out': {}}
        for direction, flows in self.flows.items():
            for currency, flow in flows.items():
                self.register_flow(direction, currency, flow)
                flow_records[direction][currency] = {c: [0] for c in flow['connections']}
        # Initialize records skeleton
        self.records = {
            'step_num': [self.model.step_num],
            'active': [self.active],
            'cause_of_death': self.cause_of_death,
            'storage': {currency: [self.storage[currency]] for currency in self.storage},
            'attributes': {attr: [self.attributes[attr]] for attr in self.attributes},
            'flows': flow_records,
        }
        self.registered = True

    def register_flow(self, direction, currency, flow):
        """Check flow, setup attributes and records. Overloadable by subclasses."""
        if 'criteria' in flow and 'buffer' in flow['criteria']:
            buffer_attr = f'{direction}_{currency}_criteria_buffer'
            if buffer_attr not in self.attributes:
                self.attributes[buffer_attr] = flow['criteria']['buffer']
        if 'deprive' in flow:
            deprive_attr = f'{direction}_{currency}_deprive'
            if deprive_attr not in self.attributes:
                self.attributes[deprive_attr] = flow['deprive']['value']
        for agent in flow['connections']:
            if agent not in self.model.agents:
                raise ValueError(f'Agent {agent} not registered')
            if currency not in self.model.agents[agent].capacity:
                raise ValueError(f'Agent {agent} does not store {currency}')

    # ------------- STEP ------------- #
    def view(self, view):
        """Return a dict with storage amount for single currency or all of a class"""
        currency_type = self.model.currency_dict[view]['currency_type']
        if currency_type == 'currency':
            if view not in self.storage:
                return {view: 0}
            return {view: self.storage[view]}
        elif currency_type == 'class':
            class_currencies = self.model.currency_dict[view]['currencies']
            return {currency: self.storage[currency]
                    for currency in class_currencies
                    if currency in self.storage}

    def increment(self, currency, value):
        """Increment currency in storage as available, return actual receipt"""
        if value < 0:  # Can be currency or currency_class
            available = self.view(currency)
            total_available = sum(available.values())
            if total_available == 0:
                return {currency: 0}
            actual = -min(-value, total_available)
            increment = {currency: actual * stored/total_available
                         for currency, stored in available.items()}
            for currency, amount in increment.items():
                self.storage[currency] += amount
            return increment
        elif value > 0:  # Can only be currency
            if self.model.currency_dict[currency]['currency_type'] != 'currency':
                raise ValueError(f'Cannot increment currency by class ({currency})')
            if currency not in self.storage:
                return {currency: 0}
            remaining_capacity = self.capacity[currency] - self.storage[currency]
            actual = min(value, remaining_capacity)
            self.storage[currency] += actual
            return {currency: actual}

    def get_step_value(self, dT, direction, currency, flow, influx):
        """Return the baseline step value. Overloadable by subclasses."""
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
                    self.attributes[buffer_attr] -= 1
                    step_value = 0
            else:
                if 'buffer' in criteria and self.attributes[buffer_attr] == 0:
                    self.attributes[buffer_attr] = criteria['buffer']
                step_value = 0
        weighted = flow.get('weighted')
        if step_value > 0 and weighted:
            for field in weighted:
                if field in self.storage:  # e.g. Biomass
                    weight = self.storage[field] / self.active
                elif field in self.properties:  # e.g. 'grown'
                    weight = self.properties[field]['value']
                elif field in self.attributes:  # e.g. 'te_factor'
                    weight = self.attributes[field]
                else:
                    raise ValueError(f'Weighted field {weight} not found in '
                                     f'{self.agent_id} storage, properties, or attributes.')
                if field == 'growth_rate':
                    weight *= 2  # For an un-skewed sigmoid curve, max height is 2x mean
                    # TODO: Move to PlantAgent._get_step_value
                step_value *= weight
        return step_value

    def step(self, dT=1):
        """Update agent for given timedelta. Overloadable by subclasses."""
        if not self.registered:
            self.register()
        if self.active:
            # Check thresholds
            for currency, threshold in self.thresholds.items():
                if evaluate_reference(self, threshold):
                    self.kill(f'{self.agent_id} passed {currency} threshold')

        # Execute flows
        influx = {}  # Which currencies were consumed, and what fraction of baseline
        for direction in {'in', 'out'}:
            if direction not in self.flows:
                continue
            for currency, flow in self.flows[direction].items():

                # Calculate Target Value
                if self.active and 'value' in flow:
                    target_value = self.get_step_value(dT, direction, currency, flow, influx)
                else:
                    target_value = 0

                # Process Flow
                remaining = target_value * self.active
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

                # Respond to availability
                available_ratio = 0 if target_value == 0 else 1 - (remaining / target_value)
                if direction == 'in':
                    influx[currency] = available_ratio
                if 'deprive' in flow:
                    deprive_attr = f'{direction}_{currency}_deprive'
                    if available_ratio < 1:
                        remaining = self.attributes[deprive_attr] - available_ratio
                        self.attributes[deprive_attr] = max(0, remaining)
                        if remaining < 0:
                            n_dead = math.ceil(-remaining * self.active)
                            self.kill(f'{self.agent_id} deprived of {currency}', n_dead=n_dead)
                    else:
                        self.attributes[deprive_attr] = flow['deprive']['value']

        # Update remaining records
        self.records['step_num'].append(self.model.step_num)
        self.records['active'].append(self.active)
        for currency in self.storage:
            self.records['storage'][currency].append(self.storage[currency])
        for attribute in self.attributes:
            self.records['attributes'][attribute].append(self.attributes[attribute])
        self.records['cause_of_death'] = self.cause_of_death

    def kill(self, reason, n_dead=None):
        """Kill n_dead agents, or all if n_dead is None. Overloadable by subclasses."""
        n_dead = n_dead or self.active
        self.active = max(0, self.active - n_dead)
        if self.active <= 0:
            self.cause_of_death = reason

    # ------------- INSPECT ------------- #
    def get_records(self, static_fields=False, clear_cache=False):
        """Return records dict and optionally clear cache"""
        output = copy.deepcopy(self.records)
        if static_fields:
            output['agent_id'] = self.agent_id
            output['amount'] = self.amount
            output['description'] = self.description
            output['agent_class'] = self.agent_class
            output['properties'] = copy.deepcopy(self.properties)
            output['capacity'] = copy.deepcopy(self.capacity)
            output['thresholds'] = copy.deepcopy(self.thresholds)
            output['flows'] = copy.deepcopy(self.flows)
        if clear_cache:
            def recursively_clear_lists(r):
                if isinstance(r, dict):
                    return {k: recursively_clear_lists(v) for k, v in r.items()}
                elif isinstance(r, list):
                    return []
            self.records = recursively_clear_lists(self.records)
        return output

    def export(self):
        """Return a serializable copy of the agent"""
        return {
            'agent_id': self.agent_id,
            'amount': self.amount,
            'description': self.description,
            'agent_class': self.agent_class,
            'properties': self.properties,
            'capacity': self.capacity,
            'thresholds': self.thresholds,
            'flows': self.flows,
            'active': self.active,
            'storage': self.storage,
            'attributes': self.attributes,
        }
