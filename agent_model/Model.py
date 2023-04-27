import random
from copy import deepcopy
import datetime
import numpy as np
from .util import get_default_currency_data, load_data_file, merge_json, recursively_clear_lists
from .Agent import Agent, PlantAgent, LampAgent, SunAgent, AtmosphereEqualizerAgent

DEFAULT_START_TIME = '1991-01-01T00:00:00'
DEFAULT_TIME_UNIT = 'hours'
DEFAULT_LOCATION = 'earth'
FLOATING_POINT_PRECISION = 6
DEFAULT_PRIORITIES = ["structures", "storage", "power_generation", "inhabitants",
                      "eclss", "plants"]

class Model:
    
    floating_point_precision = FLOATING_POINT_PRECISION
    time_unit = DEFAULT_TIME_UNIT

    def __init__(self, termination=None, location=None, priorities=None, 
                 start_time=None, elapsed_time=None, step_num=None, seed=None, 
                 is_terminated=None, termination_reason=None):
        
        # Initialize model data fields
        self.termination = [] if termination is None else termination
        self.location = DEFAULT_LOCATION if location is None else location
        self.priorities = DEFAULT_PRIORITIES if priorities is None else priorities
        self.start_time = datetime.datetime.fromisoformat(DEFAULT_START_TIME if start_time is None else start_time)
        self.elapsed_time = datetime.timedelta(seconds=0 if elapsed_time is None else elapsed_time)
        self.step_num = 0 if step_num is None else step_num
        self.seed = seed if seed is not None else random.getrandbits(32)
        self.is_terminated = None if is_terminated is None else is_terminated
        self.termination_reason = '' if termination_reason is None else termination_reason
        self.agents = {}
        self.currencies = {}

        # NON-SERIALIZABLE
        self.rng = None
        self.scheduler = None
        self.registered = False
        self.records = {'time': [], 'step_num': []}

    def add_agent(self, agent_id, agent):
        if agent_id in self.agents:
            raise ValueError(f'Agent names must be unique ({agent_id})')
        self.agents[agent_id] = agent
    
    def add_currency(self, currency_id, currency_data):
        if currency_id in self.currencies:
            raise ValueError(f'Currency and currency class names must be unique ({currency_id})')
        self.currencies[currency_id] = currency_data

    def register(self, record_initial_state=False):
        self.rng = np.random.RandomState(self.seed)
        self.scheduler = Scheduler(self)
        if record_initial_state:
            self.records['time'].append(self.time.isoformat())
            self.records['step_num'].append(self.step_num)
        for agent in self.agents.values():
            agent.register(record_initial_state)
        self.registered = True

    @classmethod
    def from_config(cls, agents={}, currencies={}, record_initial_state=None, **kwargs):
        # Initialize an empty model
        model = cls(**kwargs)

        # Overwrite generic connections
        replacements = {'habitat': None, 'greenhouse': None}
        for agent_id in agents.keys():
            if 'habitat' in agent_id:
                replacements['habitat'] = agent_id
            elif 'greenhouse' in agent_id:
                replacements['greenhouse'] = agent_id
        def replace_generic_connections(conns):
            """Replace if available, otherwise remove connection"""
            replaced = [replacements.get(c, c) for c in conns]
            pruned = [c for c in replaced if c is not None and c in agents]
            return pruned

        # Merge user agents with default agents
        default_agent_desc = load_data_file('agent_desc.json')
        currencies_in_use = set()
        for agent_id, agent_data in agents.items():

            # Load default agent data and/or prototypes
            prototypes = agent_data.pop('prototypes', [])
            if agent_id in default_agent_desc:
                prototypes.append(agent_id)
            while len(prototypes) > 0:
                [prototype, *prototypes] = prototypes
                if prototype not in default_agent_desc:
                    raise ValueError(f'Agent prototype not found ({prototype})')
                agent_data = merge_json(default_agent_desc[prototype], deepcopy(agent_data))
                if 'prototypes' in agent_data:
                    prototypes += agent_data.pop('prototypes')
            agent_data['agent_id'] = agent_id

            if 'flows' in agent_data:
                for flows in agent_data['flows'].values():
                    for currency, flow_data in flows.items():
                        # Record currencies in use
                        currencies_in_use.add(currency)
                        # Replace generic connections
                        flow_data['connections'] = replace_generic_connections(flow_data['connections'])
            if 'storage' in agent_data:
                for currency in agent_data['storage'].keys():
                    currencies_in_use.add(currency)

            # Determine agent class. TODO: Remove hard-coding somehow?
            if 'agent_class' in agent_data and agent_data['agent_class'] == 'plants':
                build_from_class  = PlantAgent
            elif 'lamp' in agent_id:
                build_from_class = LampAgent
            elif 'sun' in agent_id:
                build_from_class = SunAgent
            elif 'atmosphere_equalizer' in agent_id:
                build_from_class = AtmosphereEqualizerAgent
            else:
                build_from_class = Agent

            agent = build_from_class(model, **agent_data)
            model.add_agent(agent_id, agent)

        # Merge user currencies with default currencies
        currencies = {**get_default_currency_data(), **currencies}
        for currency_id, currency_data in currencies.items():
            # Only add currencies and currency classes with active flows
            if (currency_id in currencies_in_use or 
                (currency_data.get('currency_type') == 'class' and 
                any(c in currencies_in_use for c in currency_data['currencies']))):
                model.add_currency(currency_id, currency_data)

        if record_initial_state is None:
            record_initial_state = model.step_num == 0
        model.register(record_initial_state)
        return model

    @property
    def time(self):
        return self.start_time + self.elapsed_time

    def step(self, dT=1):
        """Advance the model by one step.
        
        Args:
            dT (int, optional): delta time in base time units. Defaults to 1.
        """
        if not self.registered:
            self.register()
        self.step_num += 1
        self.elapsed_time += datetime.timedelta(**{self.time_unit: dT})
        for term in self.termination:
            if term['condition'] == 'time':
                if term['unit'] in ('day', 'days'):
                    reference = self.elapsed_time.days
                elif term['unit'] in ('hour', 'hours'):
                    reference = self.elapsed_time.total_seconds() // 3600
                else:
                    raise ValueError(f'Invalid termination time unit: '
                                     f'{term["unit"]}')
                if reference >= term['value']:
                    self.is_terminated = True
                    self.termination_reason = 'time'
        self.scheduler.step(dT)
        self.records['time'].append(self.time.isoformat())
        self.records['step_num'].append(self.step_num)

    def run(self, dT=1, max_steps=365*24*2):
        """Run the model until termination.
        
        Args:
            dT (int, optional): delta time in base time units. Defaults to 1.
            max_steps (int, optional): maximum number of steps to run. Defaults to 365*24*2.
        """
        while not self.is_terminated and self.step_num < max_steps:
            self.step(dT)

    def get_records(self, static=False, clear_cache=False):
        output = deepcopy(self.records)
        output['agents'] = {name: agent.get_records(static, clear_cache) 
                            for name, agent in self.agents.items()}
        if static:
            output['static'] = {
                'currencies': self.currencies,
                'termination': self.termination,
                'location': self.location,
                'priorities': self.priorities,
                'start_time': self.start_time.isoformat(),
                'seed': self.seed,
            }
        if clear_cache:
            self.records = recursively_clear_lists(self.records)
        return output

    def save(self, records=False):
        output = {
            'agents': {name: agent.save(records) for name, agent in self.agents.items()},
            'currencies': self.currencies,
            'termination': self.termination,
            'location': self.location,
            'priorities': self.priorities,
            'start_time': self.start_time.isoformat(),
            'elapsed_time': self.elapsed_time.total_seconds(),
            'step_num': self.step_num,
            'seed': self.seed,
            'is_terminated': self.is_terminated,
            'termination_reason': self.termination_reason,
        }
        if records:
            output['records'] = deepcopy(self.records)
        return output

class Scheduler:
    def __init__(self, model):
        self.model = model
        self.priorities = [*model.priorities, 'other']
        self.class_agents = {p: [] for p in self.priorities}
        for agent, agent_data in model.agents.items():
            if agent_data.agent_class in self.priorities:
                self.class_agents[agent_data.agent_class].append(agent)
            else:
                self.class_agents['other'].append(agent)

    def step(self, dT):
        for agent_class in self.priorities:
            queue = self.model.rng.permutation(self.class_agents[agent_class])
            for agent in queue:
                self.model.agents[agent].step(dT)
