import random
from copy import deepcopy
import datetime
import numpy as np
from .util import get_default_currency_data, get_default_agent_data, merge_json, recursively_clear_lists
from .Agent import Agent

DEFAULT_START_TIME = '1991-01-01 00:00:00'
DEFAULT_TIME_UNIT = 'hours'
DEFAULT_LOCATION = 'earth'
FLOATING_POINT_ACCURACY = 6
DEFAULT_PRIORITIES = ["structures", "storage", "power_generation", "inhabitants",
                      "eclss", "plants"]

class Model:
    
    floating_point_accuracy = FLOATING_POINT_ACCURACY
    time_unit = DEFAULT_TIME_UNIT

    def __init__(self, agents, currencies, termination=None, location=None, 
                 priorities=None, start_time=None, elapsed_time=None, 
                 step_num=None, seed=None):
        
        # Initialize model data fields
        self.termination = [] if termination is None else termination
        self.location = DEFAULT_LOCATION if location is None else location
        self.priorities = DEFAULT_PRIORITIES if priorities is None else priorities
        self.start_time = datetime.datetime.fromisoformat(DEFAULT_START_TIME if start_time is None else start_time)
        self.elapsed_time = datetime.timedelta(seconds=0 if elapsed_time is None else elapsed_time)
        self.step_num = 0 if step_num is None else step_num
        self.seed = seed if seed is not None else random.getrandbits(32)
        
        # Initialize agents
        self.agents = {}
        for agent_id, agent_data in agents.items():
            if agent_id in self.agents:
                raise ValueError(f'Agent names must be unique ({agent_id})')
            # TODO: Add a 'prototype' arg to specify default agent other than agent_id
            default_agent_data = get_default_agent_data(agent_id)
            if default_agent_data is not None:
                # Merge user agent data with default agent data, if available
                agent_data = merge_json(default_agent_data, deepcopy(agent_data))
            # TODO: Select agent class based on agent_data
            self.agents[agent_id] = Agent(self, agent_id, agent_data)
        
        # Initialize currencies | TODO: Only include currencies used by agents
        self.currencies = get_default_currency_data()
        for currency_id, currency_data in currencies.items():
            if currency_id in self.currencies:
                raise ValueError(f'Currency and currency class names must be unique ({currency_id})')
            self.currencies[currency_id] = currency_data

        # NON-SERIALIZABLE
        self.rng = np.random.RandomState(self.seed)
        self.scheduler = Scheduler(self)
        self.records = {'time': [], 'step_num': []}
        if self.step_num == 0:
            self.records['time'].append(self.time)
            self.records['step_num'].append(self.step_num)
            for agent in self.agents.values():
                agent.register(record_initial_state=True)
        else:
            for agent in self.agents.values():
                agent.register(record_initial_state=False)

    @property
    def time(self):
        return self.start_time + self.elapsed_time

    def step(self, dT=1):
        """Advance the model by one step.
        
        Args:
            dT (int, optional): delta time in base time units. Defaults to 1.
        """
        self.step_num += 1
        self.elapsed_time += datetime.timedelta(**{self.time_unit: dT})
        self.scheduler.step(dT)
        # TODO: Evaluate termination conditions
        self.records['time'].append(self.time)
        self.records['step_num'].append(self.step_num)

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
            'seed': self.seed
        }
        if records:
            output['records'] = deepcopy(self.records)

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
