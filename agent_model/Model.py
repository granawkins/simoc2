from datetime import datetime
from .util import load_data_file, get_default_agent_data, merge_json
from .Agent import Agent

DEFAULT_START_TIME = '1991-01-01 00:00:00'
FLOATING_POINT_ACCURACY = 6
class Model:
    floating_point_accuracy = FLOATING_POINT_ACCURACY
    step_num = 0
    def __init__(self, agents, termination=None, location=None, minutes_per_step=None,
                 priorities=None, start_time=None, currency_desc=None):

        # Setup model
        self.termination = [] if termination is None else termination
        self.location = 'earth' if location is None else location
        self.minutes_per_step = 60 if minutes_per_step is None else minutes_per_step
        self.priorities = ["structures", "storage", "power_generation", "inhabitants",
                           "eclss", "plants"] if priorities is None else priorities
        self.start_time = datetime.fromisoformat(DEFAULT_START_TIME) if start_time is None else start_time

        # Setup currencies
        self.currency_dict = self.build_currency_dict(currency_desc)

        # Setup agents
        self.agents = {}
        for agent, user_agent_data in agents.items():
            if agent in self.agents:
                raise ValueError(f'Agent names must be unique. Found duplicate: {agent}')
            default_agent_data = get_default_agent_data(agent)
            if default_agent_data is not None:
                agent_data = merge_json(default_agent_data, user_agent_data)
            else:
                agent_data = user_agent_data
            self.agents[agent] = Agent(self, agent, **agent_data)
        for agent in self.agents:
            agent.register()

    @classmethod
    def build_currency_dict(cls, user_currency_desc=None):
        currency_dict = {}
        default_currency_desc = load_data_file('currency_desc.json')
        if user_currency_desc is not None:
            currency_desc = merge_json(default_currency_desc, currency_desc)
        else:
            currency_desc = default_currency_desc
        for currency_class, currencies in currency_desc.items():
            for currency, currency_data in currencies.items():
                currency_dict[currency] = currency_data
                currency_dict[currency]['currency_type'] = 'currency'
                currency_dict[currency]['class'] = currency_class
            currency_dict[currency_class] = {'currency_type': 'class',
                                                  'currencies': list(currencies.keys())}
        return currency_dict

    def step(self, dT=1):
        self.step_num += 1
        for agent in self.agents.values():
            agent.step(dT)
