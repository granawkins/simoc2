import numpy as np
from . import BaseAgent

class LampAgent(BaseAgent):
    """A lamp which provides light to plants

    :ivar list connected_plants: The agent_id's of the plants connected to this lamp
    :ivar list daily_growth: The daily_growth_factor for each hour of the day; mean=1
    :ivar dict lamp_configuration: The number of active lamps for each connected plant

    Custom Attributes:
        * **daily_growth_factor** (float): The factor by which the plant's growth rate is multiplied each day
        * **par_rate** (float): The rate at which the lamp emits photosynthetically active radiation (PAR)
        * **photoperiod** (float): The number of hours per day the lamp is on

    """
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
        """Update the lamp's attributes based on the active connected plants"""
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
        """Find and record connected plants and initialize lamp attributes
        
        Save the agent_id's of all agents which have this lamp in flows.in.par.connections
        """
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
        """Update the lamp attributes based on time of day and active connected plants"""
        if not self.registered:
            self.register()
        self.storage['par'] = 0
        self._update_lamp_attributes()
        hour_of_day = self.model.time.hour
        self.attributes['daily_growth_factor'] = self.daily_growth[hour_of_day]
        super().step(dT)