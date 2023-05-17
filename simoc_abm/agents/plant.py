import numpy as np
from . import BaseAgent
from ..util import recursively_check_required_kwargs

class PlantAgent(BaseAgent):
    """Plant agent with growth and reproduction.
    
    :ivar list daily_growth: The daily_growth_factor for each hour of the day; mean=1
    :ivar float max_growth: The maximum kilograms of biomass an individual plant can produce under ideal conditions

    Custom Attributes:
        * **delay_start** (int): The number of days of simulation before the plant begins growing
        * **grown** (bool): Whether the plant has reached maturity
        * **daily_growth_factor** (float): Growth factor for diurnal growth cycle
        * **par_factor** (float): Growth factor for available:ideal light
        * **cu_factor** (float): Growth factor for available:ideal carbon dioxide
        * **te_factor** (float): Growth factor for how carbon dioxide affects transpiration
        * **growth_rate** (float): The ratio of current to maximum lifetime biomass
    """

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
        """Initialize the daily_growth and max_growth variables"""
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
        """Modulate growth and harvest exchanges based on 'grown' attribute"""
        is_grown = self.attributes['grown']
        on_grown = ('criteria' in flow and 
                    any(c['path'] == 'grown' for c in flow['criteria']))
        if ((is_grown and not on_grown) or 
            (not is_grown and on_grown)):
            return 0.
        return super().get_flow_value(dT, direction, currency, flow, influx)
        
    def _calculate_co2_response(self):
        """Calculate the CO2 response attributes based on connected atmosphere

        To avoid intra-step fluctuation, the response of this function is cached
        each step in the model.
        
        :returns: tuple (cu_factor, te_factor): The CO2 uptake and transpiration efficiency factors"""
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
        """Calculate and update growth factor attributes."""
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
        """Convert dead biomass to inedible biomass when killed."""
        if n_dead is None:
            n_dead = self.active
        dead_biomass = self.view('biomass')['biomass'] * n_dead / self.active
        if dead_biomass:
            self.storage['biomass'] -= dead_biomass
        ined_bio_str_agent = self.flows['out']['inedible_biomass']['connections'][0]
        ined_bio_str_agent = self.model.agents[ined_bio_str_agent]
        ined_bio_str_agent.increment('inedible_biomass', dead_biomass)
        super().kill(reason, n_dead=n_dead)
