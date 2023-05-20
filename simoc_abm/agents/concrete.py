from . import BaseAgent
from ..util import recursively_check_required_kwargs
import numpy as np

class ConcreteAgent(BaseAgent):
    """One exposed square meter of carbonating concrete

    The processes in this agent are based on kilomoles of carbonation, rather
    than kilograms. Flows are equal to the molar mass of each reactant, e.g.
    carbon dioxide, and stored attributes of carbonation and carbonation_rate 
    are in kmoles. 

    TODO: Describe Reference studies

    :ivar float diffusion_rate: The rate at which carbonation occurs across the ppm gradient.
    :ivar float saturation_when_measured: The percentage of total lifetime carbonation which had occurred at the second reference measurement.
    :ivar list rate_scale: The maximum amount of carbonation possible at the two corresponding ppm levels
    :ivar list ppm_range: Carbon dioxide ppm for the two reference measurements (inside and outside B2)
    :ivar float density: Density of concrete, in kg/m3

    Custom Attributes:
        * **carbonation_rate** (float): Current co2-dependent carbonation rate, in kmoles
        * **carbonation** (float): Total lifetime carbonation, in kmoles
    """
    diffusion_rate = .000018        # Tune manually. Match Table 2 total kmoles
    saturation_when_measured = 0.3  # Tune manually. Lit suggests up to 20yr of carb.
    rate_scale = [12.7, 12.7 + 35 / saturation_when_measured]  # Figure 2 integrals
    ppm_range = [350, 3000]  # External and enclosed ppms
    density = 1.21 / 1000  # Table 2, 'structural concrete', convert grams to kg

    default_attributes = {
        'carbonation_rate': 0,  # Current kmoles/h rate
        'carbonation': 0,       # Cumulative kmoles
    }

    required_kwargs = {
        'flows': {'in': {'co2': 0, 'caoh2': 0},
                  'out': {'caco3': 0, 'moisture': 0}},
        'capacity': {'caoh2': 0, 'caco3': 0, 'moisture': 0}}

    def __init__(self, *args, attributes=None, **kwargs):
        """Set interal storages based on initial carbonation.

        If carbonation is 0, set internal caoh2 to the maximum amount of
        carbonation at the highest ppm level. If carbonation is nonzero, reduce
        internal caoh and increase internal caco3 and moisture by the 
        corresponding amounts.
        """
        recursively_check_required_kwargs(kwargs, self.required_kwargs)
        attributes = {} if attributes is None else attributes
        attributes = {**self.default_attributes, **attributes}
        super().__init__(*args, attributes=attributes, **kwargs)

        # Set internal caoh2 to the maximum amount of carbonation at the highest ppm level
        caoh2_flow = self.flows['in']['caoh2']['value']
        caco3_flow = self.flows['out']['caco3']['value']
        moisture_flow = self.flows['out']['moisture']['value']
        initial_storage = {
            'caoh2': self.calc_max_carbonation(3000) * caoh2_flow * self.active,
            'caco3': 0,     # Byproduct, accumulates internally
            'moisture': 0,  # Byproduct, accumulates internally
        }
        # If carbonation has already occured (Mission 2), update storages accordingly
        carbonation = self.attributes['carbonation']
        if carbonation > 0:
            initial_storage['caoh2'] -= carbonation * caoh2_flow * self.active
            initial_storage['caco3'] += carbonation * caco3_flow * self.active
            initial_storage['moisture'] += carbonation * moisture_flow * self.active
        self.storage = {**self.storage, **initial_storage}

    @classmethod
    def calc_max_carbonation(cls, ppm):
        """Return max kmoles CO2 uptake by structural concrete
        
        :param float ppm: Carbon dioxide concentration in ppm
        :return: Maximum carbonation in kmoles
        """
        saturation_point_kmoles = np.interp(ppm, cls.ppm_range, cls.rate_scale)
        return saturation_point_kmoles * cls.density

    def step(self, dT=1):
        """Update carbonation rate and carbonation, which will be used to weight flows"""
        if not self.registered:
            self.register()
        # Calculate ppm of CO2 in atmosphere
        ref_agent_name = self.flows['in']['co2']['connections'][0]
        ref_agent = self.model.agents[ref_agent_name]
        ref_atm = ref_agent.view('atmosphere')
        ppm = ref_atm['co2'] / sum(ref_atm.values()) * 1e6
        # Calculate carbonation rate
        max_carbonation = self.calc_max_carbonation(ppm)
        gradient = max(0, max_carbonation - self.attributes['carbonation'])
        self.attributes['carbonation_rate'] = gradient * self.diffusion_rate
        self.attributes['carbonation'] += self.attributes['carbonation_rate']
        super().step(dT)
