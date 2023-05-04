from . import BaseAgent

class SunAgent(BaseAgent):
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
