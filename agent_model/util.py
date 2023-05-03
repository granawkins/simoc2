import json, copy, operator, math
from pathlib import Path
import numpy as np

# DATA HANDLING

def load_data_file(fname, data_dir=None):
    """Load data file from data directory."""
    if data_dir is None:
        # Get the absolute path of the directory containing the current script
        script_dir = Path(__file__).resolve().parent.parent
        data_dir = script_dir / 'data_files'
    else:
        data_dir = Path(data_dir)
    assert data_dir.exists(), f'Data directory does not exist: {data_dir}'
    data_file = data_dir / fname
    assert data_file.exists(), f'Data file does not exist: {data_file}'
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data

def get_default_agent_data(agent):
    """Return the relevant dict from default agent_desc.json"""
    default_agent_desc = load_data_file('agent_desc.json')
    if agent in default_agent_desc:
        return copy.deepcopy(default_agent_desc[agent])
    return None

def get_default_currency_data():    
    """Load default currency_desc.json and convert to new structure"""
    currencies = {}
    currency_desc = load_data_file('currency_desc.json')
    for currency_class, class_currencies in currency_desc.items():
        for currency, currency_data in class_currencies.items():
            currencies[currency] = currency_data
            currencies[currency]['currency_type'] = 'currency'
            currencies[currency]['class'] = currency_class
        currencies[currency_class] = {'currency_type': 'class',
                                     'currencies': list(class_currencies.keys())}
    return currencies

def merge_json(default, to_merge):
    """Merge two objects of arbitrary depth/elements"""
    if isinstance(to_merge, dict):
        for k, v in to_merge.items():
            default[k] = v if k not in default else merge_json(default[k], v)
        return default
    elif isinstance(to_merge, list):
        if len(to_merge) == 0:
            return default
        elif isinstance(to_merge[0], dict):
            return to_merge
        elif isinstance(to_merge[0], str):
            return list(set(default).union(set(to_merge)))
        else:
            raise ValueError(f'Cannot merge lists of type {type(to_merge[0])}')
    elif isinstance(to_merge, (str, int, float, bool)):
        return to_merge

def recursively_clear_lists(r):
    if isinstance(r, (int, float, str)):
        return r
    elif isinstance(r, dict):
        return {k: recursively_clear_lists(v) for k, v in r.items()}
    elif isinstance(r, list):
        return []

# LIMIT FUNCTIONS (THRESHOLD AND CRITERIA)
operator_dict = {
    '>': operator.gt, '<': operator.lt, 
    '>=': operator.ge, '<=': operator.le,
    '=': operator.eq, '!=': operator.ne,
}
def evaluate_reference(agent, reference):
    """Evaluate a reference dict and return a boolean

    Supported path elements:
        - 'grown': from attributes
        - 'in_co2_ratio': ratio (0-1) of co2 to total class (atmosphere) from first connected agent
    """
    path, limit, value = reference['path'], reference['limit'], reference['value']
    ref_agent = agent
    # Parse connected agent
    if path.startswith('in_') or path.startswith('out_'):
        # Evaluate connections by direction/currency
        direction, remainder = path.split('_', 1)
        if path.endswith('_ratio'):
            currency = '_'.join(remainder.split('_')[:-1])
        else:
            currency = remainder
        conns = agent.flows[direction][currency]['connections']
        updated_reference = {**reference, 'path': remainder}
        results = (evaluate_reference(agent.model.agents[c], updated_reference) for c in conns)
        # Return group eval connections
        if 'connections' in reference and reference['connections'] == 'all':
            return all(results)
        return any(results)
    # Parse field
    if path in ref_agent.attributes:
        target = ref_agent.attributes[path]
    elif path in ref_agent.storage:
        target = ref_agent.storage[path]
    elif path.endswith('_ratio'):
        currency = path[:-6]
        currency_data = ref_agent.model.currencies[currency]
        total = sum(ref_agent.view(currency_data['class']).values())
        target = 0 if not total else ref_agent.storage[currency] / total
    # Evaluate
    return operator_dict[limit](
        round(target, agent.model.floating_point_precision),
        round(value, agent.model.floating_point_precision))

# GROWTH FUNCTIONS

def pdf(_x, std, cache={}):
    """Return y-value of normal distribution at x-value for mean=0"""
    if (_x, std) not in cache:
        numerator = math.exp(-1 * (_x ** 2) / (2 * (std ** 2)))
        denominator = math.sqrt(2 * math.pi) * std
        cache[(_x, std)] = numerator / denominator
    return cache[(_x, std)]

def pdf_mean(std, center, n_samples, cache={}):
    """Calculate the mean y-value of the pdf"""
    if (std, center) not in cache:
        x_vals = [i/n_samples for i in range(n_samples)]
        y_vals = [pdf((x - center) / std, std) for x in x_vals]
        cache[(std, center)] = sum(y_vals) / n_samples
    return cache[(std, center)]

def sample_norm(rate, std=math.pi/10, center=0.5, n_samples=100):
    """Return y-value of normal distribution at x-value, such mean(y) = 1
    
    Arguments:
        rate: x-value to sample at
        std: standard deviation of normal distribution
        center: x-value to center the distribution at
        n_samples: number of samples to use for mean calculation
    """
    if any(v < 0 or v > 1 for v in (rate, std, center)):
        raise ValueError('rate, std, and center must be between 0 and 1.')
    # Shift x-value to center at 0
    x = (rate - center) / std
    # Calculate y-value
    y = pdf(x, std)
    # Normalize y-value to mean of 1
    y_mean = pdf_mean(std, center, n_samples)
    return y / y_mean

def sample_clipped_norm(rate, factor=2, **kwargs):
    """Return y-value of normal distribution at x-value, clipped at center
    
    From sample_norm, multiply all values by factor, clip at original max,
    then scale to max=1.

    Arguments:
        rate: x-value to sample at
        factor: factor to multiply the normal distribution by
    """
    norm_value = sample_norm(rate, **kwargs)  # Get the norm value
    center = kwargs.get('center', 0.5)
    y_max = sample_norm(center, **kwargs)     # Get max value for that curve
    norm_value *= factor                      # Scale value by factor
    clip_value = min(norm_value, y_max)       # Clip at original max
    return clip_value / y_max                 # Scale to max=1

def sample_sigmoid(rate, min_value=0, max_value=1, steepness=1, center=0.5):
    """return the sigmoid value"""
    x = steepness * 20 * (rate - center)
    y = 1 / (1 + np.exp(-x))
    scaled = y * (max_value - min_value)
    shifted = scaled + min_value
    return shifted

def sample_switch(rate, min_value=0, max_value=1, center=0.5, duration=0.5):
    """return the switch value"""
    if rate > center - duration / 2 and rate < center + duration / 2:
        return max_value
    return min_value

def evaluate_growth(agent, mode, params):
    if mode == 'daily':
        rate = agent.model.time.hour / 24
    elif mode == 'lifetime':
        rate = agent.attributes['age'] / agent.properties['lifetime']['value']
    growth_func = {
        'norm': sample_norm,
        'sigmoid': sample_sigmoid,
        'clipped': sample_clipped_norm,
        'switch': sample_switch
    }[params['type']]
    return growth_func(rate, **{k: v for k, v in params.items() if k != 'type'})

# WORKING WITH OUTPUTS

def parse_data(data, path):
    """Recursive function to extract data at path from arbitrary object"""
    if not data and data != 0:
        return None
    elif len(path) == 0:
        return 0 if data is None else data
    # Shift the first element of path, past on the rest of the path
    index, *remainder = path
    # LISTS
    if isinstance(data, list):
        # All Items
        if index == '*':
            parsed = [parse_data(d, remainder) for d in data]
            return [d for d in parsed if d is not None]
        # Single index
        elif isinstance(index, int):
            return parse_data(data[index], remainder)
        # Range i:j (string)
        else:
            start, end = [int(i) for i in index.split(':')]
            return [parse_data(d, remainder) for d in data[start:end]]
    # DICTS
    elif isinstance(data, dict):
        # All items, either a dict ('*') or a number ('SUM')
        if index in {'*', 'SUM'}:
            parsed = [parse_data(d, remainder) for d in data.values()]
            output = {k: v for k, v in zip(data.keys(), parsed) if v or v == 0}
            if len(output) == 0:
                return None
            elif index == '*':
                return output
            else:
                if isinstance(next(iter(output.values())), list):
                    return [sum(x) for x in zip(*output.values())]
                else:
                    return sum(output.values())
        # Single Key
        elif index in data:
            return parse_data(data[index], remainder)
        # Comma-separated list of keys. Return an object with all.
        elif isinstance(index, str):
            indices = [i.strip() for i in index.split(',') if i in data]
            parsed = [parse_data(data[i], remainder) for i in indices]
            output = {k: v for k, v in zip(indices, parsed) if v or v == 0}
            return output if len(output) > 0 else None
