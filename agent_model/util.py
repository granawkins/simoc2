import json, copy, operator, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
        return list(set(default).union(set(to_merge)))
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

operator_dict = {'>': operator.gt, '<': operator.lt, '=': operator.eq}
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
        round(target, agent.model.floating_point_accuracy),
        round(value, agent.model.floating_point_accuracy))

# GROWTH FUNCTIONS

def pdf(_x, std, cache={}):
    """return Gaussian Probability Distribution"""
    if (_x, std) not in cache:
        numerator = math.exp(-1 * (_x ** 2) / (2 * (std ** 2)))
        denominator = math.sqrt(2 * math.pi) * std
        cache[(_x, std)] = numerator / denominator
    return cache[(_x, std)]

def sample_norm(rate, min_value=0, max_value=1, std=math.pi/10, center=0.5):
    """return the normalized sigmoid value"""
    # Shift x-value to center at 0
    x = (rate - center) / std
    y = pdf(x, std)
    # Shift y-value to min/max range
    norm_factor = pdf(0, std)  # First, set max=1
    normalized = y / norm_factor
    scaled = normalized * (max_value - min_value)
    shifted = scaled + min_value
    return shifted

def sample_clipped_norm(rate, factor=2, **kwargs):
    """return the clipped normalized sigmoid value"""
    norm_value = sample_norm(rate, **kwargs)
    norm_value *= factor
    return min(max(norm_value, 0), factor)

def sample_sigmoid(rate, min_value=0, max_value=1, steepness=1, center=0.5):
    """return the sigmoid value"""
    x = steepness * 20 * (rate - center)
    y = 1 / (1 + np.exp(-x))
    scaled = y * (max_value - min_value)
    shifted = scaled + min_value
    return shifted

def evaluate_growth(agent, mode, params):
    if mode == 'daily':
        rate = agent.model.time.hour / 24
    elif mode == 'lifetime':
        rate = agent.attributes['age'] / agent.properties['lifetime']['value']
    growth_func = {
        'norm': sample_norm,
        'sigmoid': sample_sigmoid,
        'clipped': sample_clipped_norm,
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

def plot_agent(data, agent, category, exclude=[], include=[], i=None, j=None, ax=None):
    """Helper function for plotting model data

    Plotting function which takes model-exported data, agent name,
    one of (flows, growth, storage, deprive), exclude, and i:j
    """
    i = i if i is not None else 0
    j = j if j is not None else data['step_num']
    ax = ax if ax is not None else plt
    if category == 'flows':
        path = [agent, 'flows', '*', '*', 'SUM', f'{i}:{j}']
        flows = parse_data(data, path)
        for direction in ('in', 'out'):
            if direction not in flows:
                continue
            for currency, values in flows[direction].items():
                label = f'{direction}_{currency}'
                if currency in exclude or label in exclude:
                    continue
                ax.plot(range(i, j), values, label=label)
    elif category in {'storage', 'attributes'}:
        path = [agent, category, '*', f'{i}:{j}']
        parsed = parse_data(data, path)
        for field, values in parsed.items():
            if field in exclude or (include and field not in include):
                continue
            ax.plot(range(i, j), values, label=field)
    ax.legend()
    return ax
