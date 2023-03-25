import json, copy, operator
from pathlib import Path
import matplotlib.pyplot as plt

def load_data_file(fname, data_dir=None):
    """Load data file from data directory."""
    if data_dir is None:
        # Get the absolute path of the directory containing the current script
        script_dir = Path(__file__).resolve().parent.parent
        data_dir = script_dir / 'data_files'
    assert data_dir.exists(), f'Data directory does not exist: {data_dir}'
    data_file = data_dir / fname
    assert data_file.exists(), f'Data file does not exist: {data_file}'
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


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
        elements = path.split('_')
        direction, currency = elements[0], elements[1]
        for conn in agent.flows[direction][currency]['connections']:
            ref_agent = agent.model.agents[conn]
            updated_reference = {**reference, 'path': '_'.join(elements[1:])}
            if evaluate_reference(ref_agent, updated_reference):
                return True
        return False
    # Parse field
    if path in ref_agent.attributes:
        target = ref_agent.attributes[path]
    elif path in ref_agent.storage:
        target = ref_agent.storage[path]
    elif path.endswith('_ratio'):
        currency = path[:-6]
        currency_data = ref_agent.model.currency_dict[currency]
        total = sum(ref_agent.view(currency_data['class']).values())
        target = 0 if not total else ref_agent.storage[currency] / total
    # Evaluate
    return operator_dict[limit](target, value)

def get_default_agent_data(agent):
    """Return the relevant dict from default agent_desc.json"""
    default_agent_desc = load_data_file('agent_desc.json')
    if agent in default_agent_desc:
        return copy.deepcopy(default_agent_desc[agent])
    return None

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
