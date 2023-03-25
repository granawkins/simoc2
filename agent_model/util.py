import json, copy, operator
from pathlib import Path

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
        conn = agent.flows[direction][currency]['connections'][0]
        ref_agent = agent.model.agents[conn]
        path = '_'.join(elements[1:])
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

