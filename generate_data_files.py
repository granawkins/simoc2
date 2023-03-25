import json
from pathlib import Path
from agent_model.util import load_data_file

# LOAD DEFAULT FILES
# Get path to data_files directory in SIMOC (above this directory)
data_files = Path(__file__).parent.parent / 'simoc' / 'data_files'
# Load default data files
default_agent_desc = load_data_file('agent_desc.json', data_files)
default_agent_conn = load_data_file('agent_conn.json', data_files)

# DEFINE CONVERSION FUNCTION
def update_desc(agent_type, desc):
    """Convert the old-style agent_desc to new-style"""
    new_desc = {
        'amount': 1,
        'storage': {},
        'properties': {},
        'flows': {'in': {}, 'out': {}},
        'capacity': {},
        'thresholds': {},
        'attributes': {},
        'description': desc.get('description', ''),
    }
    for direction in {'input', 'output'}:
        for f in desc['data'][direction]:
            currency = f.pop('type')
            d1, d2 = ('to', 'from') if direction == 'input' else ('from', 'to')
            conns = [c for c in default_agent_conn if c[d1] == f'{agent_type}.{currency}']
            if len(conns) > 1:
                if not any('priority' not in c for c in conns):
                    conns = sorted(conns, key=lambda c: c['priority'])
            f['connections'] = [c[d2].split('.')[0] for c in conns]
            if 'criteria' in f:
                name = f['criteria'].pop('name')
                path = name.split('_')
                if len(path) == 3:
                    path = [path[-1], *path[:-1]]
                f['criteria']['path'] = '_'.join(path)
            newDirection = direction[:-3]
            new_desc['flows'][newDirection][currency] = f
    for char in desc['data']['characteristics']:
        char_type = char['type']
        if char_type.startswith('capacity'):
            _, currency = char['type'].split('_', 1)
            new_desc['capacity'][currency] = char['value']
        elif char_type.startswith('threshold'):
            _, limit, currency = char_type.split('_', 2)
            new_desc['thresholds'][currency] = {
                'path': f'in_{currency}_ratio',
                'limit': '>' if limit == 'upper' else '<',
                'value': char['value'],
            }
        else:
            new_desc['properties'][char_type] = {k: v for k, v in char.items() if k != 'type'}

    return new_desc

# MAKE NEW DATA FILES
new_agent_desc = {}
for agent_class, agents in default_agent_desc.items():
    for agent_type, desc in agents.items():
        new_agent_desc[agent_type] = update_desc(agent_type, desc)
        new_agent_desc[agent_type]['class'] = agent_class

# SAVE NEW DATA FILES
with open('data_files/agent_desc.json', 'w') as f:
    json.dump(new_agent_desc, f, indent=4)
