import json
from pathlib import Path
from agent_model.util import load_data_file, get_default_currency_data

# -----------------
# UPDATE AGENT DESC
# -----------------
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
            f = {k: v for k, v in f.items() if k != 'required'}  # No longer used
            
            # Special cases
            if 'lamp' in agent_type and currency == 'par':
                f['connections'] = [agent_type]
            
            new_desc['flows'][newDirection][currency] = f

    for char in desc['data']['characteristics']:
        char_type = char['type']
        if char_type.startswith('capacity'):
            _, currency = char['type'].split('_', 1)
            new_desc['capacity'][currency] = char['value']
        elif char_type.startswith('threshold'):
            _, limit, currency = char_type.split('_', 2)
            path = f'in_{currency}_ratio'

            # Special cases
            if 'human' in agent_type and currency == 'co2':
                path = f'out_{currency}_ratio'

            new_desc['thresholds'][currency] = {
                'path': path,
                'limit': '>' if limit == 'upper' else '<',
                'value': char['value'],
                'connections': 'all',  # Require every connection to evaluate true
            }
        elif char_type == 'custom_function':
            continue  # No longer used
        else:
            new_desc['properties'][char_type] = {k: v for k, v in char.items() if k != 'type'}

    return new_desc

# MAKE NEW DATA FILES
new_agent_desc = {}
rename_agents = {'human_agent': 'human'}
for agent_class, agents in default_agent_desc.items():
    for agent_type, desc in agents.items():
        new_name = rename_agents.get(agent_type, agent_type)
        new_agent_desc[new_name] = update_desc(agent_type, desc)
        new_agent_desc[new_name]['agent_class'] = agent_class

# SAVE NEW DATA FILES
with open('data_files/agent_desc.json', 'w') as f:
    json.dump(new_agent_desc, f, indent=4)

# ---------------------
# UPDATE CONFIGURATIONS
# ---------------------
config_names = [
    '1h',
    '1hg_sam',
    '1hrad',
    '4h',
    '4hg',
    'b2_mission1a',
    'b2_mission1b',
    'b2_mission2',
]
currencies = get_default_currency_data()
for config_name in config_names:
    config = load_data_file(f'config_{config_name}.json', data_files)
    config = config['config']  
    reformatted_config = {'agents': {}}
    allowed_kwargs = {'agents', 'currencies', 'termination', 'location',
                      'priorities', 'start_time', 'elapsed_time', 'step_num', 
                      'seed', 'is_terminated', 'termination_reason'}
    ignore_kwargs = {'single_agent', 'total_amount', 'global_entropy', 
                     'minutes_per_step'}
    for k, v in config.items():
        if k not in allowed_kwargs:
            continue
        if k != 'agents':
            reformatted_config[k] = v
            continue
        for agent, agent_data in v.items():
            reformatted_agent = {}
            rename_agents = {'human_agent': 'human'}
            # Special Cases
            for field, value in agent_data.items():
                ignore_fields = {'id', 'total_capacity'}
                static_fields = {'amount'}
                attribute_fields = {'carbonation'}
                if field in ignore_fields:
                    continue
                elif field in static_fields:
                    reformatted_agent[field] = value
                elif field in attribute_fields:
                    if 'attributes' not in reformatted_agent:
                        reformatted_agent['attributes'] = {}
                    reformatted_agent['attributes'][field] = value
                elif field in currencies:
                    if value == 0:
                        continue  # These are now handled by capacity instead
                    if 'storage' not in reformatted_agent:
                        reformatted_agent['storage'] = {}
                    reformatted_agent['storage'][field] = value
                else:
                    raise ValueError(f'Unknown field in agent data: {field}: {value}')
            # Updated lamp system
            if '_lamp' in agent:
                reformatted_agent['prototypes'] = ['lamp']
            elif f'{agent}_lamp' in v:
                reformatted_agent['flows'] = {'in': {'par': {'connections': [f'{agent}_lamp']}}}

            new_name = rename_agents.get(agent, agent)
            reformatted_config['agents'][new_name] = reformatted_agent
    with open(f'data_files/config_{config_name}.json', 'w') as f:
        json.dump(reformatted_config, f, indent=4)