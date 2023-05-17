import json
from pathlib import Path
from simoc_abm.util import load_data_file, get_default_currency_data

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
            conn_types = [agent_type]
            if 'habitat' in agent_type:
                conn_types.append('habitat')
            elif 'greenhouse' in agent_type:
                conn_types.append('greenhouse')
            conns = [c for c in default_agent_conn 
                     if any(c[d1] == f'{a}.{currency}' for a in conn_types)]
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
                f['criteria'] = [f['criteria']]  # Change criteria to list of dicts
            newDirection = direction[:-3]
            f = {k: v for k, v in f.items() if k != 'required'}  # No longer used
            
            # Special cases
            if 'lamp' in agent_type:
                if currency == 'par':
                    f['connections'] = [agent_type]
                if 'par_baseline' in f['weighted']:
                    f['weighted'] = [w for w in f['weighted'] if w != 'par_baseline']
            if agent_type == 'greenhouse_b2':
                f['connections'] = ['greenhouse_b2']
            
            new_desc['flows'][newDirection][currency] = f
    
    # Special case
    if agent_type == 'b2_sun':
        new_desc['flows'] = {'out': {'par': {
            'value': 1,
            'flow_rate': {'unit': 'mol', 'time': 'hour'},
            'weighted': ['daily_growth_factor', 'monthly_growth_factor'],
            'connections': ['b2_sun'],
        }}}

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

    categories = list(new_desc.keys())
    for cat in categories:
        if not new_desc[cat]:
            del new_desc[cat]

    return new_desc

# MAKE NEW DATA FILES
new_agent_desc = {}
rename_agents = {'human_agent': 'human'}
for agent_class, agents in default_agent_desc.items():
    for agent_type, desc in agents.items():
        new_name = rename_agents.get(agent_type, agent_type)
        new_agent_desc[new_name] = update_desc(agent_type, desc)
        new_agent_desc[new_name]['agent_class'] = agent_class

# Manual changes to ECLSS components
for dir, flows in new_agent_desc['multifiltration_purifier_post_treatment']['flows'].items():
    if dir == 'in':
        flows['treated']['criteria'] = [{
            "limit": ">=",
            "value": flows['treated']['value'],
            "path": "in_treated",
        }]
        flows['kwh']['requires'] = ['treated']

for currency, flow in new_agent_desc['co2_reduction_sabatier']['flows']['in'].items():
    if currency == 'h2':
        flow['criteria'].append({
            "limit": ">=",
            "value": flow['value'],
            "path": "in_h2",
        })
    elif currency == 'co2':
        del flow['criteria']  # Covered by weight
    else:
        flow['requires'].append('h2')

new_agent_desc['co2_removal_SAWD']['flows']['out']['co2']['requires'].append('kwh')

for dir, flows in new_agent_desc['solid_waste_aerobic_bioreactor']['flows'].items():
    if dir == 'in':
        flows['feces']['criteria'] = [{
            "limit": ">=",
            "value": flows['feces']['value'],
            "path": "in_feces",
        }]
        o2 = flows.pop('o2')
        flows['kwh']['requires'] = ['feces']
        o2['requires'] = ['feces', 'kwh']
        flows['o2'] = o2
    if dir == 'out':
        for f in flows.values():
            f['requires'].append('kwh')

for dir, flows in new_agent_desc['urine_recycling_processor_VCD']['flows'].items():
    if dir == 'in':
        flows['urine']['criteria'] = [{
            "limit": ">=",
            "value": flows['urine']['value'],
            "path": "in_urine",
        }]
        flows['kwh']['requires'] = ['urine']
    if dir == 'out':
        for f in flows.values():
            f['requires'].append('kwh')

for dir, flows in new_agent_desc['dehumidifier']['flows'].items():
    if dir == 'in':
        flows['h2o']['criteria'].append({
            "limit": ">=",
            "value": flows['h2o']['value'],
            "path": "in_h2o",
        })
    elif dir == 'out':
        flows['treated']['requires'].append('h2o')



# SAVE NEW DATA FILES
with open('simoc_abm/data_files/agent_desc.json', 'w') as f:
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
                reformatted_agent['flows'] = {'out': {'par': {'connections': [agent]}}}
            elif f'{agent}_lamp' in v:
                reformatted_agent['flows'] = {'in': {'par': {'connections': [f'{agent}_lamp']}}}
            
            # Add custom b2 elements
            new_name = rename_agents.get(agent, agent)
            if 'b2' in config_name:
                if new_agent_desc[new_name].get('agent_class') == 'plants':
                    reformatted_agent['properties'] = {'density_factor': {'value': 0.5}}
                    if 'b2_mission2' in config_name:
                        reformatted_agent['properties']['crop_management_factor'] = {'value': 1.5}
                elif new_name == 'human':
                    food_flow = new_agent_desc['human']['flows']['in']['food']['value']
                    reformatted_agent['flows'] = {'in': {'food': {'value': food_flow / 2}}}
                elif new_name == 'co2_removal_SAWD':
                    criteria = new_agent_desc['co2_removal_SAWD']['flows']['in']['co2']['criteria']
                    criteria[0]['value'] = 0.0025
                    reformatted_agent['flows'] = {'in': {'co2': {'criteria': criteria}}}

            reformatted_config['agents'][new_name] = reformatted_agent
    with open(f'simoc_abm/data_files/config_{config_name}.json', 'w') as f:
        json.dump(reformatted_config, f, indent=4)