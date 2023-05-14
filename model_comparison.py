import json, gzip
import functools
import numpy as np
from simoc_abm import AgentModel
import matplotlib.pyplot as plt

def lpe(predictions, targets):
    """Lifetime percentage error"""
    _p = abs(sum(predictions))
    _t = abs(sum(targets))
    return 0 if _t == 0 else (_p-_t)/_t * 100

config_mapping = {
    '1h': 'simoc-simdata-1-human-preset.json.gz',
    '1hrad': 'simoc-simdata-1-human-radish-preset.json.gz',
    '4h': 'simoc-simdata-4-human-preset.json.gz',
    '4hg': 'simoc-simdata-4-human-garden-preset.json.gz',
    '1hg_sam': 'simoc-simdata-sam-1-human-garden-preset.json.gz',
    'b2_mission1a': 'simoc-simdata-b2-mission-1a.json.gz',
    'b2_mission1b': 'simoc-simdata-b2-mission-1b.json.gz',
    'b2_mission2': 'simoc-simdata-b2-mission-2.json.gz',
}

def load_simdata(stem):
    fname = config_mapping[stem]
    with gzip.open(f'test/v1_simdata/{fname}', 'rb') as f:
        data = json.load(f)
    return data

substitute_names = {'human': 'human_agent'}
def compare_agent(stem, agent, i=None, j=None, ncols=None, _cache={}):
    if stem not in _cache:
        with open(f'data_files/config_{stem}.json') as f:
            config = json.load(f)
        model = AgentModel.from_config(**config, record_initial_state=False)
        model.run()
        records = model.get_records()
        simdata = load_simdata(stem)
        references = simdata['data']
        _cache[stem] = dict(actual=records, target=references)
    
    data = _cache[stem]
    actual, target = data['actual'], data['target']
    _actual = actual['agents'][agent]
    _target = target[substitute_names.get(agent, agent)]
    
    # Compile all fields
    plot_fields = {}
    if 'storage' in _actual:
        for k, _a in _actual['storage'].items():
            _t = _target['storage'][k]
            i = 0 if i is None else i
            j = min(len(_a), len(_t)) if j is None else j
            plot_fields[k] = dict(actual=_a[i:j], target=_t[i:j])
    if 'flows' in _actual:
        for direction, flows in _actual['flows'].items():
            for currency, connections in flows.items():
                if currency not in _target['flows'][direction]:
                    continue
                _a = functools.reduce(lambda a, b: a+b, [np.array(v) for v in connections.values()])
                _t = functools.reduce(lambda a, b: a+b, [np.array(v) for v in _target['flows'][direction][currency].values()])
                i = 0 if i is None else i
                j = min(len(_a), len(_t)) if j is None else j
                if sum(_a) == 0 or sum(_t) == 0:
                    continue
                plot_fields[f'{direction}_{currency}'] = dict(actual=_a[i:j], target=_t[i:j])
    if 'attributes' in _actual:
        for field, _a in _actual['attributes'].items():
            if 'growth' in _target and field in _target['growth']:
                _t = _target['growth'][field]
            else:
                continue
            i = 0 if i is None else i
            j = min(len(_a), len(_t)) if j is None else j
            plot_fields[field] = dict(actual=_a[i:j], target=_t[i:j])

    # Plot everything
    ncols = min(2, len(plot_fields)) if ncols is None else ncols
    nrows = -(-len(plot_fields) // ncols)  # Ceiling division
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 8/ncols*nrows))
    fig.tight_layout()
    for index, (field, values) in enumerate(plot_fields.items()):
        i_row = index // ncols
        i_col = index % ncols
        if ncols == 1:
            ax = axs
        elif nrows > 1:
            ax = axs[i_row][i_col]
        else:
            ax = axs[i_col]
        ax.plot(values['actual'], color='red', label='actual')
        ax.plot(values['target'], color='blue', label='target')
        ymax = max(values['actual']) * 1.2
        ymin = min(values['actual']) * 0.8 or 0 - 0.2 * ymax
        ax.set_ylim([ymin, ymax])
        error = round(lpe(values['actual'], values['target']), 3)
        ax.set_title(f'{field} ({error}%)')
        if index == 0:
            ax.legend()