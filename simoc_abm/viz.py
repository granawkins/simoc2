import matplotlib.pyplot as plt
from .util import parse_data

def plot_agent(data, agent, category, exclude=[], include=[], i=None, j=None, ax=None):
    """Helper function for plotting model data

    Plotting function which takes model-exported data, agent name,
    one of (flows, growth, storage, deprive), exclude, and i:j
    """
    i = i if i is not None else 0
    j = j if j is not None else data['step_num'][-1]
    ax = ax if ax is not None else plt
    if category == 'flows':
        path = [agent, 'flows', '*', '*', 'SUM', f'{i}:{j}']
        flows = parse_data(data['agents'], path)
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
        parsed = parse_data(data['agents'], path)
        for field, values in parsed.items():
            if field in exclude or (include and field not in include):
                continue
            ax.plot(range(i, j), values, label=field)
    ax.legend()
    return ax
