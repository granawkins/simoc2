import matplotlib.pyplot as plt
from .util import parse_data

def plot_agent(data, agent, category, exclude=[], include=[], i=None, j=None, ax=None):
    """Helper function for plotting model data

    :param data: Data object returned by `AgentModel.get_records`
    :param str agent: agent_id of the agent to plot
    :param str category: category of data to plot ('active', 'flows', 'storage', 'attributes')
    :param list exclude: list of fields to exclude from the plot
    :param list include: list of fields to include in the plot (overrides exclude)
    :param int i: first step to plot (default: 0)
    :param int j: last step to plot (default: last step)
    :param ax: matplotlib axes object to plot on (default: plt)

    :returns: matplotlib axes object
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
