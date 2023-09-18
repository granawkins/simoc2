import os
import sys

# Add the parent directory of the current file to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from simoc_abm.agent_model import AgentModel
from simoc_abm.util import get_default_currency_data, load_data_file


@pytest.fixture
def default_currency_dict():
    currencies = get_default_currency_data()
    categories = {}
    for currency, data in currencies.items():
        category = data['category']
        if category not in categories:
            categories[category] = {'currency_type': 'category', 'currencies': [currency]}
        else:
            categories[category]['currencies'].append(currency)
    return {**currencies, **categories}

_records_cache = {}
def get_records_for_config(stem):
    """Load records from simdata file."""
    if stem not in _records_cache:
        config = load_data_file(f'config_{stem}.json')
        model = AgentModel.from_config(**config)
        model.run()
        records = model.get_records()
        _records_cache[stem] = records
    return _records_cache[stem]

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