import os
import sys

# Add the parent directory of the current file to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from simoc_abm.util import get_default_currency_data

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

