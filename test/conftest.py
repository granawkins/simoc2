import pytest
from ..simoc_abm.util import get_default_currency_data

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

