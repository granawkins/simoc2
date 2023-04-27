import pytest
from ..agent_model.Model import Model
from ..agent_model.util import load_data_file

config_names = [
    '1h',
    '1hrad',
    '4h',
    '4hg',
    '1hg_sam',
    'b2_mission1a',
    'b2_mission1b',
    'b2_mission2',
]

class TestConfigs:
    def test_config_1h(self):
        config = load_data_file('config_1h.json')
        model = Model.from_config(**config)
        model.run()
        assert model.elapsed_time.days == 10
        human = model.agents['human']
        assert human.active == 1

    def test_config_1hrad(self):
        config = load_data_file('config_1hrad.json')
        model = Model.from_config(**config)
        model.run()
        human = model.agents['human']
        assert human.active == 1

    def test_config_4h(self):
        config = load_data_file('config_4h.json')
        model = Model.from_config(**config)
        model.run()
        human = model.agents['human']
        assert human.active == 4

    def test_config_4hg(self):
        config = load_data_file('config_4hg.json')
        model = Model.from_config(**config)
        model.run()
        human = model.agents['human']
        assert human.active == 4

    def test_config_1hg_sam(self):
        config = load_data_file('config_1hg_sam.json')
        model = Model.from_config(**config)
        model.run()
        human = model.agents['human']
        assert human.active == 1