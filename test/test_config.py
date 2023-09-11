from .conftest import get_records_for_config

class TestConfigs:
    def test_config_1h(self):
        stem = '1h'
        records = get_records_for_config(stem)
        assert len(records['step_num']) == 10 * 24 + 1
        assert records['agents']['human']['active'][-1] == 1
        
    def test_config_1hrad(self):
        stem = '1hrad'
        records = get_records_for_config(stem)
        assert len(records['step_num']) == 30 * 24 + 1
        assert records['agents']['human']['active'][-1] == 1

    def test_config_4h(self):
        stem = '4h'
        records = get_records_for_config(stem)
        assert records['agents']['human']['active'][-1] == 4

    def test_config_4hg(self):
        stem = '4hg'
        records = get_records_for_config(stem)
        assert records['agents']['human']['active'][-1] == 4

    def test_config_1hg_sam(self):
        stem = '1hg_sam'
        records = get_records_for_config(stem)
        assert records['agents']['human']['active'][-1] == 1

    def test_config_b2_mission1a(self):
        stem = 'b2_mission1a'
        records = get_records_for_config(stem)
        assert records['agents']['human']['active'][-1] == 8

    def test_config_b2_mission1b(self):
        stem = 'b2_mission1b'
        records = get_records_for_config(stem)
        assert records['agents']['human']['active'][-1] == 8

    def test_config_b2_mission2(self):
        stem = 'b2_mission2'
        records = get_records_for_config(stem)
        assert records['agents']['human']['active'][-1] == 8
