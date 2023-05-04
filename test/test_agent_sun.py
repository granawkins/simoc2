import datetime
from ..agent_model.agents import SunAgent

class MockModel:
    time = datetime.datetime(1900, 1, 1, 0)

class TestAgentSun:
    def test_agent_sun_init(self):
        """Initialize attributes properly"""
        model = MockModel()
        sun = SunAgent(model, 'sun')
        for k, v in SunAgent.default_attributes.items():
            assert sun.attributes[k] == v

    def test_agent_sun_step(self):
        # Midnight, January 1 1900 (years before '91 use '91)
        model = MockModel()
        sun = SunAgent(model, 'sun')
        sun.step()
        expected_daily_growth = SunAgent.hourly_par_fraction[0]
        expected_monthly_growth = SunAgent.monthly_par[0]
        assert sun.attributes['daily_growth_factor'] == expected_daily_growth
        assert sun.attributes['monthly_growth_factor'] == expected_monthly_growth

        # Noon, April 5, 1993
        model.time = datetime.datetime(1993, 4, 5, 12)
        sun.step()
        expected_daily_growth = SunAgent.hourly_par_fraction[12]
        reference_i = 12 * 2 + 3
        expected_monthly_growth = SunAgent.monthly_par[reference_i]
        assert sun.attributes['daily_growth_factor'] == expected_daily_growth
        assert sun.attributes['monthly_growth_factor'] == expected_monthly_growth

        # 5pm, December 31 2001 (years after '95 use '95)
        model.time = datetime.datetime(2001, 12, 31, 17)
        sun.step()
        expected_daily_growth = SunAgent.hourly_par_fraction[17]
        expected_monthly_growth = SunAgent.monthly_par[-1]
        assert sun.attributes['daily_growth_factor'] == expected_daily_growth
        assert sun.attributes['monthly_growth_factor'] == expected_monthly_growth