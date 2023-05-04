from collections import defaultdict
from . import BaseAgent

class AtmosphereEqualizerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        # -- NON_SERIALIZED
        self.atms = {}
        super().__init__(*args, **kwargs)

    def register(self, record_initial_state=True):
        self.atms = {a: self.model.agents[a] for a in self.flows['in']['atmosphere']['connections']}
        for agent_id in self.atms.keys():
            for direction in ('in', 'out'):
                conns = self.flows[direction]['atmosphere']['connections']
                if agent_id not in conns:
                    conns.append(agent_id)
        super().register(record_initial_state)

    def step(self, dT=1):
        if not self.registered:
            self.register()
        volumes = {}  # agent_type: m3
        current = {}  # agent_type: {atmo_currency: kg}
        total_atm = defaultdict(float)  # atmo_currency: kg
        for agent_id, agent in self.atms.items():
            volumes[agent_id] = agent.properties['volume']['value'] * agent.amount
            current[agent_id] = agent.view(view='atmosphere')
            for currency, amount in current[agent_id].items():
                total_atm[currency] += amount

        total_volume = sum(volumes.values())
        for agent_id, agent in self.atms.items():
            atm_ratio = volumes[agent_id] / total_volume
            targets = {k: v * atm_ratio for k, v in total_atm.items()}
            deltas = {k: v - current[agent_id][k] for k, v in targets.items()}
            for currency, delta in deltas.items():
                if delta != 0:
                    # TODO: Add these to flows records
                    agent.increment(currency, delta)
                inflow = abs(max(0, delta))
                outflow = abs(min(0, delta))
                self.records['flows']['in'][currency][agent_id].append(inflow)
                self.records['flows']['out'][currency][agent_id].append(outflow)
                