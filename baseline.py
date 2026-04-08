class BaselineAgent:
    def act(self, obs):
        demand = obs.demand
        supply = obs.supply

        # 🧠 Priority weights
        weights = [5, 3, 2]

        weighted = [d * w for d, w in zip(demand, weights)]
        total = sum(weighted)

        allocations = [
            int((w / total) * supply) for w in weighted
        ]

        return {"allocations": allocations}