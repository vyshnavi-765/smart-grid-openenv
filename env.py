import random
from models import Observation, Action, Reward


class SmartGridEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.max_steps = 24
        self.history = []
        self.reset()

    def reset(self):
        self.time = 0
        self.state_data = self._generate_state()
        self.history = []
        return Observation(**self.state_data)

    def step(self, action: Action):
        demand = self.state_data["demand"]
        supply = self.state_data["supply"]
        renewable = self.state_data["renewable"]
        allocation = action.allocations

        reward = 0

        # 🎯 Demand matching
        for d, a in zip(demand, allocation):
            reward -= abs(d - a) * 0.1

        # ❌ Overuse penalty
        if sum(allocation) > supply:
            reward -= 25

        # ✅ Safe usage bonus
        if sum(allocation) <= supply:
            reward += 5

        # 🌱 Renewable usage bonus
        renewable_used = min(sum(allocation), renewable)
        reward += renewable_used * 0.1

        # 🔥 PRIORITY (Hospital > Industry > Homes)
        priorities = [3, 2, 1]
        for i, (d, a) in enumerate(zip(demand, allocation)):
            if a < d:
                reward -= (d - a) * priorities[i] * 0.3

        # 🚨 Emergency mode
        if supply < sum(demand) * 0.6:
            print("🚨 Emergency Mode Activated")
            reward += 10

        # ⚠️ Blackout detection
        if sum(allocation) < sum(demand) * 0.5:
            print("⚠️ Blackout Risk!")
            reward -= 40

        # 💰 Cost optimization
        cost = sum(allocation) * 0.05
        reward -= cost

        # 🌍 Carbon emission penalty
        non_renewable = sum(allocation) - renewable_used
        carbon_penalty = non_renewable * 0.05
        reward -= carbon_penalty

        # ⚡ Fault simulation
        faults = ["line failure", "overload", "generator issue"]
        if self.task == "hard" and random.random() < 0.3:
            fault = random.choice(faults)
            print(f"⚠️ Fault Occurred: {fault}")
            reward -= 25

        # 🔮 Demand prediction
        predicted = [d + random.randint(-10, 10) for d in demand]
        print("🔮 Predicted Demand:", predicted)

        # Store history (learning signal)
        self.history.append(reward)

        self.time += 1
        done = self.time >= self.max_steps

        self.state_data = self._generate_state()

        return Observation(**self.state_data), Reward(value=reward), done, {}

    def state(self):
        return self.state_data

    def _generate_state(self):
        if self.task == "easy":
            return {
                "demand": [80, 80, 80],
                "supply": 250,
                "renewable": 50,
                "time": self.time
            }

        elif self.task == "medium":
            return {
                "demand": [random.randint(50, 150) for _ in range(3)],
                "supply": random.randint(220, 300),
                "renewable": random.randint(30, 80),
                "time": self.time
            }

        elif self.task == "hard":
            return {
                "demand": [random.randint(100, 200) for _ in range(3)],
                "supply": random.randint(200, 260),
                "renewable": random.randint(10, 50),
                "time": self.time
            }