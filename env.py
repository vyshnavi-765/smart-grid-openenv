def reset(self):
    self.current_step = 0
    return self._get_observation()

def step(self, action):
    self.current_step += 1

    obs = self._get_observation()
    reward = self._compute_reward(action)
    done = self.current_step >= 10

    return {
        "observation": obs.dict(),
        "reward": float(reward),
        "done": done
    }

def state(self):
    return {
        "step": self.current_step
    }