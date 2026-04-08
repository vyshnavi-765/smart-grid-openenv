from env import SmartGridEnv
from baseline import BaselineAgent

def evaluate():
    agent = BaselineAgent()

    for task in ["easy", "medium", "hard"]:
        env = SmartGridEnv(task)
        total = 0

        for _ in range(5):
            obs = env.reset()
            done = False

            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(
                    type("A", (), action)
                )
                total += reward.value

        print(task, "score:", total/5)

if __name__ == "__main__":
    evaluate()