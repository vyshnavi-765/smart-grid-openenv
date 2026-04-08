import os
from openai import OpenAI
from env import SmartGridEnv

# ✅ Environment variables (required)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy_key")  # fallback for local run

# ✅ OpenAI Client (MANDATORY)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ✅ Simple decision using LLM (can fallback safely)
def get_action_from_model(obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"""
You are a smart grid controller.

Demand: {obs.demand}
Supply: {obs.supply}
Renewable: {obs.renewable}

Return ONLY a Python list of 3 integers representing allocations.
"""
                }
            ],
            max_tokens=50
        )

        text = response.choices[0].message.content.strip()

        # Extract list safely
        allocations = eval(text)
        return allocations

    except Exception:
        # ✅ Fallback (ensures no crash)
        total = sum(obs.demand)
        return [int((d / total) * obs.supply) for d in obs.demand]


# ✅ MAIN EXECUTION
def run():
    print("[START] Smart Grid Inference")

    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"[STEP] Task: {task}")

        env = SmartGridEnv(task)
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            allocations = get_action_from_model(obs)

            action = type("Action", (), {"allocations": allocations})
            obs, reward, done, _ = env.step(action)

            total_reward += reward.value
            step_count += 1

            print(f"[STEP] {task} | Step {step_count} | Reward: {reward.value}")

        avg_reward = total_reward / step_count
        score = max(0.0, min(1.0, avg_reward / 500))

        print(f"[STEP] {task} FINAL SCORE: {score}")

    print("[END] Inference Complete")


if __name__ == "__main__":
    run()