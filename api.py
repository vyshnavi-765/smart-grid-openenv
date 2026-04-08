from fastapi import FastAPI
from env import SmartGridEnv

app = FastAPI()
env = SmartGridEnv()

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.dict()}

@app.post("/step")
def step(action: dict):
    obs, reward, done, _ = env.step(type("A", (), action))
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done
    }

@app.get("/state")
def state():
    return env.state()