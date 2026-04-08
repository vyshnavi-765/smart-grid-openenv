from fastapi import FastAPI
from pydantic import BaseModel
from env import SmartGridEnv

app = FastAPI()

env = SmartGridEnv()

# Home route
@app.get("/")
def home():
    return {"status": "running"}

# ✅ RESET (POST — must accept empty body)
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "status": "ok",
        "observation": obs.dict() if hasattr(obs, "dict") else obs
    }

# ✅ STEP (POST)
class ActionInput(BaseModel):
    allocations: list[int]

@app.post("/step")
def step(action: ActionInput):
    obs, reward, done, _ = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done
    }

# ✅ STATE (GET)
@app.get("/state")
def state():
    return env.state()