from fastapi import FastAPI
from env import SmartGridEnv

app = FastAPI()
env = SmartGridEnv()

@app.get("/")
def home():
    return {"status": "running"}

# ✅ RESET
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.dict()
    }

# ✅ STEP
@app.post("/step")
def step(action: dict):
    result = env.step(action)
    return result

# ✅ STATE
@app.get("/state")
def state():
    return env.state()