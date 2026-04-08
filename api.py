from api import FastAPI
from env import SmartGridEnv

app = FastAPI()

env = SmartGridEnv()

# ✅ Root
@app.get("/")
def home():
    return {"status": "ok"}

# ✅ VERY IMPORTANT: MUST accept POST with/without body
@app.post("/reset")
async def reset():
    obs = env.reset()

    # Convert safely
    try:
        obs_data = obs.dict()
    except:
        obs_data = obs

    return {
        "observation": obs_data
    }

# ✅ STEP
@app.post("/step")
async def step(action: dict):
    obs, reward, done, _ = env.step(type("A", (), action))

    try:
        obs_data = obs.dict()
    except:
        obs_data = obs

    return {
        "observation": obs_data,
        "reward": float(reward.value),
        "done": bool(done)
    }

# ✅ STATE
@app.get("/state")
def state():
    return env.state()