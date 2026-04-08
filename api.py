from fastapi import FastAPI
from env import SmartGridEnv

app = FastAPI()

env = SmartGridEnv()

# Home route
@app.get("/")
def home():
    return {"status": "running"}

# ✅ FIXED RESET (POST)
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "status": "ok",
        "observation": obs.dict()
    }