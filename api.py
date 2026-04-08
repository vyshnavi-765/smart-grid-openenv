from fastapi import FastAPI
from env import SmartGridEnv

app = FastAPI()

env = SmartGridEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/reset")
def reset():
    obs = env.reset()
    return {
        "status": "ok",
        "observation": obs.dict()
    }