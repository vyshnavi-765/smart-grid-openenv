from pydantic import BaseModel

class Observation(BaseModel):
    demand: float
    supply: float

class Action(BaseModel):
    allocation: float

class Reward(BaseModel):
    value: float