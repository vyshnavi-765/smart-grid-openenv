from pydantic import BaseModel
from typing import List

class Observation(BaseModel):
    demand: List[int]
    supply: int
    renewable: int
    time: int

class Action(BaseModel):
    allocations: List[int]

class Reward(BaseModel):
    value: float