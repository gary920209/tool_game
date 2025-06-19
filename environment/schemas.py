from pydantic import BaseModel, Field
from typing import Literal, List


class ToolUseAction(BaseModel):
    analysis: str
    toolname: Literal["obj1", "obj2", "obj3"] = Field(..., description="Name of the tool to place")
    position: List[float] = Field(..., min_items=2, max_items=2, description="2D position where the tool should be placed")

class ToolUseActionCheck(BaseModel):
    analysis: str
    correct: bool
    toolname: Literal["obj1", "obj2", "obj3"]
    position: List[float]

class ToolUseVideoAction(BaseModel):
    analysis: str
    correct: bool
    toolname: Literal["obj1", "obj2", "obj3"]
    position: List[float]
