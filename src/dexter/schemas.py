from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Task(BaseModel):
    """Represents a single task in a task list."""
    id: int = Field(..., description="Unique identifier for the task.")
    description: str = Field(..., description="The description of the task.")
    done: bool = Field(False, description="Whether the task is completed.")

class TaskList(BaseModel):
    """Represents a list of tasks."""
    tasks: List[Task] = Field(..., description="The list of tasks.")

class IsDone(BaseModel):
    """Represents the boolean status of a task."""
    done: bool = Field(..., description="Whether the task is done or not.")

class Answer(BaseModel):
    """Represents an answer to the user's query."""
    answer: str = Field(..., description="A comprehensive answer to the user's query, including relevant numbers, data, reasoning, and insights.")

class OptimizedToolArgs(BaseModel):
    """Represents optimized arguments for a tool call."""
    arguments: Dict[str, Any] = Field(..., description="The optimized arguments dictionary for the tool call.")

class ToolCallSpec(BaseModel):
    """Represents a single tool call selection."""
    name: str = Field(..., description="The tool name to invoke.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call.")

class ToolCallList(BaseModel):
    """Represents a list of tool calls to execute."""
    tool_calls: List[ToolCallSpec] = Field(default_factory=list, description="Tool calls to execute.")
