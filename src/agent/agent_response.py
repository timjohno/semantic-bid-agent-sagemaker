from dataclasses import dataclass, field
from typing import List, Dict, Any
from .agent_message import AgentMessage

@dataclass
class AgentResponse:
    messages: List[AgentMessage]
    thread: Any  # Use specific type if available
    metrics: Dict[str, Any] = field(default_factory=dict)
