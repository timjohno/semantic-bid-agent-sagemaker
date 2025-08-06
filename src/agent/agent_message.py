from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class AgentMessage:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
