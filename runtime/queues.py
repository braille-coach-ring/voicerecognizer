from dataclasses import dataclass, field
from queue import Queue
from typing import Any


@dataclass
class RuntimeQueues:
    audio: Queue[Any] = field(default_factory=Queue)
    speech: Queue[Any] = field(default_factory=Queue)
    text: Queue[str] = field(default_factory=Queue)
