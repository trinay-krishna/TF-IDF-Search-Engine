from dataclasses import dataclass


@dataclass
class Query:
    id: int
    content: str
