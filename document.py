from dataclasses import dataclass


@dataclass
class Document:
    id: int
    title: str
    author: str
    content: str
