from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

@dataclass(frozen=True)
class Posting:
    doc_id: int
    tf: int
    positions: Optional[Tuple[int, ...]] = None

@dataclass(frozen=True)
class DocMeta:
    doc_id: int
    filename: str
    length: int

@dataclass(order=True)
class SearchResult:
    score: float
    doc_id: int = field(compare=False)
    filename: str = field(compare=False)
    snippet: str = field(default="", compare=False)

# AST Nodes для рекурсивного парсера запитів
class ASTNode:
    def evaluate(self, index: Any) -> set:
        raise NotImplementedError

    def __and__(self, other: 'ASTNode') -> 'And':
        return And(self, other)

    def __or__(self, other: 'ASTNode') -> 'Or':
        return Or(self, other)

    def __invert__(self) -> 'Not':
        return Not(self)

@dataclass(frozen=True)
class Term(ASTNode):
    term: str

@dataclass(frozen=True)
class Phrase(ASTNode):
    terms: Tuple[str, ...]

@dataclass(frozen=True)
class And(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass(frozen=True)
class Or(ASTNode):
    left: ASTNode
    right: ASTNode

@dataclass(frozen=True)
class Not(ASTNode):
    operand: ASTNode