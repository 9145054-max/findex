import re
from typing import List
from findex.models import ASTNode, Term, Phrase, And, Or, Not

class QueryParser:
    def __init__(self, query: str):
        self.tokens = self._tokenize_query(query)
        self.pos = 0

    def _tokenize_query(self, query: str) -> List[str]:
        pattern = r'\(|\)|\"[^\"]+\"|NOT|AND|OR|[^\s()\"]+'
        tokens = re.findall(pattern, query, re.IGNORECASE)
        cleaned = []
        for t in tokens:
            if t.startswith('"') and t.endswith('"'):
                cleaned.append(t)
            elif t.upper() in ("AND", "OR", "NOT", "(", ")"):
                cleaned.append(t.upper())
            else:
                cleaned.append(t.lower())
        return cleaned

    def parse(self) -> ASTNode:
        return self._parse_or()

    def _current(self) -> str:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def _consume(self, expected: str = None) -> str:
        token = self._current()
        if expected and token != expected:
            raise ValueError(f"Очікувалося '{expected}', але отримано '{token}'")
        self.pos += 1
        return token

    def _parse_or(self) -> ASTNode:
        node = self._parse_and()
        while self._current() == "OR":
            self._consume("OR")
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> ASTNode:
        node = self._parse_not()
        while self._current() in ("AND", "(", '"') or (self._current() and self._current() not in ("OR", ")")):
            if self._current() == "AND":
                self._consume("AND")
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> ASTNode:
        if self._current() == "NOT":
            self._consume("NOT")
            operand = self._parse_primary()
            return Not(operand)
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        token = self._current()
        if token == "(":
            self._consume("(")
            node = self._parse_or()
            self._consume(")")
            return node
        elif token.startswith('"') and token.endswith('"'):
            self._consume()
            phrase_text = token[1:-1].strip().lower()
            terms = tuple(phrase_text.split())
            return Phrase(terms)
        else:
            term_text = self._consume().lower()
            return Term(term_text)

def parse_query(query: str) -> ASTNode:
    return QueryParser(query).parse()