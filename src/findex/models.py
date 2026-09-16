from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass(frozen=True)
class Posting:
    doc_id: int
    tf: int  # Term Frequency
    positions: Optional[Tuple[int, ...]] = None

@dataclass(frozen=True)
class DocMeta:
    doc_id: int
    filename: str
    length: int
