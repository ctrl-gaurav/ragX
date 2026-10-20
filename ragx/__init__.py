"""ragX — agentic retrieval that names what the question leaves unnamed.

Point it at a folder; an agent ranks and reads passages, a chain of sub-questions resolves
the entities the question does not name, and one compose call answers with exact
``path:line`` citations.

Quickstart
----------
>>> from ragx import RagX
>>> rag = RagX("./docs")                          # ragx-lexical: BM25, no embedding index
>>> answer = rag.ask("Who founded the company that acquired Acme?")
>>> print(answer.text, answer.sources)
>>> rag = RagX("./docs", variant="ragx")          # the full method (pip install "pyragx[dense]")
"""

from . import pricing
from .__about__ import __version__
from .config import Settings
from .core import RagX
from .types import Answer, Citation, Event
from .variants import VARIANTS, get_variant

__all__ = [
    "VARIANTS",
    "Answer",
    "Citation",
    "Event",
    "RagX",
    "Settings",
    "__version__",
    "get_variant",
    "pricing",
]
