from ragx.config import Settings
from ragx.ingest import ingest
from ragx.tools import Toolbelt


def _belt(corpus_dir):
    c = ingest(corpus_dir, Settings(model="fake:m"))
    return Toolbelt(corpus=c)


def test_search_finds_term(corpus_dir):
    t = _belt(corpus_dir)
    out = t.search("Dana Whitfield")
    assert "company.md" in out
    assert "company.md" in t.seen_files


def test_rank_returns_relevant_doc(corpus_dir):
    t = _belt(corpus_dir)
    out = t.rank("Perseverance crater Jezero")
    assert "mars.txt" in out


def test_read_records_evidence(corpus_dir):
    t = _belt(corpus_dir)
    out = t.read("company.md", 1, 2)
    assert "Acme Robotics" in out
    assert t.reads and t.reads[0][0] == "company.md"


def test_list_files(corpus_dir):
    t = _belt(corpus_dir)
    out = t.list_files()
    assert "company.md" in out and "mars.txt" in out


def test_read_unknown_file(corpus_dir):
    t = _belt(corpus_dir)
    assert "no such file" in t.read("nope.md", 1, 2)
