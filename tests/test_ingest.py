from ragx.config import Settings
from ragx.ingest import ingest


def test_ingest_builds_mirror_and_index(corpus_dir):
    c = ingest(corpus_dir, Settings(model="fake:m"))
    assert c.stats()["files"] == 2
    assert (c.cache_dir / "mirror").exists()
    assert (c.cache_dir / "index" / "df.json").exists()
    # document frequency captured
    assert c.df.get("acme", 0) >= 1
    # mirror files are line-addressable
    mfile = c.mirror_dir / c.files[0].mirror_path
    assert mfile.read_text().strip()


def test_ingest_incremental(corpus_dir):
    s = Settings(model="fake:m")
    c1 = ingest(corpus_dir, s)
    sha_before = {f.source_path: f.sha1 for f in c1.files}
    # touch one file
    (corpus_dir / "company.md").write_text(
        (corpus_dir / "company.md").read_text() + "\nNew line.\n"
    )
    c2 = ingest(corpus_dir, s)
    sha_after = {f.source_path: f.sha1 for f in c2.files}
    assert sha_after["company.md"] != sha_before["company.md"]
    assert sha_after["mars.txt"] == sha_before["mars.txt"]


def test_ingest_prunes_deleted(corpus_dir):
    s = Settings(model="fake:m")
    ingest(corpus_dir, s)
    (corpus_dir / "mars.txt").unlink()
    c = ingest(corpus_dir, s)
    assert "mars.txt" not in {f.source_path for f in c.files}
    assert not (c.mirror_dir / "mars.txt.txt").exists()
