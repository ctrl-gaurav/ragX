import pytest

from ragx.llm import Usage


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "company.md").write_text(
        "# Acme Robotics\n"
        "Acme Robotics was founded in 2014 by Dana Whitfield in Portland.\n"
        "The flagship product is the Atlas warehouse robot.\n"
        "In 2021 Acme opened a second factory in Austin, Texas.\n"
    )
    (d / "mars.txt").write_text(
        "Perseverance landed in Jezero Crater on February 18, 2021.\n"
        "Ingenuity made the first powered flight on April 19, 2021.\n"
    )
    return d


class FakeLLM:
    """A scripted stand-in for ragx.llm.LLM (no network/GPU)."""

    def __init__(self, spec="fake:model", script=None, **_):
        self.spec = spec
        self.provider, self.model = spec.split(":", 1)
        self.usage = Usage()
        self._script = list(script or [])
        self.calls = []

    def chat(self, messages, max_tokens=None, temperature=None):
        self.usage.calls += 1
        self.calls.append(messages)
        if self._script:
            return self._script.pop(0)
        return "ANSWER: (no script)"


@pytest.fixture
def FakeLLMCls():
    return FakeLLM
