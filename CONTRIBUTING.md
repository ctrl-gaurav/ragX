# Contributing to ragX

Thanks for your interest in ragX. Bug reports, documentation fixes, new loaders, providers and
rankers are all welcome.

## Development setup

```bash
git clone https://github.com/ctrl-gaurav/ragX && cd ragX
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dense,server,dev]"
```

## Checks

```bash
ruff check ragx tests          # lint
ruff format --check ragx tests # formatting
pytest                         # tests: no network, GPU or API key needed
python -m build && twine check dist/*   # the package builds and its README renders
```

The website and documentation build with Node 20:

```bash
(cd website && npm ci && npm run build)
(cd docs && npm ci && npm run build)
```

## Guidelines

- Keep the core install light: heavy dependencies (`torch`, `sentence-transformers`, server
  frameworks) belong in extras.
- Add tests for new behaviour. Tests must run offline; `tests/conftest.py` has a scripted `FakeLLM`
  and `tests/test_method.py` shows fake embedders and cross-encoders.
- The prompts in `ragx/method/prompts.py` and the defaults in `ragx/config.py` are the method as
  published. Changes to them change results; propose them in an issue first.
- Update `docs/` and the README when the public API changes.
- One logical change per pull request, with a descriptive title.

## Where things live

| To add | Edit |
|---|---|
| a file format | `ragx/ingest/loaders.py` |
| a ranker | `ragx/rankers.py` and `make_ranker` |
| a provider | `ragx/llm/__init__.py` (an OpenAI-compatible provider is a one-line entry) |
| a variant | `ragx/variants.py` |

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
