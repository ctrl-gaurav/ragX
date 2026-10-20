"""``ragx`` command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .__about__ import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ragX — agentic retrieval with exact citations. Point it at a folder and ask.",
)
console = Console()

_VARIANT_HELP = (
    "ragx | ragx-core | ragx-lexical | ragx-fused | ragx-rerank | ragx-fast | ragx-think | classic"
)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    corpus: Path = typer.Option(".", "--corpus", "-c", help="File or folder to search."),
    model: str | None = typer.Option(
        None, "--model", "-m", help="provider:model (defaults to env)."
    ),
    variant: str | None = typer.Option(None, "--variant", "-v", help=_VARIANT_HELP),
    steps: int | None = typer.Option(None, "--steps", help="Forager turns (default 10)."),
    think: bool | None = typer.Option(None, "--think/--no-think", help="Thinking mode."),
    live: bool = typer.Option(True, "--live/--no-live", help="Show the live trace."),
    as_json: bool = typer.Option(
        False, "--json", help="Print the answer as JSON (implies --no-live)."
    ),
) -> None:
    """Ask a question over a corpus and print a grounded, cited answer."""
    from .core import RagX
    from .ui import print_answer, run_live

    kw: dict = {}
    if steps is not None:
        kw["steps"] = steps
        kw["max_steps"] = steps
    if think is not None:
        kw["think"] = think
    rag = RagX(corpus, model=model, variant=variant, **kw)
    if as_json:
        import json

        print(json.dumps(rag.ask(question).to_dict(), indent=2, ensure_ascii=False))
    elif live:
        run_live(rag, question, console)
    else:
        print_answer(rag.ask(question), console)


@app.command()
def search(
    query: str = typer.Argument(..., help="The query."),
    corpus: Path = typer.Option(".", "--corpus", "-c", help="File or folder to search."),
    variant: str | None = typer.Option(None, "--variant", "-v", help=_VARIANT_HELP),
    top: int = typer.Option(8, "--top", "-k", help="How many passages."),
) -> None:
    """Retrieval only (no LLM): the top passages for a query."""
    from .core import RagX

    rag = RagX(corpus, variant=variant)
    for i, name in enumerate(rag.search(query, top=top), 1):
        console.print(f"{i:>2}. {name}")


@app.command()
def ingest(
    corpus: Path = typer.Argument(..., help="File or folder to ingest."),
    variant: str | None = typer.Option(None, "--variant", "-v", help=_VARIANT_HELP),
) -> None:
    """Ingest a corpus (text mirror, index, and embeddings for dense variants)."""
    from .core import RagX

    rag = RagX(corpus, variant=variant)
    if rag.variant.route != "classic":
        n = len(rag.store)
        _ = rag.ranker  # builds (and caches) the embeddings for dense variants
    s = rag.stats()
    console.print(f"[green]✓[/] Ingested [bold]{s['corpus']}[/]")
    console.print(f"  files: {s['files']}   lines: {s['lines']}   vocab: {s['vocab']}")
    if rag.variant.route != "classic":
        console.print(f"  passages: {n}   retriever: {s['retriever']}")
    console.print(f"  cache: {rag.corpus.cache_dir}")


@app.command()
def variants() -> None:
    """List the ragX variants."""
    from .variants import DEFAULT_VARIANT, VARIANTS

    t = Table(show_header=True, header_style="bold")
    for col in ("variant", "retriever", "route", "backfill", "think", ""):
        t.add_column(col)
    for v in VARIANTS.values():
        t.add_row(
            v.name + (" (default)" if v.name == DEFAULT_VARIANT else ""),
            v.retriever,
            v.route,
            str(v.backfill),
            "yes" if v.think else "",
            v.summary,
        )
    console.print(t)


@app.command()
def serve(
    corpus: Path = typer.Argument(..., help="File or folder to serve."),
    model: str | None = typer.Option(None, "--model", "-m"),
    variant: str | None = typer.Option(None, "--variant", "-v", help=_VARIANT_HELP),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Serve an OpenAI-compatible /v1/chat/completions endpoint backed by ragX."""
    from .server import serve as _serve

    kw = {"variant": variant} if variant else {}
    _serve(corpus, model=model, host=host, port=port, **kw)


@app.command()
def version() -> None:
    """Print the ragX version."""
    console.print(f"ragx {__version__}")


def _version_cb(value: bool) -> None:
    if value:
        console.print(f"ragx {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    _v: bool = typer.Option(
        False, "--version", callback=_version_cb, is_eager=True, help="Show version and exit."
    ),
) -> None:
    pass


if __name__ == "__main__":
    app()
