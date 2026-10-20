"""Real-time "watch it think" rendering for the terminal (Rich).

Falls back to plain line logging when not attached to a TTY.
"""

from __future__ import annotations

from collections.abc import Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from ..types import Answer, Event

_ICON = {
    "plan": ("🧭", "bold cyan"),
    "thought": ("💭", "dim"),
    "search": ("🔎", "yellow"),
    "fuzzy": ("〰️", "yellow"),
    "rank": ("📊", "magenta"),
    "list": ("📁", "blue"),
    "toc": ("🗂️", "blue"),
    "read": ("📖", "green"),
    "peek": ("👀", "green"),
    "semantic": ("🧠", "magenta"),
    "hop": ("🔗", "cyan"),
    "backfill": ("➕", "blue"),
    "verify": ("✓", "bold green"),
    "answer": ("💡", "bold white"),
    "error": ("⚠️", "red"),
    "observation": ("·", "dim"),
}


def _line(ev: Event) -> Text:
    icon, style = _ICON.get(ev.kind, ("·", "white"))
    t = Text()
    t.append(f"{icon} ", style=style)
    t.append(f"{ev.kind:<8} ", style=style)
    t.append(ev.text[:100], style="default" if ev.kind != "thought" else "dim italic")
    return t


def run_live(rag, question: str, console: Console | None = None) -> Answer:
    """Stream the foraging loop into a live-updating panel; return the final Answer."""
    console = console or Console()
    if not console.is_terminal:
        return _run_plain(rag, question, console)

    events: list[Event] = []
    answer: Answer | None = None
    spinner = Spinner("dots", text=Text(" foraging…", style="dim"))

    def render() -> Panel:
        body = Group(*[_line(e) for e in events[-18:]], spinner)
        title = f"[bold]rag[/][bold #818cf8]X[/] · {question[:70]}"
        return Panel(body, title=title, border_style="cyan")

    gen: Iterator[Event] = rag.ask_stream(question)
    with Live(render(), console=console, refresh_per_second=12) as live:
        try:
            while True:
                ev = next(gen)
                events.append(ev)
                live.update(render())
        except StopIteration as stop:
            answer = stop.value

    print_answer(answer, console)
    return answer  # type: ignore[return-value]


def _run_plain(rag, question: str, console: Console) -> Answer:
    gen = rag.ask_stream(question)
    answer: Answer | None = None
    try:
        while True:
            ev = next(gen)
            console.print(_line(ev))
    except StopIteration as stop:
        answer = stop.value
    print_answer(answer, console)
    return answer  # type: ignore[return-value]


def print_answer(answer: Answer, console: Console | None = None) -> None:
    console = console or Console()
    console.print()
    console.print(Panel(Text(answer.text, style="bold"), title="Answer", border_style="green"))
    if answer.citations:
        cites = Text()
        for c in answer.citations:
            cites.append(f"  • {c}\n", style="cyan")
            if c.quote:
                cites.append(f"      {c.quote}\n", style="dim italic")
        console.print(Panel(cites, title="Citations", border_style="blue"))
    if not answer.grounded:
        console.print("[yellow]⚠ answer is not fully grounded in the corpus[/]")
    _print_usage(answer, console)


def _print_usage(answer: Answer, console: Console) -> None:
    """A compact one-line footer: LLM calls, cache hits, and latency."""
    usage = answer.usage or {}
    if not usage:
        return
    per_model = {k: v for k, v in usage.items() if isinstance(v, dict)}
    calls = sum(u.get("calls", 0) for u in per_model.values())
    hits = sum(u.get("cache_hits", 0) for u in per_model.values())
    latency = sum(u.get("latency_s", 0.0) for u in per_model.values())
    models = " + ".join(per_model.keys())
    parts = [f"{calls} LLM call{'s' if calls != 1 else ''}", f"{latency:.1f}s"]
    if hits:
        parts.append(f"{hits} cached")
    cost = usage.get("total_cost_usd")
    if cost:
        parts.append(f"~${cost:.4f}")
    foot = Text(f"  ⚡ {' · '.join(parts)}  ·  {models}", style="dim")
    console.print(foot)


__all__ = ["run_live", "print_answer"]
