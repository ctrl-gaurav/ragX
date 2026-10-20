"""ragX quickstart over the bundled multi-hop example corpus.

    python examples/quickstart.py                                   # model from your environment
    python examples/quickstart.py --model openai:gpt-4o-mini
    python examples/quickstart.py --variant ragx                    # needs: pip install "pyragx[dense]"
    python examples/quickstart.py --model hf:Qwen/Qwen3.5-4B        # local; needs pyragx[local]

The corpus is a handful of invented articles, so a model cannot answer from memory: every answer
has to be found by following the chain from one document to the next.
"""

import argparse
from pathlib import Path

from ragx import RagX

QUESTIONS = [
    "In which city is the university attended by the author of Harbor Light?",
    "Who founded the university where the author of Harbor Light studied?",
    "Whom did the founder of Ashford University marry?",
    "In which city is the college attended by the author of the novel that won the 2011 Penrose Prize?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="provider:model (defaults to your environment)")
    ap.add_argument("--variant", default=None, help="ragx | ragx-core | ragx-lexical | ...")
    ap.add_argument("--corpus", default=str(Path(__file__).parent / "multihop_corpus"))
    args = ap.parse_args()

    rag = RagX(args.corpus, model=args.model, variant=args.variant)
    print(rag, "\n")
    for q in QUESTIONS:
        ans = rag.ask(q)
        print(f"Q: {q}")
        print(f"A: {ans.text}   (grounded: {ans.grounded})")
        print(f"   sources: {ans.sources}\n")


if __name__ == "__main__":
    main()
