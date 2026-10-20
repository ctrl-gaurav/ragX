# FAQ

### Do I need a vector database?

No. The default variant, `ragx-lexical`, ranks with BM25 and needs no embeddings. The dense variants
keep their embeddings in a NumPy file next to the text mirror and search them exactly in memory.

### Do I need a GPU?

Not for `ragx-lexical` with a hosted model. The dense variants embed the corpus once, which is much
faster on a GPU but works on a CPU; `ragx-rerank` runs a cross-encoder on every `RANK`, which a GPU
speeds up considerably. Local models (`hf:`, vLLM) need whatever their size requires.

### Which variant should I use?

Start with `ragx-lexical`. Move to `ragx-rerank` for the best accuracy or `ragx` for the most
complete pipeline when you can run an embedder. See [Variants](./guide/variants).

### Does it work on single-hop questions?

Yes. The forager often answers in two or three turns, and the slot chain then has a single hop. For
questions that one lookup answers, a single-shot pipeline is cheaper; ragX pays off when the
evidence is spread across documents the question does not all name.

### Is ragX trained or fine-tuned?

No. It works with any instruction-following model through prompts alone.

### Which models work best?

Models that reliably follow a one-action-per-turn format: most models from about 4B parameters up,
and every hosted frontier model tested. Very small models (below about 4B) often break the action
format.

### How large a corpus can it handle?

The paper's benchmark ran it on all of English Wikipedia (21M passages) with dedicated retrieval
services. The library searches in memory, which comfortably covers corpora of tens of thousands to
a few hundred thousand passages on one machine.

### Can I see what the agent did?

Yes: `answer.trace` holds every step, `ask_stream` yields them live, and `ragx ask` shows them as it
runs.

### How do I cite ragX?

See [Citing ragX](./citing).
