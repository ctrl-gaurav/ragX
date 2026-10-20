# Citations and grounding

Every answer from `rag.ask` carries the passages it came from.

```python
answer = rag.ask("Whom did the founder of Ashford University marry?")

answer.text          # 'Eleanor Hale'
answer.sources       # ['thomas_wren.md:1-5']
answer.citations     # [Citation(path='thomas_wren.md', line_start=1, line_end=5, quote='# Thomas Wren')]
answer.grounded      # True
```

## How citations are chosen

After the compose call, ragX checks the answer against every passage it composed over. A passage is
cited when it contains the answer verbatim (ignoring case and punctuation), or when it contains at
least 80% of the answer's content words. Citations point at the source file and line range, not at
the internal text mirror, so `path:start-end` opens the right lines in your editor.

## Grounded or not

`answer.grounded` is `True` only when at least one composed passage contains the answer in that
sense. When none does, ragX still lists the first composed passages as citations, so you can see
what the model was given, and sets `grounded=False`.

Treat an ungrounded answer as unsupported: the model answered from something other than the cited
text, often its own memory.

## Abstaining

To get a refusal instead of an ungrounded answer, set:

```python
rag = RagX("./docs", abstain_on_weak_evidence=True)
```

The compose prompt then allows `I don't know`, and any ungrounded answer is replaced by a message
saying the corpus did not contain enough evidence.

## Turning the check off

`verify_citations=False` cites the composed passages without checking them and reports the answer
as grounded whenever there are passages. Use it only when you verify answers yourself.
