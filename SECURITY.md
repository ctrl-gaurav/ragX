# Security Policy

## Reporting a vulnerability
Please report security issues privately via GitHub Security Advisories ("Report a vulnerability")
rather than a public issue. We aim to acknowledge reports within 72 hours.

## Scope & notes
- **API keys** are read from the environment / `.env` by name and are never logged or printed by
  ragX. Never commit your `.env`.
- ragX reads the files you point it at and never executes their contents. Retrieved text is fenced
  as untrusted data by default (see the Security page of the documentation), but an agent that
  reads planted instructions is an attack surface: ingest corpora you trust where you can.
- The optional `serve` endpoint has no auth by default — put it behind your own gateway before
  exposing it.
