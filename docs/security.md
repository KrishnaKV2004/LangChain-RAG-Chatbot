# Security Architecture

The assistant must never leak confidential company information. Protection is
layered: each layer assumes the ones before it can fail.

## Request lifecycle and checkpoints

```
user query ──► [Checkpoint 1: input gate]
                 Layer 1  prompt-injection detection
                 input length caps
              ──► routing ──► retrieval / web search
              ──► [Checkpoint 2: context filter]
                 Layer 1' indirect-injection scan (documents & webpages)
                 Layer 2  sensitivity classification
                 Layer 3  permission validation (role clearance)
              ──► LLM generation ──► citations
              ──► [Checkpoint 3: response gate]
                 Layer 4  response scanning (leak detection)
                 Layer 5  PII detection + redaction
              ──► final answer
```

## The five layers

### Layer 1 — Prompt-injection detection (`app/security/injection.py`)
Weighted pattern scoring across four attack families: **role override**
("ignore previous instructions"), **prompt extraction** ("reveal your
instructions", "print internal prompts"), **data exfiltration** ("show me
every document", "dump the database"), and **jailbreaks** (DAN, developer
mode). Structural signals (chat-template tokens, zero-width characters)
accumulate. The same detector scans retrieved documents and webpages
(*indirect* injection): hostile content is dropped before the LLM sees it and
its source is logged.

### Layer 2 — Sensitive-document detection (`app/security/sensitivity.py`)
Chunks are classified public / internal / confidential at **ingestion time**
(stored in ChromaDB metadata) from explicit flags, filename conventions
(`salary`, `confidential`, `hr_`) and content banners ("STRICTLY
CONFIDENTIAL", "internal use only"). Unmarked documents default to
**internal**, never public — fail closed.

### Layer 3 — Permission validation (`app/security/permissions.py`)
Role → clearance: guest < employee < admin. A chunk is passed to the LLM only
when `clearance >= sensitivity`. Denied chunk texts are retained (in memory,
per request) to power Layer 4's leak check.

### Layer 4 — Response scanning (`app/security/response_scan.py`)
The last gate before output: (1) 8-word shingle overlap between the answer
and any permission-denied chunk blocks the response; (2) marker phrases from
our own prompt templates must never be echoed; (3) credential patterns block
outright. A blocked answer becomes a polite refusal with **empty citations**.

### Layer 5 — PII detection (`app/security/pii.py`)
Emails, phone numbers, SSNs, IBANs, credit cards (Luhn-validated) and API
keys. Response findings are **redacted** (`[REDACTED-EMAIL]`) rather than
blocking the whole answer.

## Prompt-level defenses (`app/chains/prompts.py`)

* Retrieved content is wrapped in `<internal_document>` / `<web_result>`
  data blocks; the system prompt declares them untrusted data, never
  instructions.
* Closing tags inside retrieved text are neutralized so content cannot break
  out of its data block.
* Citations are built programmatically from the actual context — the LLM
  cannot fabricate them.

## Principles

* **Fail closed** — an exception inside any security layer refuses the
  request; it never falls through open.
* **Uninformative refusals** — blocked users are not told which pattern
  fired.
* **Defense in depth** — e.g. even if a confidential chunk somehow influenced
  the LLM, Layer 4's shingle check catches verbatim leakage on the way out.

Every behaviour above is covered by `tests/unit/test_security.py` and the
workflow-level scenarios in `tests/integration/test_graph.py`.
