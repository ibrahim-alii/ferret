# Conversational QA scenario catalog

Probes for the `qa-conversational` agent (and for humans). Each entry lists the **intended
behavior class**, not an exact expected string — judge responses by mode semantics, not
string match.

Behavior classes:

- **chat** — greeting / small talk / meta. No retrieval, no citations.
- **clarify** — ambiguous, or references an unnamed paper set in Ask mode. One short
  question; in Ask mode, nudge toward Deep Dive when a specific paper is implied. No
  retrieval, no citations.
- **general** — general-knowledge / conceptual. Answered from model knowledge, flagged as
  not drawn from an ingested paper. No citations.
- **research (ask)** — searches the corpus on a *named* topic. May cite papers (by title +
  year). Must not imply the corpus is an exhaustive survey.
- **research (deep_dive)** — grounded in the single ingested paper only. Cites that paper's
  sections; must not drift to other papers.

When a new bug is found in the wild, append it here as a scenario so it becomes permanent
coverage.

---

## Ask mode

### Mode confusion — demonstrative reference to an unnamed paper set → clarify
The canonical bug. "These papers" / "this paper" has no referent in Ask mode.

- "What problem do these papers address and what methods do they use?" → **clarify**
- "Summarize this paper's contributions." → **clarify** (+ nudge to Deep Dive)
- "What do those papers conclude?" → **clarify**
- "Give me the key results from the above papers." → **clarify**
- "How do these papers compare?" → **clarify**

### Vague / underspecified → clarify
- "Tell me about the recent papers." (no topic) → **clarify**
- "Help me." → **clarify**
- "What's interesting?" → **clarify**

### General knowledge → general (no citations)
- "What is backpropagation?" → **general**
- "Explain attention." → **general**

### Legitimate corpus research → research (ask)
- "Compare diffusion models for image generation." → **research (ask)**
- "What are the latest papers on RLHF?" → **research (ask)**, recency-aware
- "Summarize recent work on retrieval-augmented generation." → **research (ask)**

### Greetings → chat (no retrieval)
- "hi there!" → **chat**
- "what can you do?" → **chat**

---

## Deep Dive mode

Ingest a paper first. Suggested arXiv id: **1706.03762** ("Attention Is All You Need") —
small, fast, well known. Any already-ingested paper is fine.

### Single-paper questions stay grounded → research (deep_dive)
The same demonstrative phrasing is *legitimate* here (there is exactly one paper).

- "What does this paper conclude?" → **research (deep_dive)**, grounded in the paper
- "What methods do these papers use?" → **research (deep_dive)** (guard must NOT fire here)
- "What is the main contribution?" → **research (deep_dive)**
- "Explain the architecture proposed here." → **research (deep_dive)**

### Drift probes — must stay on the ingested paper
- "How does this compare to other transformer papers?" → answer about *this* paper's
  stated comparisons only; flag **FAIL** if it invents/retrieves other corpus papers.
- "What came after this work?" → should note it cannot speak beyond the paper (or offer to
  search), not fabricate follow-up papers.

### Greetings → chat (no retrieval)
- "thanks!" → **chat**
