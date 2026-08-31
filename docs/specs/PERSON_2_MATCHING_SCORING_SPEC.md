# Person 2 — Matching & Scoring SPEC
## Google Autocomplete Project

**Status:** Implementation-ready  
**Owner:** Person 2  
**Primary goal:** Correctly verify whether a normalized query matches any substring of a normalized candidate sentence with at most one allowed edit, and return the highest valid score.

---

## 1. Source of Truth and Precedence

This document defines the implementation contract for **Person 2 only**.

If any conflict appears, use this precedence:

1. Official Google project requirements.
2. Approved Team SPEC.
3. This Person 2 SPEC.
4. Implementation details chosen by Codex/developer.

Codex must not change an official or shared team contract on its own.

---

## 2. Responsibility

Person 2 owns:

- Exact substring matching.
- One-substitution matching.
- One-extra-character-in-query matching.
- One-missing-character-in-query matching.
- Edit position calculation.
- Penalty calculation.
- Score calculation.
- Returning the highest valid score for one candidate sentence.
- Matching/scoring unit tests and official Golden Tests.

Person 2 does **not** own:

- Archive extraction.
- File reading.
- Corpus loading.
- Text normalization.
- Source paths.
- Offsets.
- Candidate retrieval/indexing.
- `SentenceRecord`.
- `AutoCompleteData` construction.
- Top-5 selection.
- Alphabetical tie-breaking.
- CLI behavior.
- `#` handling.
- Cloud deployment.
- Protobuf.
- C++ optimization.

---

## 3. Public API

The required public function is:

```python
def calculate_best_match(
    query: str,
    sentence: str
) -> int | None:
    ...
```

### Input contract

`query`:
- A normalized user query.
- Lowercase/case-normalized according to the shared normalizer.
- Punctuation already handled by the shared normalizer.
- Repeated spaces already collapsed by the shared normalizer.

`sentence`:
- A normalized candidate sentence from the corpus.
- Uses the exact same normalization rules as `query`.

Person 2 **must not normalize either input again**.

### Output contract

Return:

```python
int
```

when at least one legal match exists.

Return:

```python
None
```

when no legal match exists.

`None` is the **only** no-match value.

A valid score may be:
- positive,
- zero,
- negative.

Never reject a legal match merely because its score is `<= 0`.

---

## 4. Defensive Empty-Input Behavior

Person 2 expects a non-empty normalized query during normal integration.

For defensive behavior:

```python
query == "" -> None
sentence == "" -> None
```

Person 3 is expected to handle the user-facing empty-query behavior.

This rule exists only to keep the matcher deterministic and safe if called directly.

---

## 5. Official Match Definition

The query is a valid match if **some contiguous substring** of `sentence` can match the entire `query` using:

- 0 edits, or
- exactly 1 allowed edit.

Allowed single edits:

1. Substitution — one query character is wrong.
2. Extra character in query — deleting one query character produces the target substring.
3. Missing character in query — inserting one character into the query produces the target substring.

More than one required edit is invalid.

The query may match:
- at the beginning of the sentence,
- in the middle,
- at the end.

This is **substring matching**, not whole-sentence similarity and not prefix-only matching.

---

## 6. Score Formula

For every legal match:

```text
score = 2 * matching_characters - edit_penalty
```

For an exact match:

```text
edit_penalty = 0
```

A substituted, extra, or missing character earns **no matching points**.

Spaces count as characters.

---

## 7. Character Positions

All edit positions are:

```text
1-based
```

and calculated relative to the **normalized query**.

Example:

```text
query = "to be"

t  o     b  e
1  2  3  4  5
```

The space is position `3`.

### Missing-character position

For a character missing from the query, use the position where that character would be inserted into the normalized query.

Example:

```text
query  = "or nt"
target = "or not"
```

The missing `o` is inserted at position `5`.

---

## 8. Penalty Tables

### 8.1 Substitution

```text
Position 1 -> 5
Position 2 -> 4
Position 3 -> 3
Position 4 -> 2
Position 5+ -> 1
```

Required helper:

```python
def substitution_penalty(position: int) -> int:
    ...
```

### 8.2 Insertion / Deletion

This penalty applies to:
- one extra character in the query, or
- one missing character in the query.

```text
Position 1 -> 10
Position 2 -> 8
Position 3 -> 6
Position 4 -> 4
Position 5+ -> 2
```

Required helper:

```python
def insertion_deletion_penalty(position: int) -> int:
    ...
```

Both helpers receive a **1-based** position.

---

## 9. Reference Algorithm — Team Implementation Decision

The reference implementation will solve the problem as:

```text
Approximate Substring Matching with edit distance <= 1
```

using:

- sliding-window scanning over possible target substrings,
- direct equal-length comparison,
- two-pointer comparison for one missing/extra character,
- early exit when more than one edit is required.

Do **not** use a full Levenshtein dynamic-programming matrix in the reference implementation.

Do **not** add fuzzy-matching libraries or third-party dependencies.

Correctness comes before optimization.

---

## 10. Why Only Three Target Lengths Are Needed

Let:

```python
n = len(query)
```

Because at most one edit is allowed, a legal target substring can only have length:

```text
n - 1
n
n + 1
```

Meaning:

```text
target length n
    -> exact match
    -> or one substitution

target length n - 1
    -> query contains one extra character

target length n + 1
    -> query is missing one character
```

Any other length difference requires more than one insertion/deletion and cannot be a valid match.

---

## 11. Sliding-Window Responsibility

Sliding-window scanning is used to find **where inside the sentence** a legal target substring may exist.

It does not itself calculate the score.

For every relevant target length, scan all valid contiguous start positions in `sentence`.

Conceptually:

```text
sentence
    ↓
possible target window
    ↓
matching helper
    ↓
score or None
```

The implementation does not need to pre-split the sentence or store all windows.

It may iterate using start/end indexes.

---

## 12. Equal-Length Matching — Exact or Substitution

When:

```text
len(target) == len(query)
```

compare characters position by position.

### Case A — 0 mismatches

This is an exact match.

```text
matching_characters = len(query)
score = 2 * len(query)
```

### Case B — exactly 1 mismatch

This is one substitution.

If the mismatch is at 1-based position `p`:

```text
matching_characters = len(query) - 1

score =
    2 * matching_characters
    - substitution_penalty(p)
```

### Case C — 2 or more mismatches

Invalid.

Return no score for this target.

The comparison should stop as soon as the second mismatch is detected.

---

## 13. Extra Character in Query

Condition:

```text
len(query) == len(target) + 1
```

Interpretation:

The user typed one extra character.

Use two pointers:

```text
i -> query
j -> target
```

Rules:

1. If `query[i] == target[j]`, advance both pointers.
2. At the first mismatch, treat `query[i]` as the possible extra character and advance only `i`.
3. Continue comparing.
4. If another mismatch remains after the single skip, this target is invalid.
5. If all remaining characters match, the target is valid.
6. A remaining final character in `query` may be the one extra character.

The extra character position is its **1-based position in query**.

Score:

```text
matching_characters = len(target)

score =
    2 * matching_characters
    - insertion_deletion_penalty(extra_position)
```

---

## 14. Missing Character in Query

Condition:

```text
len(target) == len(query) + 1
```

Interpretation:

The user omitted one character.

Use two pointers:

```text
i -> query
j -> target
```

Rules:

1. If `query[i] == target[j]`, advance both.
2. At the first mismatch, treat `target[j]` as the possible missing query character and advance only `j`.
3. Continue comparing.
4. If another mismatch remains after the single skip, the target is invalid.
5. If all remaining characters match, the target is valid.
6. A remaining final character in `target` may represent the missing character.

The penalty position is the **1-based insertion position in the normalized query**.

Score:

```text
matching_characters = len(query)

score =
    2 * matching_characters
    - insertion_deletion_penalty(insertion_position)
```

---

## 15. Multiple Possible Matches in One Sentence

A sentence may contain multiple legal target positions.

Example:

```text
query:
abc

sentence:
xbc ... ayc
```

Different legal matches may receive different scores because edit position changes the penalty.

`calculate_best_match()` must return:

```text
max(all_valid_scores)
```

Do not automatically stop after the first approximate match.

### Exact-match optimization

If an exact substring match is found, its score is:

```text
2 * len(query)
```

This is the maximum possible score for that query.

Therefore the implementation may safely return immediately after finding an exact match.

This optimization is optional for the reference version.

---

## 16. More Than One Edit

Any target requiring 2 or more edits is invalid.

Examples:

```text
query  = "xx be"
target = "to be"
```

Two substitutions are required.

Result for that target:

```python
None
```

Example:

```text
query  = "hxlpo"
target = "hello"
```

More than one correction is required.

Result:

```python
None
```

Do not calculate a lower score for a two-edit match.

It is not a legal match.

---

## 17. Required Internal File Ownership

Person 2 owns only:

```text
src/matching/penalties.py
src/matching/matcher.py
src/matching/scoring.py

tests/test_matcher.py
tests/test_scoring.py
```

Recommended responsibilities:

### `src/matching/penalties.py`

Contains:

```python
substitution_penalty(position: int) -> int
insertion_deletion_penalty(position: int) -> int
```

### `src/matching/matcher.py`

Contains the public matcher:

```python
calculate_best_match(query: str, sentence: str) -> int | None
```

and private/internal matching helpers if needed.

### `src/matching/scoring.py`

Contains pure scoring helpers if separating score computation improves clarity.

Do not create unrelated modules without a concrete need.

---

## 18. Prohibited Scope Changes

Codex must **not**:

- modify shared dataclasses,
- change `calculate_best_match()` signature,
- implement normalization,
- implement corpus/index logic,
- read files,
- implement Top-5 logic,
- create CLI behavior,
- add databases,
- add APIs/web servers,
- add Gemini runtime calls,
- add Protobuf,
- add C++,
- add GCP services,
- add third-party fuzzy-search packages,
- refactor other team members' modules,
- rename shared/public functions without team approval.

If implementation requires a shared-contract change, stop and report the need instead of making the change.

---

## 19. Official Golden Tests

The official source sentence is:

```text
To be or not to be, that is the question.
```

Person 2 receives normalized input, therefore tests should use:

```python
sentence = "to be or not to be that is the question"
```

Required normalized-query results:

```text
"to be"    -> 10
"or not"   -> 12
"be that"  -> 14
"2o be"    -> 3
"to pe"    -> 6
"or knot"  -> 8
"or nt"    -> 8
"not be"   -> None
```

These are non-negotiable correctness tests.

---

## 20. Additional Required Tests

### Exact matching

- Exact query at sentence beginning.
- Exact query in sentence middle.
- Exact query at sentence end.
- Spaces counted in score.

### Substitution

- Position 1.
- Position 2.
- Position 3.
- Position 4.
- Position 5+.
- Substitution at final query character.
- Two substitutions -> `None`.

### Extra character in query

- Extra character at beginning.
- Extra character in middle.
- Extra character at end.
- Second mismatch after skip -> invalid.

### Missing character in query

- Missing character in an internal position.
- Missing-character insertion position calculated correctly.
- Second mismatch after skip -> invalid.

### Multiple matches

- Several approximate matches in one sentence.
- Return the highest score, not the first score.

### Score edge cases

- Valid score `0` remains valid.
- Valid negative score remains valid.
- `None` alone means no match.

### Defensive cases

- Empty query -> `None`.
- Empty sentence -> `None`.
- Query much longer than sentence and impossible with one edit -> `None`.

---

## 21. Performance Requirements for Reference Version

Correctness is the first objective.

Reference worst-case behavior may be approximately:

```text
O(sentence_length * query_length)
```

per candidate sentence.

The implementation should still use simple low-risk improvements:

- stop equal-length comparison on the second mismatch,
- stop two-pointer comparison when a second edit is required,
- avoid storing all sentence windows at once,
- avoid unnecessary work after an exact match is found.

Do not introduce complex optimization before benchmarks/profiling show a need.

---

## 22. Optimization Contract

Future optimization is allowed only if:

1. All unit tests pass.
2. All official Golden Tests pass.
3. Behavior remains identical to the reference matcher.
4. `calculate_best_match()` signature remains unchanged.

Optimization may improve runtime.

Optimization may **not** change:

- whether a match is valid,
- the returned score,
- the one-edit rule,
- position penalties.

---

## 23. Definition of Done

Person 2 is complete when:

- `calculate_best_match()` exists with the exact agreed signature.
- Exact substring matching works anywhere in the sentence.
- One substitution works.
- One extra query character works.
- One missing query character works.
- Two or more edits are rejected.
- Position rules are correct and 1-based.
- Spaces count as characters.
- Penalty tables are exact.
- Scores are exact.
- Zero/negative valid scores are preserved.
- Highest score is returned when several matches exist.
- All official Golden Tests pass.
- Additional edge-case tests pass.
- No other team member's responsibilities were implemented or modified.

---

## 24. One-Sentence Contract

> Given a normalized query and a normalized candidate sentence, find the highest-scoring substring match that requires at most one allowed character edit, and return that score; return `None` if no legal match exists.
