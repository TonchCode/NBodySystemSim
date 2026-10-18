# Git history note

On **2026-10-18**, before the first push to GitHub, this repository's history was rewritten once
with `git filter-branch`. The only reasons:

1. **Remove one line that named a private person** (a fellow student) from an early version of
   `docs/00-course-requirements.md`. The line was already gone from the current files.
2. **Update the commit IDs cited in the docs** to the new IDs that the rewrite produced.

**Nothing else changed:**
- no code, data or results;
- no predictions;
- no commit dates (author and committer timestamps are preserved);
- no commit order.

Every other changed line differs from the original only by a commit ID. This was verified
mechanically before pushing. The original history is kept as a local git bundle by the author and
can be shown on request.

## Pre-registration commits (old → new ID)

Predictions were committed **before** the measurements that test them (CLAUDE.md rule 7). These are
the commits that carry that evidence:

| Round | Original ID | New ID | Commit time (unchanged) |
|---|---|---|---|
| All-pairs round 2 (timeline, gates) | ec31cdd | 9bae5fa | 2026-10-17 09:36:22 +0300 |
| Tree round 1 (#4, #5) | d738d77 | 581fc47 | 2026-10-17 10:28:05 +0300 |
| Tree round 2 (#6, #7) | bac1a84 | c6a12cc | 2026-10-17 14:41:34 +0300 |
| IC round (shapes, masses) | 3f923f3 | 56505d8 | 2026-10-17 22:25:08 +0300 |

Also remapped: fea9185 → 18313ca (ch.2 refresh, cited in critic review 1).

To check pre-registration yourself, compare a predictions commit's time with the timestamps in that
round's `env_before.txt`, or run `git log -- docs/predictions/<file>`.
