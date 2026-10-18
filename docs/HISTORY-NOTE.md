# Git history note

Before this repository was (re)published on GitHub, its history was rewritten with
`git filter-branch`. That happened **twice**, both times for the same kind of reason: a factual or
privacy correction to text that was present from the first commit.

| When | What changed | Why |
|---|---|---|
| 2026-10-18 (1st) | One line in an early version of `docs/00-course-requirements.md` that named a private person (a fellow student) was removed | Privacy. The line was already gone from the current files |
| 2026-10-18 (2nd) | The institution in `README.md` and `CLAUDE.md` was corrected from the author's *previous* (bachelor's) university to the correct one: **Technical University of Sofia, Plovdiv Branch, Faculty of Electronics and Automation (FEA)**, Master's programme | Factual error since the first commit |

Each rewrite also updated the commit IDs cited in the docs to the new IDs it produced.

**Nothing else changed:**
- no code, data or results;
- no predictions;
- no commit dates (author and committer timestamps are preserved);
- no commit order.

Every other changed line differs from the original only by a commit ID. This was verified
mechanically after each rewrite. The original histories are kept by the author as local git bundles
and can be shown on request.

## Pre-registration commits (original → current ID)

Predictions were committed **before** the measurements that test them (CLAUDE.md rule 7). These are
the commits that carry that evidence:

| Round | Original ID | After 1st rewrite | Current ID | Commit time (unchanged) |
|---|---|---|---|---|
| All-pairs round 2 (timeline, gates) | ec31cdd | aeae461 | 9bae5fa | 2026-10-17 09:36:22 +0300 |
| Tree round 1 (#4, #5) | d738d77 | 6636388 | 581fc47 | 2026-10-17 10:28:05 +0300 |
| Tree round 2 (#6, #7) | bac1a84 | c8d9fcc | c6a12cc | 2026-10-17 14:41:34 +0300 |
| IC round (shapes, masses) | 3f923f3 | be68419 | 56505d8 | 2026-10-17 22:25:08 +0300 |
| Clean re-run (replication plan) | — | fd8c577 | fbe0742 | 2026-10-18 04:31:30 +0300 |
| Clean re-run (resume option) | — | 92e9598 | bfe3c5b | 2026-10-18 05:02:02 +0300 |

Also remapped: ch.2 refresh fea9185 → f95706b → 18313ca (cited in critic review 1).

To check pre-registration yourself, compare a predictions commit's time with the timestamps in that
round's `env_before.txt`, or run `git log -- docs/predictions/<file>`.
