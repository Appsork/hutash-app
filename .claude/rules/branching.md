# Branching

**Never commit directly to `main` UNLESS the operator explicitly instructs a
merge in the goal. When the operator says merge to main, do it.**

Always create a feature branch, develop there, and push the branch:

```bash
git checkout -b feat/<short-description>     # new capability
git checkout -b fix/<short-description>      # bug fix
git checkout -b refactor/<short-description> # no behaviour change
```

CI must pass on the branch before it can be merged. Absent an explicit
operator instruction to merge, a human reviews and merges to `main` — not the
branch author, and not automatically.

## Why

- `main` is what ships. A direct push puts unreviewed code in front of users
  with nothing between them and a mistake — the default path keeps a human in
  the loop for exactly that reason.
- CI runs on every branch push, so a broken change is caught before review
  rather than after it is live.
- A branch is where a change can still be discussed. Once it is on `main` the
  conversation is a revert.

## The explicit-instruction exception

The operator can override the human-review default by explicitly instructing
a merge to `main` in the goal itself ("merge to main", "push to main"). That
instruction is the equivalent of the human review this rule normally
requires — the operator is the human. Absent that explicit instruction, there
is no other exception: an implied urgency, a "quick fix" framing, or a prior
approval for a different action does not authorize a direct merge.
