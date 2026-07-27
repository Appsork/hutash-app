# Branching

**Never commit directly to `main`.**

Always create a feature branch, develop there, and push the branch:

```bash
git checkout -b feat/<short-description>     # new capability
git checkout -b fix/<short-description>      # bug fix
git checkout -b refactor/<short-description> # no behaviour change
```

CI must pass on the branch before it can be merged. A human reviews and merges
to `main` — not the branch author, and not automatically.

## Why

- `main` is what ships. A direct push puts unreviewed code in front of users
  with nothing between them and a mistake.
- CI runs on every branch push, so a broken change is caught before review
  rather than after it is live.
- A branch is where a change can still be discussed. Once it is on `main` the
  conversation is a revert.

## The one exception

There isn't one. If something is urgent enough to skip review, it is urgent
enough to get a branch, a passing CI run and one pair of eyes — that path takes
minutes, and a bad hotfix on `main` costs far more.
