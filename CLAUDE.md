# CLAUDE.md — ngram-movers-distance

The durable contract for this repo: rules, gate, invariants. Volatile state — what
is true now, ranked open items, open questions — lives in `HANDOFF.md`; read that
next, in full.

## Gate

```bash
"C:/Users/user/anaconda3/envs/ngram-movers-distance/python.exe" -m pytest -q
# 1180 passed, 1 xfailed (2026-09-18)
```

Green before every push, no exceptions. If it is red, say what is red and leave it.

`.github/workflows/ci.yml` runs the same suite on every push, on ubuntu/3.10 (and additionally
ubuntu/3.14 plus the dependency-free install check on `master`). It is a **cheap subset**, not a
replacement for the local gate: the full 11-cell matrix only runs on the release path. Workflow
YAML cannot be tested locally — lint it with `actionlint`, per `docs/release-dry-run.md`.

## Invariants

- **`nmd` stays dependency-free, and the default path stays pure python.**
  `import nmd` exports exactly `ngram_movers_distance` and `WordList`, and pulls in
  nothing third-party. Anything needing `numpy` / `scipy` / `pyroaring` / `regex` is
  reached by its full module path, is not exported from `nmd/__init__.py`, and is
  never added to `pyproject.toml` as a dependency or an extra. Pinned by the
  `verify-minimal` job in `.github/workflows/publish-to-pypi.yml`.
- **`ApproxWordListV3` and `ApproxWordListV5` are frozen.** They are kept unchanged
  for comparison and their bugs are documented rather than fixed. `ApproxWordListV6`
  (the shipped `WordList`) and `ApproxWordListV7` are not frozen.
- **Date-stamp every measured number written into a doc**, and point at something in
  the tree that regenerates it — e.g. `experiments/readme_example_check.py` for the
  README's edit-distance numbers. Re-measure rather than differencing against a
  recorded count.

## Never release unless asked

Do not tag a release, run the publish workflow, or bump `__version__` in order to
trigger one, unless the user asked **to release**. "Push", "the pipeline is ready",
"the gate is green", and an approved version-bump commit are none of them a release
ask.

The pipeline is tag-triggered, so **creating a `v*` tag is the release** — there is
no confirmation step between the tag and the upload.

A push is recoverable; a release is not. PyPI refuses to re-upload a version that
has ever existed, including one that was deleted or yanked, so the version number is
spent the moment the upload lands.

### To release, once asked

1. **Bump `__version__` in `nmd/__init__.py`** and commit it on `master`. flit reads
   the version from that literal; it cannot be dropped, and injecting it from the tag
   would mean the published bytes are not the bytes at the tagged commit.
2. **Tag from the file**, so the two cannot disagree, and push the tag *by name*.
   `test -n` matters: without it a failed read tags `v`, which matches the `v*` trigger.
   `-a` matters too: `--follow-tags` pushes annotated tags only, so a lightweight tag
   stays on the laptop and the workflow never fires. See `docs/releasing.md`.
   ```bash
   VERSION=$(sed -n "s/^__version__ = '\(.*\)'/\1/p" nmd/__init__.py)
   test -n "$VERSION" && git tag -a "v$VERSION" -m "v$VERSION" && git push origin master "v$VERSION"
   ```
3. `workflow_dispatch` runs everything **except** the upload, as a repeatable dry
   run. Worth doing first: the pipeline has never actually run.

⚠ `0.0.6` is already on PyPI, so `__version__` must move before any release can
succeed — `validate-tag` checks this and stops in ~20s rather than at the upload.

Full procedure, what each guard catches, and the trusted-publisher prerequisites:
`docs/releasing.md`.
