# Releasing

How `nmd` gets to PyPI.

This replaces the "Publishing (notes for myself)" section that sat at the bottom of `README.md`
until 2026-09-17. That section described `flit publish` from a laptop, which is no longer how
this is done; it is archived at the end of this file rather than deleted.

## The procedure

Publishing is done by `.github/workflows/publish-to-pypi.yml`, triggered by pushing a tag
matching `v*`. Nothing in the workflow writes to the repository — no version bump, no commit, no
tag, no GitHub Release. **The tag you push is the only source of truth.**

1. **Bump `__version__` in `nmd/__init__.py`** and commit it on `master`.

   It cannot be dropped or injected from the tag. With `dynamic = ["version"]`, flit_core looks
   for the literal via AST and falls back to *importing* the module; delete the line and the
   build fails in `flit_core/common.py::get_info_from_module` (verified 2026-09-17). Injecting
   it in CI would work but would mean the published bytes are not the bytes at the tagged
   commit, which is the one thing the build attestation exists to guarantee.

2. **Derive the tag from the file** rather than retyping it, so the two cannot disagree:

   ```bash
   git tag "v$(grep -oP "__version__\s*=\s*'\K[^']+" nmd/__init__.py)"
   git push --follow-tags
   ```

3. **Watch the run.** Stages: validate → test matrix + dependency-free install check → build,
   `twine check`, clean-venv smoke test, provenance, publish. Build through publish are steps in
   one job, so the bytes that are smoke-tested and attested are the bytes that are uploaded.

`workflow_dispatch` runs everything *except* the upload, as a repeatable dry run.

## What the guards catch, before anything irreversible happens

A tag is deletable (`git push --delete origin <tag>`); a PyPI upload is not. Everything in
`validate-tag` runs in ~20s, ahead of the matrix.

| guard | catches |
|---|---|
| tag commit is an ancestor of `master` | a tag pushed from a side branch, publishing code that is not on `master` |
| tag matches `__version__` | the two drifting apart — flit reads the file, not the tag |
| version is not already on PyPI | re-tagging a published version. Fails closed: only a 404 proceeds |
| `verify-minimal` | `import nmd` growing a third-party import, or `pyproject.toml` growing a dependency. Asserts the optional packages are genuinely absent first, so it cannot pass vacuously |
| clean-venv smoke test | a file missing from the wheel, which is invisible from the source tree because the source tree still has it |

## Prerequisites

* **PyPI trusted publisher** for project `nmd` — owner `averykhoo`, repo
  `ngram-movers-distance`, workflow `publish-to-pypi.yml`, environment `release`. Reported
  configured by the owner on 2026-09-17; not verifiable from inside the repo.
* **The `release` GitHub Environment**, which the `deploy` job references. GitHub creates it on
  first use if it does not exist, with no protection rules.

## Status

**Never run, as of 2026-09-17.** The workflow is not yet pushed; `actions/workflows` on the
repository reports `total_count: 0`. Published versions `0.0.1`–`0.0.6` all predate it and went
out by hand. The first tag pushed will be the first real exercise of the pipeline — a
`workflow_dispatch` dry run first is worth the two minutes.

## Archived: how this used to work (0.0.1 – 0.0.6)

Verbatim from the old README section, for the record:

> * init
>     * `pip install flit`
>     * `flit init`
>     * make sure `nmd/__init__.py` contains a docstring and version
> * publish / update
>     * increment `__version__` in `nmd/__init__.py`
>     * `flit publish`

The `flit init` half is history — the package is already initialized and `pyproject.toml` is in
the tree. The `flit publish` half should not be used again: it uploads bytes that nothing has
tested on any python but the laptop's, matched against a tag, or attested.
