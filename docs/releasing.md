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
   VERSION=$(sed -n "s/^__version__ = '\(.*\)'/\1/p" nmd/__init__.py)
   test -n "$VERSION" && git tag -a "v$VERSION" -m "v$VERSION" && git push origin master "v$VERSION"
   ```

   `sed` rather than `grep -oP`: Git Bash on this machine runs with `LANG` unset, where
   `grep -P` refuses with "supports only unibyte and UTF-8 locales" and returns nothing — which
   in a `git tag "v$(...)"` one-liner silently creates a tag named `v`, and `v` matches the
   `v*` trigger. The `test -n` guard is what stops that; the workflow's own copy of the read
   runs on ubuntu and already checks for an empty result.

   ⚠ `-a`, and pushing the tag by name, are both load-bearing — this line got them wrong
   until 2026-09-18. `git tag` without `-a` creates a **lightweight** tag, and
   `git push --follow-tags` pushes *annotated* tags only, so the old form pushed the
   branch, silently left the tag on the laptop, and this workflow never fired. It fails
   by looking like it worked: the push succeeds and says nothing about the tag. Verified
   2026-09-18 with `git push --dry-run --follow-tags`, which lists the branch alone for a
   lightweight tag and the branch plus the tag for an annotated one.

3. **Watch the run.** Stages: validate → test matrix + dependency-free install check → build,
   `twine check`, clean-venv smoke test, provenance, publish. Build through publish are steps in
   one job, so the bytes that are smoke-tested and attested are the bytes that are uploaded.

`workflow_dispatch` runs everything *except* the upload, as a repeatable dry run. To check the
same things locally first, off a real build rather than off the source tree, follow
`docs/release-dry-run.md` — it is ~10 minutes and it is repeatable.

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

**Dry run green, 2026-09-18.** `workflow_dispatch` run `35346407033` on `5cb17ef`: `validate-tag`
with its three checks skipped (non-tag ref), all 11 matrix cells green (ubuntu 20-29s, windows
47-62s, macos 18s), the bare install green, and the `deploy` job through build, `twine check`,
the clean-venv smoke test (`wheel smoke test OK: 0.438...`) and provenance, with the upload step
skipped as designed. Its triggers are `push: tags: ['v*']` and `workflow_dispatch`, so pushing to
`master` does not start it. **Never uploaded.** What a real tag will exercise for the first time
is exactly the three `validate-tag` checks and the upload step; everything else has now run on
real runners.

`ci.yml` was added on 2026-09-18 and
runs on every push to any branch, so the suite and the dependency-free check also execute off the
release path. **It ran green on its first attempt** — run `35305903082`, commit `18672a7`,
29s total: ubuntu/3.10 25s, ubuntu/3.14 23s, bare install 11s. `publish-to-pypi.yml` correctly did
not fire.

That first run matters for this file, because `ci.yml` and `publish-to-pypi.yml` **share two of
the four workflows**. `template-test.yml` and `template-verify-minimal.yml` have now executed on
real runners, so the reusable-workflow wiring, the dependency install and the bare-install
assertion are proven rather than merely linted. The dry run above then covered the rest of the
release path except `validate-tag` and the upload. `publish-to-pypi.yml` is still the only thing
that builds, attests or uploads, and is still the only place the full 11-cell matrix runs — but
"has this code ever been tested on linux" is now answered before release time rather than during
it. Scope is documented in the header of `ci.yml`.

Published versions `0.0.1`–`0.0.6` all predate the pipeline and went out by hand. The first tag
pushed will be the first real upload through it. Repeat the `workflow_dispatch` dry run after
any change to the workflows or the packaging; it is two minutes.

**Hand dry run, 2026-09-18.** Everything the pipeline claims to check was checked locally first,
off a real `python -m build` rather than off the source tree: `twine check --strict` green on both
artifacts; METADATA carries no `Requires-Dist`, `Requires-Python: >=3.10` and a Summary resolved
from the module docstring; the wheel installed alone into a bare python **3.10.21** env (the
declared floor) leaks none of `numpy` / `scipy` / `pyroaring` / `regex` / `numba` on `import nmd`;
every README code block runs against that installed wheel and reproduces its documented numbers;
and the suite is green on 3.10 as well as the repo's 3.12. It is now a runbook rather than a
one-off — `docs/release-dry-run.md`, driving `experiments/release_check.py`. Write-up:
`docs/session-log.md`, session 2026-09-18. This substitutes for none of the CI — it covers two of
the eleven matrix cells and cannot exercise trusted publishing — but it means a
`workflow_dispatch` run is now expected to be the first thing that fails, if anything does.

⚠ **`twine check` does not render a markdown description.** `twine/commands/check.py::_RENDERERS`
maps `"text/markdown": None` with the comment "Rendering cannot fail", so on this package it
validates metadata only, under `--strict` and regardless of installed backends. The workflow step
is named "Check the metadata" and means it literally; nothing in the pipeline renders `README.md`.

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
