# Release dry run

How to verify a built `nmd` wheel locally, before pushing a tag. The release procedure itself —
what to bump, how to tag, what the CI guards catch — is `docs/releasing.md`; this file is only
the local half.

Worth doing because the publish pipeline is tag-triggered and a PyPI upload is not reversible,
and because as of 2026-09-18 the pipeline has still never run. First performed 2026-09-18;
that session's findings are in `docs/session-log.md`.

> ⚠ **Delete both conda envs when you are done.** They are throwaways, they are named so you can
> find them, and step 6 is not optional. This laptop runs several Claude sessions across
> different repos at once; leaving envs behind spends disk that nothing else can see is spent.

## What it checks, and what it cannot

Everything here runs against **the built wheel**, never the source tree — a file missing from the
wheel is invisible from the tree, because the tree still has it. `experiments/release_check.py`
refuses to run if `import nmd` resolves inside the repo.

Covered: the artifacts build; `twine check --strict`; the metadata declares no dependencies; the
wheel installs and works alone on the declared python floor; `import nmd` pulls in nothing
third-party; the README's dependency table names the right package for each optional module;
every README code block runs and reproduces the numbers printed in its comments; the suite passes
on the floor.

**Not** covered, and left to CI: nine of the eleven matrix cells — this runs python 3.10 and 3.12
on windows, which is two of them (counted from the workflow 2026-09-18: 2 OSes × 5 pythons, plus
macos/3.12) — along with trusted publishing, build provenance, and the `validate-tag` guards,
which need a real tag. Nothing here renders `README.md` either — see the note in
`docs/releasing.md` about `twine check` not rendering markdown descriptions.

## Prerequisites

Bare `python` is a Store stub on this machine and `conda` is not on `PATH`, so everything below
uses full paths. Run it from **Git Bash**, at the repo root.

```bash
CONDA="C:/Users/user/anaconda3/Scripts/conda.exe"
REPO_PY="C:/Users/user/anaconda3/envs/ngram-movers-distance/python.exe"
BARE_PY="C:/Users/user/anaconda3/envs/nmd-release-bare/python.exe"
FULL_PY="C:/Users/user/anaconda3/envs/nmd-release-full/python.exe"
```

## 1. Gate

Nothing downstream means anything if the suite is red.

```bash
"$REPO_PY" -m pytest -q      # 1102 passed, 1 xfailed (2026-09-18)
```

## 2. Build the artifacts

A throwaway venv, so `build` and `twine` never enter the repo env. `.scratch/` is gitignored.

```bash
rm -rf dist                                     # never test a stale artifact
"$REPO_PY" -m venv .scratch/venv-build
.scratch/venv-build/Scripts/python.exe -m pip install --quiet --upgrade pip build twine
.scratch/venv-build/Scripts/python.exe -m build
.scratch/venv-build/Scripts/python.exe -m twine check --strict dist/*
```

Expect `nmd-<version>-py3-none-any.whl` and `nmd-<version>.tar.gz`, both `PASSED`.

## 3. Create the two environments

Two, not one, because the checks need contradictory environments: one asserts the optional
packages are **absent**, the other that they are **present**. Separate envs also let step 5 run
both at once.

Create them one after the other — conda serializes on its package cache, so concurrent `create`
calls contend rather than overlap. Each takes well under a minute.

```bash
"$CONDA" create -n nmd-release-bare -y python=3.10   # the declared floor, per pyproject.toml
"$CONDA" create -n nmd-release-full -y python=3.10
```

## 4. Install

The bare env gets **only** the wheel. Do not install anything else into it, including pytest —
that is the whole point of it.

```bash
"$BARE_PY" -m pip install --quiet dist/*.whl

"$FULL_PY" -m pip install --quiet dist/*.whl numpy scipy pyroaring regex pytest
```

## 5. Run the checks — these two can go in parallel

```bash
# terminal A, or a background job
"$BARE_PY" experiments/release_check.py bare

# terminal B
"$FULL_PY" experiments/release_check.py full
"$FULL_PY" -m pytest -q          # the suite on the declared floor
```

Each `release_check` run prints one `ok` / `FAIL` line per claim and exits non-zero on any
failure. ⚠ Read the **exit code**, not just the tail: piping either command through `tee` or
`tail` reports the pipe's status, not the script's.

`release_check` starts by asserting the environment is the one that mode needs, so running
`bare` in the full env fails loudly instead of passing vacuously. It also pins every expected
number to a literal in `README.md`, so editing the README without re-measuring goes red.

Worth running alongside, from the repo env, since it needs the corpus rather than the wheel:

```bash
"$REPO_PY" experiments/readme_example_check.py    # the edit-distance numbers in the README intro
```

## 6. Tear down — do not skip this

```bash
"$CONDA" env remove -n nmd-release-bare -y
"$CONDA" env remove -n nmd-release-full -y
rm -rf .scratch/venv-build dist
"$CONDA" env list        # confirm neither nmd-release-* is left
```

Keep `experiments/release_check.py`; it is tracked, and re-running it is cheaper than
reconstructing what it checked.

## Linting the workflows

Workflow YAML cannot be exercised locally at all — the only way to run it is to push. `actionlint`
is the nearest thing to a check, and it does validate the reusable-workflow wiring, which is the
part most likely to be wrong:

```bash
"$REPO_PY" -m venv .scratch/venv-lint
.scratch/venv-lint/Scripts/python.exe -m pip install --quiet actionlint-py
.scratch/venv-lint/Scripts/actionlint.exe -no-color -oneline      # silent + rc 0 means clean
rm -rf .scratch/venv-lint
```

`actionlint-py` ships the binary, so there is no Go toolchain involved; note it installs as
`Scripts/actionlint.exe` rather than an importable module. Verified 2026-09-18 to catch both an
input that the called workflow does not declare and a `uses:` pointing at a file that does not
exist — the two ways the split between `ci.yml`, `publish-to-pypi.yml` and the two templates can
break.

## Inspecting the artifacts by hand

Not part of the routine run, but this is how the 2026-09-18 findings about the sdist and the
metadata were made:

```bash
.scratch/venv-build/Scripts/python.exe - <<'PY'
import glob, tarfile, zipfile
whl, = glob.glob('dist/*.whl')
print('\n'.join(sorted(zipfile.ZipFile(whl).namelist())))
sdist, = glob.glob('dist/*.tar.gz')
print('\n'.join(sorted(tarfile.open(sdist).getnames())))
print(zipfile.ZipFile(whl).read(
    [n for n in zipfile.ZipFile(whl).namelist() if n.endswith('METADATA')][0]).decode('utf8')[:900])
PY
```

Two things to look at. METADATA must have **no `Requires-Dist` line at all** — that is the
dependency-free promise as the installer sees it. And the sdist contains only `nmd/*.py`,
`pyproject.toml`, `README.md`, `LICENSE` and `PKG-INFO`: no `tests/`, no `docs/`, no
`experiments/`. That is flit's default and is fine, but it is why the README's links are absolute
and why its `# Testing` section is a repository instruction rather than one an sdist download can
follow.
