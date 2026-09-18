"""Verify a built `nmd` wheel against what the README promises, before a release.

    python experiments/release_check.py bare   # in an env holding ONLY the wheel
    python experiments/release_check.py full   # + numpy, scipy, pyroaring, regex

Runbook, including how to build the wheel and how to create and delete the two
environments: docs/release-dry-run.md. First run 2026-09-18.

The two modes are separate because they need contradictory environments, and
because the bare one is the load-bearing claim: `import nmd` pulls in nothing
third-party. Run them in two envs, in parallel; neither writes anything.

    bare  asserts numpy / scipy / pyroaring / regex are ABSENT, then checks the
          dependency-free promise, `__all__`, the README quickstart block, and
          that each optional module fails on a package its README dependency
          table actually lists
    full  asserts those packages are PRESENT, then runs every remaining README
          code block and checks the numbers printed in their comments

Both modes refuse to run against the source tree: the point is to test the
built artifact, and a missing file in the wheel is invisible from the tree
because the tree still has it. `import nmd` must resolve outside this repo.

Every expected number is also grepped for in README.md as a literal, so that
editing one without the other goes red rather than passing quietly.

Sabotage-checked 2026-09-18, each of these turning it red rather than merely
not-green: perturbing an expected score (`0.437` -> `0.438`) fails the
measurement; deleting `regex` from the README's `nmd_word_set` dependency row
fails the README pin while the import check stays green, which is the drift the
pin exists for; and running `bare` in the full env fails the environment
assertion instead of passing vacuously. Run in an env with no wheel installed
it fails `nmd is importable` and stops, rather than falling back to the tree.
"""
import argparse
import importlib.util
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO, 'README.md')
WORDS_PATH = os.path.join(REPO, 'experiments', 'words_en.txt')

# numba is deliberately not here: nothing shipped imports it, so it is neither
# required by `full` nor fatal to `bare`. See HANDOFF's environment section.
OPTIONAL = ('numpy', 'scipy', 'pyroaring', 'regex')

# each row of the README's dependency table: module, the packages it lists
DEPENDENCY_TABLE = (
    ('nmd.nmd_index_v7', ('numpy',)),
    ('nmd.nmd_bow', ('scipy',)),
    ('nmd.nmd_segments', ('scipy', 'numpy')),
    ('nmd.nmd_word_set', ('pyroaring', 'regex')),
)

_failures = []
_readme = None


def check(ok, label, detail=''):
    print(f'{"ok  " if ok else "FAIL"}  {label}{"" if not detail else "   " + detail}')
    if not ok:
        _failures.append(f'{label}{"" if not detail else " -- " + detail}')
    return ok


def readme_says(snippet, label):
    """Pin an expected value to the README text, so the two cannot drift apart."""
    global _readme
    if _readme is None:
        _readme = io.open(README, encoding='utf8').read()
    return check(snippet in _readme, f'README still says {label}', repr(snippet))


def require_installed_not_source():
    try:
        import nmd
    except ImportError:
        return check(False, 'nmd is importable',
                     'install the built wheel into this env first -- see docs/release-dry-run.md')
    where = os.path.abspath(nmd.__file__)
    inside = os.path.normcase(where).startswith(os.path.normcase(REPO) + os.sep)
    return check(not inside, 'nmd imported from outside the repo', where)


def require_optional(present):
    """Fail loudly if the environment is not the one this mode needs.

    Without this, `bare` passes for the wrong reason in any env that happens to
    have numpy -- which is worse than not running it at all.
    """
    ok = True
    for module in OPTIONAL:
        found = importlib.util.find_spec(module) is not None
        if found != present:
            ok = check(False, f'{module} is {"present" if found else "absent"}',
                       f'this mode needs it {"present" if present else "absent"}') and ok
    return check(ok, f'environment is {"complete" if present else "bare"}')


def mode_bare():
    require_optional(present=False)
    if not require_installed_not_source():
        return  # everything below would just raise; the message above is the finding

    import nmd
    print(f'      nmd {nmd.__version__} from {nmd.__file__}')

    leaked = [m for m in OPTIONAL if m in sys.modules]
    check(not leaked, 'import nmd pulls in nothing third-party',
          f'leaked {leaked}' if leaked else '')
    check(nmd.__all__ == ('ngram_movers_distance', 'WordList'),
          'nmd.__all__ is the documented pair', repr(nmd.__all__))

    # --- the README quickstart block, verbatim ---
    from nmd import WordList
    word_list = WordList((2, 4), filter_n=0)
    for word in ('photographer', 'cartographer', 'choreographer', 'photography'):
        word_list.add_word(word)
    got = [(w, (round(a, 2), round(b, 2))) for w, (a, b) in word_list.lookup('fotografer')]
    expected = [('photographer', (9.9, 13.87)), ('cartographer', (7.95, 11.92)),
                ('photography', (7.76, 9.73)), ('choreographer', (5.91, 9.86))]
    check(got == expected, 'README quickstart output', f'got {got}' if got != expected else '')
    readme_says("# [('photographer', (9.9, 13.87)), ('cartographer', (7.95, 11.92)),",
                'the quickstart output')

    # --- the README ngram_movers_distance block, verbatim ---
    from nmd import ngram_movers_distance
    ngram_movers_distance('hello', 'yellow')
    ngram_movers_distance('hello', 'yellow', n=1)
    ngram_movers_distance('hello', 'yellow', invert=True)
    ngram_movers_distance('hello', 'yellow', normalize=True)
    check(0.0 <= ngram_movers_distance('hello', 'yellow', invert=True, normalize=True) <= 1.0,
          'README ngram_movers_distance block runs')

    # --- the README dependency table, checked by provoking each failure ---
    for module, packages in DEPENDENCY_TABLE:
        try:
            __import__(module)
            check(False, f'{module} needs one of {list(packages)}',
                  'imported with none of them installed')
        except ImportError as exc:
            check(getattr(exc, 'name', None) in packages,
                  f'{module} fails on a documented package',
                  f'ImportError on {getattr(exc, "name", None)!r}, table lists {list(packages)}')
        readme_says('| `' + module + '.', f'a dependency-table row for {module}')
        for package in packages:
            readme_says('`' + package + '`', f'{package} in the dependency table')


def mode_full():
    require_optional(present=True)
    if not require_installed_not_source():
        return  # everything below would just raise; the message above is the finding

    import nmd
    print(f'      nmd {nmd.__version__} from {nmd.__file__}')
    words = sorted({w.strip() for w in io.open(WORDS_PATH, encoding='utf8') if w.strip()})
    print(f'      {len(words)} words from experiments/words_en.txt')

    # --- the README WordList block ---
    from nmd import WordList
    word_list = WordList((2, 4), filter_n=0)
    for word in words:
        word_list.add_word(word)
    for query, word, scores, snippet in (
            ('fotografer', 'photographer', (9.90, 13.87),
             "# -> [('photographer', (9.90, 13.87)), ...]"),
            ('beaurocracy', 'bureaucracy', (11.73, 17.45),
             "# -> [('bureaucracy', (11.73, 17.45)), ...]"),
    ):
        top = word_list.lookup(query)[0]
        got = (top[0], (round(top[1][0], 2), round(top[1][1], 2)))
        check(got == (word, scores), f'WordList.lookup({query!r})[0]', f'got {got}')
        readme_says(snippet, f'the {query!r} result')

    # --- the README ApproxWordListV7 block ---
    from nmd.nmd_index_v7 import ApproxWordListV7
    v7 = ApproxWordListV7((2, 4))
    v7.add_words(words)
    ApproxWordListV7((2, 4), idf_exponent=2.0)  # the README's `weighted = ...` line
    top = v7.lookup('fotografer')[0]
    got = (top[0], round(top[1], 3))
    check(got == ('photographer', 0.437), "ApproxWordListV7.lookup('fotografer')[0]", f'got {got}')
    readme_says("# -> [('photographer', 0.437), ...]", "V7's 'fotografer' result")
    check(len(v7.lookup('fotografer', top_k=3)) == 3, 'V7 top_k=3 returns 3')
    check(bool(v7.lookup('sony camera dsc123', position_weight=0.0, denominator='geo')),
          'V7 position_weight / denominator knobs run')

    # --- the README bow_ngram_movers_distance block, verbatim ---
    from nmd.nmd_bow import bow_ngram_movers_distance
    score = bow_ngram_movers_distance(bag_of_words_1='Clementi Sports Hub'.casefold().split(),
                                      bag_of_words_2='sport hubs clemmeti'.casefold().split(),
                                      invert=True, normalize=True)
    check(0.0 < score < 1.0, 'README bow block runs', f'similarity {score:.4f}')

    # --- the README segment_movers_distance block, verbatim ---
    from nmd.nmd_segments import align_segments, explain_alignment, segment_movers_distance
    a = 'sony black earbud style headphones mdrex55bk'.split()
    b = 'ex series earbuds black mdr ex55/blk'.split()
    score = segment_movers_distance(a, b, lam=0.0, min_sim=0.2, invert=True, normalize=True)
    check(0.0 < score < 1.0, 'README segments block runs', f'similarity {score:.4f}')
    check(bool(explain_alignment(a, b, align_segments(a, b, lam=0.0, min_sim=0.2))),
          'explain_alignment returns an explanation')

    # --- the WordSet row of the README dependency table ---
    from nmd.nmd_word_set import WordSet
    word_set = WordSet()
    word_set.add('photographer')
    word_set.add('cartographer')
    check('photographer' in word_set and bool(word_set.find_similar('fotografer')),
          'WordSet add / __contains__ / find_similar run')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('mode', choices=('bare', 'full'))
    mode = parser.parse_args().mode

    print(f'release_check {mode}: python {sys.version.split()[0]}')
    {'bare': mode_bare, 'full': mode_full}[mode]()

    print()
    if _failures:
        print(f'{len(_failures)} FAILED:')
        for line in _failures:
            print(f'  * {line}')
        return 1
    print(f'release_check {mode}: everything the README promises holds.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
