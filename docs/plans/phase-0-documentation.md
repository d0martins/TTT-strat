# Document Phase 0 with Sphinx, published to GitHub Pages

## Context

Phase 0 of the TT pacing optimizer (`course.py`, `wind.py`, `rider.py`, `physics.py`, `w_prime/`,
`simulator.py`) is fully implemented and already carries NumPy-style docstrings on every public
module, class, and function — enforced by `ruff`'s `D` rule family (`convention = "numpy"`), which
currently passes clean (`ruff check src/` → all checks passed). None of that documentation is
published anywhere; the only entry points today are reading source or the implementation-plan spec
in `docs/`. The goal is to auto-generate an API reference site from the existing docstrings and
host it on GitHub Pages, so the physics/model API is browsable without opening the source.

Decisions already made with the user:
- **Tool**: Sphinx (`autodoc` + `napoleon`), the standard for NumPy-style scientific-Python
  docstrings, over MkDocs.
- **Scope**: auto-generated API reference + one hand-written architecture-overview landing page
  (adapted from `docs/tt-pacing-optimizer-implementation-plan.md`). No rendered notebook.
- **Deployment**: GitHub Actions workflow, auto-deploying on every push to `main`. Repo remote is
  `https://github.com/d0martins/TTT-strat` — Pages must be switched to "GitHub Actions" source
  once in repo settings (manual, one-time; flagged in verification section).

## Known risk to handle explicitly

`physics.py`'s functions are decorated `@numba.njit(cache=True)`, which wraps them in a
`CPUDispatcher` object. Sphinx `autodoc` introspects `inspect.signature()` on the object it's given;
dispatcher objects don't reliably expose the original signature/docstring the way plain functions
do. Mitigation: add an `autodoc-process-signature` (and if needed `autodoc-process-docstring`) hook
in `conf.py` that detects Numba-wrapped callables (`hasattr(obj, "py_func")`) and substitutes
`obj.py_func` before Sphinx inspects it. This must be verified by actually building the docs and
inspecting the rendered `physics` module page (see Verification).

## File layout to add

```
docs/sphinx/
├── source/
│   ├── conf.py
│   ├── index.md              # landing page: project summary + links
│   ├── architecture.md       # adapted from implementation-plan Architecture/Module Map/
│   │                          # naming-convention sections, scoped to Phase 0 modules only
│   └── api/
│       ├── index.md          # toctree of module pages below
│       ├── course.md
│       ├── wind.md
│       ├── rider.md
│       ├── physics.md
│       ├── simulator.md
│       └── w_prime.md        # covers w_prime/__init__.py Protocol + linear/skiba/bartram/
│                              # differential/caen submodules
└── build/                    # gitignored; CI builds fresh each run
.github/workflows/docs.yml
```

Rationale for `docs/sphinx/` rather than dropping Sphinx source directly into `docs/`: the existing
`docs/` directory already holds the narrative spec markdown files, PDFs, and course images at its
top level — nesting the Sphinx project avoids mixing generated-site source with those reference
documents.

Use **MyST markdown** (`myst-parser`) for `index.md`/`architecture.md`/`api/*.md` instead of `.rst`,
so the narrative pages stay consistent in format with the rest of `docs/` and content can be lifted
from `tt-pacing-optimizer-implementation-plan.md` with minimal translation. `automodule` directives
work fine inside MyST via the `eval-rst` fence or MyST's native directive syntax.

## Steps

1. **Add a `docs` extra to `pyproject.toml`**:
   `sphinx`, `furo` (theme), `myst-parser`. Keep it separate from the existing `dev` extra since CI
   for docs doesn't need `pytest`/`cyipopt`.

2. **Write `docs/sphinx/source/conf.py`**:
   - `extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.mathjax", "myst_parser"]`
   - `napoleon_numpy_docstring = True`, `napoleon_google_docstring = False`
   - `html_theme = "furo"`
   - `autodoc_typehints = "description"` (keeps signatures readable given how parameter-heavy
     `physics.py`/`simulator.py` functions are)
   - The Numba-dispatcher signature/docstring workaround described above, wired via
     `app.connect("autodoc-process-signature", ...)` / `autodoc-process-docstring`.
   - `sys.path` insertion pointing at `../../../src` so `ttt_strat` is importable without an
     editable install being required in the docs build step (still install the package properly in
     CI, this is just a safety net).

3. **Write the API pages** (`docs/sphinx/source/api/*.md`), one per module, each using
   `automodule:: ttt_strat.<module>` with `:members:` and `:undoc-members:` (the latter should end
   up unused since everything's already documented, but catches regressions). `w_prime.md` documents
   the `WPrimeModel` Protocol from `w_prime/__init__.py` plus the five model classes
   (`LinearModel`, `SkibaModel`, `BartramModel`, `DifferentialModel`, `CaenModel`) each via their own
   `automodule`/`autoclass` block.

4. **Write `architecture.md`**: condense the "Architecture Overview" diagram, the Phase-0 slice of
   the "Module Map" (`course.py`, `wind.py`, `rider.py`, `physics.py`, `w_prime/`, `simulator.py`
   only — leave out unimplemented later-phase modules), and the "Naming convention — unit suffixes"
   section verbatim from `docs/tt-pacing-optimizer-implementation-plan.md:22-123`, since that
   convention is essential context for reading the API reference.

5. **Write `index.md`**: short project summary (from the implementation plan's Context section) +
   a toctree linking `architecture` and `api/index`.

6. **Add `.github/workflows/docs.yml`**:
   - Trigger: `push` to `main` (paths filter on `src/ttt_strat/**`, `docs/sphinx/**` optional but
     keeps CI minutes down), plus `workflow_dispatch` for manual reruns.
   - Job: checkout → setup Python 3.12 → `pip install -e '.[docs]'` → `sphinx-build -b html
     docs/sphinx/source docs/sphinx/build/html -W` (the `-W` turns warnings, e.g. a broken autodoc
     import, into a hard CI failure so silent breakage doesn't ship) → `actions/upload-pages-artifact`
     with the build dir → `actions/deploy-pages`.
   - Standard Pages permissions block: `permissions: {pages: write, id-token: write}`, environment
     `github-pages`, and `concurrency` group to avoid overlapping deploys.

7. **Update `.gitignore`**: add `docs/sphinx/build/`.

## Verification

1. Local build: `pip install -e '.[docs]'` then
   `sphinx-build -b html docs/sphinx/source docs/sphinx/build/html -W` — must exit 0 with no
   warnings.
2. Open `docs/sphinx/build/html/api/physics.html` specifically and confirm `rolling_force_N`,
   `aero_force_N`, `grav_force_N`, `dv_ds`, `dw_ds` all show correct parameter lists and rendered
   docstrings (this is the Numba-dispatcher risk called out above — if signatures show up as
   `(*args, **kwargs)` or docstrings are missing, the `conf.py` hook needs fixing before moving on).
3. Spot-check `api/w_prime.html` renders the `WPrimeModel` Protocol and all five model classes.
4. Push the workflow to a branch, open a PR, and confirm the Actions run builds successfully (it
   won't deploy until merged to `main` given the trigger).
5. After merging to `main`, in GitHub repo Settings → Pages, set Source to "GitHub Actions" (one-time
   manual step — flag this to the user rather than doing it via `gh api` without confirmation, since
   it's a shared-repo setting change). Confirm the published URL
   (`https://d0martins.github.io/TTT-strat/`) loads and matches the local build.
