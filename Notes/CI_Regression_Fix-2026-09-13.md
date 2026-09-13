# Windows CI regression fix

Recorded: 2026-09-13 14:32 (America/Chicago).

## Scope and cause

Branch: `codex/fix-windows-ci`, based on merged `origin/main` at `5f2abea`.
The separate MiDaS/DPT checkout and the merged DA3 worktree were preserved.

The failing PR 2 run was
https://github.com/lelandg/EdgeMesh/actions/runs/34772775101.
Checkout and installation passed. Application imports and pipeline tests failed.

The CI profile omitted Shapely and PyVista, although both were declared in
the application metadata. PyVista also requires Pooch and Scooby. All four
were absent from the 91-package profile. Metadata traversal added the four
missing packages without changing an existing pin.

The new packaging regression test failed on both omitted application roots
before the profile correction and passed afterward. Installing EdgeMesh
without its dependencies reproduced the missing requirements through `pip check`:

```text
edgemesh 1.1.0 requires pyvista, which is not installed.
edgemesh 1.1.0 requires shapely, which is not installed.
```

After installing the completed profile, `pip check` reported no broken
requirements. CI now installs application metadata before that check and runs
an early mesh/depth import check. The preflight uses EdgeMesh's existing Windows
runtime guard. Faulthandler and fail-fast unittest execution preserve useful
diagnostics if another failure occurs.

## Validation

- Fresh CPython 3.12.10 environment: 95 pinned packages plus EdgeMesh.
- uv installed the pins with dependency resolution disabled and a seven-day
  package-age cutoff. pip built and installed EdgeMesh without dependency
  resolution or build isolation.
- `pip check`: passed after the fix; failed for both missing roots before it.
- Guarded mesh/depth imports: passed.
- EdgeMesh: 357 tests, OK, two opt-in native GUI tests skipped.
- MeshTools: 11 tests, OK.
- Scoped Ruff fatal-error checks, Python compilation, and workflow YAML parse:
  passed. Scoped mypy also passed. Independent read-only review found no actionable defects.

The GitHub folder-memory failure did not reproduce in the complete fresh
environment. It also passed alone, with its test module, and with an 8.3 TEMP
parent. No speculative persistence change was made.

An initial sandboxed run hit local log-file permissions. The normal Windows
run passed. An unguarded import probe also reached the known CPython WMI fault;
the final preflight invokes the existing application guard first.

## Delivery boundary

No source behavior or dependency version already pinned by main was changed.
The user authorized commit, push, and PR creation after local validation.
Hosted CI and the configured automated review remain the next verification gates.
A fresh GitHub run must verify the hosted result before calling CI green.


## Hosted path failure and automated correction

PR 4 run 34780057607 passed dependency installation, application installation,
`pip check`, and both imports. Fail-fast then exposed the earlier hidden assertion:
Qt returned `C:/Users/runneradmin/...` while the test expected the equivalent
`C:/Users/RUNNER~1/...`. This was a test spelling assumption, not a persistence
failure. The reviewer on Ubuntu EC2 independently confirmed the cause.

Draft PR 5 supplied commit `a6170e96c2c835f8796b3c3e25245cb60727cb3b`.
It resolves temporary test roots once and consistently uses those roots in
mask/settings and dialog-persistence tests. This commit is integrated into PR 4
with its original author and history. The equivalent manual assertion edits
were set aside in favor of the reviewed automation. Their focused test logs
record 15 mask and 22 persistence tests passing.

The final correction uses cross-platform `Path.resolve()`. It adds no Windows
path literals or Windows-only API calls to the tests. Product code is unchanged.
The exact integrated code passed all 357 tests (two native GUI tests skipped);
see `CI_Autofix_Tests-2026-09-13.txt`. Scoped Ruff and mypy also passed.
A workflow comment distinguishes source imports from isolated installed-copy
subprocess tests. Hosted verification and renewed review are still pending.

## Remaining path fixtures and test-process initialization

The next hosted run, 34780726729, passed 248 tests before finding the same path
spelling assumption in `test_project_store`. Its temporary root, and the roots
in `test_user_state` and `test_workflow_ui`, now use `Path.resolve()` as well.
These are portable test-fixture changes. No product paths were changed.

A local Windows-only reproduction harness generated and verified a real 8.3
alias with `GetShortPathNameW`. The project-folder assertion failed before the
fix, then all 35 tests in those three modules passed under that alias. The
harness stays in the ignored local environment; it is not required on Ubuntu.
See `CI_Project_Alias_Before-2026-09-13.txt` and
`CI_Path_Modules_Alias_Tests-2026-09-13.txt`.

An experimental whole-suite alias run aborted natively at a cancellation test.
A separate normal discovery run exposed the known CPython 3.12 WMI fault during
Trimesh import and then a native process-start failure. Neither run passed.
The workflow previously applied the existing runtime guard only in its import
preflight. That separate process cannot initialize the test process.
Both test steps now invoke the guard before unittest discovery. This is a no-op
on Ubuntu, and fresh-subprocess startup regression tests remain independent.

The final guarded EdgeMesh command passed all 357 tests, with two opt-in GUI
tests skipped. The guarded MeshTools command passed all 11 tests. Scoped Ruff
and mypy passed for the three additional fixtures. Workflow YAML parsing and
`git diff --check` passed.
The full alias experiment is not claimed as successful; the targeted alias test
and the complete guarded suite are the validation evidence.
