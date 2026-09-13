# DA3 PR review corrections

Recorded: 2026-09-13 13:46 (America/Chicago).

Addressed Claude's review on PR #2 in its existing feature branch. Preserved
CL Auto-Fix commit `23b588f` from draft PR #3 through a fast-forward. No PR was
merged into main. PR #3 targets main and contains the original feature plus
the documentation fix; its draft status does not block updating PR #2.

- Replaced machine-specific packaged setup/launch commands with portable source
  and installed-package commands. Removed placeholders from copyable commands.
- Moved the prepared-preview instructions into the existing Notes record and
  described the ignored render as a local artifact instead of a broken link.
- Added DA3 backend, worker and setup rows to the CodeMap module table.
- Marked the fixed-image validation helper as a one-off reproduction, restored
  its trailing newline and removed an unused import found by lint.
- Added a packaging regression for portable commands and matching copy buttons.
- Extended the real Windows startup test to verify restored window size,
  maximized state and Parameters dock placement/visibility, with an independent
  saved configuration copy that startup cannot overwrite.

Validation: 12 packaging/native tests passed, including both opt-in native
startup tests. Ruff and scoped mypy passed for all three touched Python files.
Both documented module entry points returned help successfully. Wheel and
source archive built with the existing project setuptools environment. The
wheel's setup HTML matches source byte for byte and contains no developer paths.
See [test log](DA3_PR_Review_Tests-2026-09-13.txt).

The first lint run identified the unused import, which was removed. The GUI
venv lacks the optional build frontend; packaging succeeded through setup.py
with the installed setuptools backend. No dependencies were installed.

Remaining review dispositions: keep the explicitly requested review-policy
change in its existing separate commit. A fully hashed DA3 runtime dependency
lockfile is a future hardening suggestion, not the review's blocking finding.
First-download checkpoint trust remains consistent with the existing model path.

Windows CI remains unresolved. Both base and PR runs stop before unittest emits
its final tracebacks; new DA3 errors coincide with imports of depth_to_3d but
the precise cause is unproven. Next diagnostic: run that import or DA3 tests
separately in CI before the full UI suite to retain the traceback. Vercel is
not an intended deployment target, per Leland's comment on PR #3.

The feature keeps its existing 1.1.0 release bump while this open PR is revised.
