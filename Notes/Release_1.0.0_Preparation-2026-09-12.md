# EdgeMesh major-release commit preparation

Prepared: 2026-09-12 07:55 (America/Chicago).

The requested release target is **0.6.7 to 1.0.0**. This is preparation for a
future commit while **Fix project mesh and dialogs** works in the same checkout.
No staging, commit, tag, push, version change or changelog insertion is part of
this preparation. The active task's application and UI files remain its scope.

## Prepared material

- [Curated release-note body](Release_1.0.0_Notes.md), suitable for the version
  manager's notes input after the feature work has been committed and reviewed.
- [Feature commit-message draft](Release_1.0.0_Commit_Message.txt). Add the final
  bug-fix result and current validation evidence when the combined work is ready.
  Add AI attribution at commit time only from verified session metadata; none was
  guessed in this draft. The version manager makes a separate release commit.
- Packaging preparation aligns README references with Git's `README.md` spelling
  and checks the Torch/torchvision extra against the CPU constraint pins. Runtime
  dependency versions are unchanged.

The release-note body describes implemented scope. It deliberately contains no
claim that SAM3, Depth Anything 3 inference, source image/video generation or full
360-degree reconstruction has shipped. It does not claim public distribution,
authenticated assistant responses, frozen builds, or new platform validation.

## Working-tree snapshot

| Item | Observed state |
| --- | --- |
| EdgeMesh branch | `codex/codebase-review-fixes` |
| HEAD | `29ed31ee10e90d017c5a8c32dc3999ea6bad8486` |
| Cached `origin/main` | Same commit as HEAD; no fetch was performed |
| Commits in cached `origin/main..HEAD` | None |
| Index | No staged paths when inspected |
| Current version | `version.py`: `0.6.7`; `_version.py` re-exports it |
| Packaging version | Dynamic `version.__version__`; no separate static version |
| MeshTools gitlink | `51c4adb6b388ded455a6a649cc6b85b083dfb424` |
| MeshTools branch | `codex/mesh-topology-fixes` |
| MeshTools working changes | `viewport_3d.py`, `tests/test_mesh_topology.py` |
| Code map | Updated 2026-09-11 14:32; current under the seven-day rule |

The substantial release work is still uncommitted, including many new runtime,
test, documentation and packaging files. A tracked-only diff is not a complete
commit inventory. Recheck the live status after the other task finishes.

## Commit scope

| Disposition | Scope |
| --- | --- |
| Feature candidates | Workspace, project/session persistence, masks, history, viewport, generation jobs, data contracts, mesh health, model handling/licensing, assistants, and the existing depth/startup fixes, with their tests |
| Packaging candidates | `pyproject.toml`, `setup.py`, `_build_support.py`, `MANIFEST.in`, `edgemesh_bootstrap/`, dependency profiles, CI workflow, README and packaging guides |
| Documentation candidates | The matching `docs/` guides/assets, `Plans/` checklists and this task's release-preparation files |
| Separate repository | Commit approved MeshTools changes within MeshTools, then update EdgeMesh's gitlink; the parent cannot include its uncommitted file contents |
| Preserve pending scope review | Pre-existing `Notes/` reports, assistant-review outputs, depth diagnostics and `.scan-reports/`; do not stage them wholesale |
| Leave local | User state, model weights, build output, caches, credentials and machine-local configuration |

The file-disposition registry specifically preserves pre-existing EdgeMesh Notes
and local settings edits pending separate authorization. No disposition was
changed or written to that registry here. The new release-preparation notes are
deliverables from this task and are candidates for the eventual approved commit.

## Version-manager checks

The installed tool identifies `version.py` as canonical and finds five commits
since `v0.6.7`. It also reports `0.6.6` missing from the structured changelog ledger.
That version already exists as a legacy bullet below the structured entries in
`docs/CHANGELOG.md`; this is a historical format discrepancy, not a reason to
invent release history. The repository adopted the tool in commit `d9cd194`, so
one-time backfill was not repeated and historical tags were not modified.

The actual `release major` dry run exits 1 with:

> REFUSED: working tree is dirty — commit or stash first

The [version-manager skill](../../.claude_code/claude/skills/version-manager/SKILL.md)
states, “release refuses on a dirty tree,” and its implementation checks that
condition before producing the release preview. Its apply path updates versions
and changelog, runs `git add -A`, commits and creates a tag. The clean-tree check
was not bypassed and the active task's work was not stashed. No successful
release-plan preview or application is claimed.

After the whole feature is ready, the ordinary commit/review gates precede the
version-manager dry run and curated major release. A release application is a
commit/tag operation, not merely a file edit.

## Validation during this preparation

The first focused packaging run used the default Python 3.12 installation with
setuptools 75.8.2: six passed and three failed. Two failures were metadata parsing
errors caused by that old setuptools not accepting the declared SPDX license
format; the project requires setuptools 83.0.0. No global package was installed
or upgraded.

The pre-fix rerun used the existing project `.venv` (Python 3.12.10, setuptools
83.0.0): eight passed and one failed. The remaining failure expected
`torch==2.8.0`, whereas both the depth extra and CPU constraint profile now pin
`torch==2.13.0` and `torchvision==0.28.0`. The assertion needed to track the shared
dependency contract rather than assert an obsolete version.

Git and the actual directory both use `README.md`; package metadata, manifest,
cache key and test expectations used `ReadMe.md`. Windows accepted the old
spelling, but it disagreed with a case-sensitive checkout. Packaging references
are aligned with the existing tracked filename without renaming it.

Final focused verification passed: all **9 packaging tests in 20.408 seconds**
with the project Python environment, exit 0. Scoped Ruff passed for
`tests/test_packaging.py`, `setup.py` and `_build_support.py`, exit 0.
Independent readback verified all four packaging edits, unchanged dependency
pins, three saved release artifacts, valid local document links, clean artifact
whitespace, parsed packaging Python/TOML, and canonical README metadata.
`version.py` and `docs/CHANGELOG.md` have no content diff, and the index remains
unstaged.

The project has no configured type-checker gate; syntax checks do not constitute
a project-wide typecheck. The full application suite and packaged build are
reserved for the combined final state after the active bug task finishes.

## Remaining release gates

- Incorporate the completed project/dialog investigation and validate the final
  combined source; earlier successful test totals and package hashes predate it.
- Resolve the documented cross-provider authentication review gate. The existing
  [desktop completion record](Desktop_Completion-2026-09-11.md) says a previous
  automatic approval review rejected sending the selected private sources to
  Claude. This preparation made no provider request or approval retry. The
  bounded local review recorded there is not a completed Claude review.
- Review the final file list, including new files, and the separate MeshTools
  changes. Do not use a blanket stage operation on the present working tree.
- Run scoped lint, relevant tests and the final build/review pass on the settled
  feature. Rebuild and inspect release artifacts for the final version before
  claiming they match it; current `dist/` artifacts still carry the older version.
- Recheck upstream history before any eventual push. Publishing the submodule
  commit must precede publishing a parent gitlink that depends on it.

These gates describe the remaining commit/release pass; they are not claims that
the current preparation performed it.

Preparation verified: 2026-09-12 08:08 (America/Chicago).
