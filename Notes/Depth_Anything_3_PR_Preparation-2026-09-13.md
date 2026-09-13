# Depth Anything 3 PR Preparation

Date: 2026-09-13 12:45 (America/Chicago).

The completed feature adds four optional Depth Anything 3 models with a separate runtime, checkpoint verification, cancellation and offline generation. It includes the inverse-depth correction and the saved-layout native startup repair.

Final validation: 355 tests passed with two native GUI tests skipped in the headless suite. Both native tests passed separately. The previous MeshTools suite passed 11 tests. Ruff fatal-error checks passed on touched Python files. Mypy passed on da3_backend.py, da3_worker.py, da3_setup.py and data_contracts.py. Wheel and source archive builds passed. Real DA3 Small CPU inference, OBJ/STL export and saved-project native startup passed. Other DA3 variants and CUDA have not been tested with real inference.

The version-manager dry run proposes 1.0.1 to 1.1.0. Curated notes cover the DA3 feature and the two fixes.

Leland clarified that requested PRs use the configured GitHub Claude Code review. The agent opens the PR after local checks and the version bump, then waits for the review comment, normally 2-5 minutes. A separate local Claude CLI review is not a prerequisite. The shared global rules, Codex global rules, delegation guidance and this repository's matching rule were updated. No code diff was sent through the blocked local Claude review command.

The main checkout's unrelated work remains separate. This feature is committed from the isolated codex/depth-anything-3 worktree.
