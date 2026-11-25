@AGENTS.md

# CLAUDE.md — Claude Code specifics

The line above imports `AGENTS.md`, the canonical guide for every agent. This
file adds only the Claude Code mechanisms that implement those conventions.
Copilot also loads this file. Every rule is in `AGENTS.md`; an agent that is
not Claude Code follows the script or file path given in each bullet.

- **Date:** read "Today's date" from the `<env>` block before writing a date.
- **Version and landing:** the `land-pr` skill (`/land-pr land lelandg/EdgeMesh#N`)
  cuts the bump at merge time; `/version-manager check` reads the state. Script
  paths: `AGENTS.md` -> "Version and changelog".
- **Code map refresh:** the `update-code-map` skill. No standalone script
  exists; the procedure is `~/.claude/skills/update-code-map/SKILL.md`.
- **Code review:** the built-in `/code-review` skill for the structured
  review. The review council runs through the runner in
  `~/.claude/instructions/model-delegation.md`, never through the `/codex:*`
  plugin commands.
- **Dependency audit:** the `scan-source` skill (a Claude Code plugin; no path
  for other CLIs, which use the scanners in `AGENTS.md` -> "Scanning &
  review"). It does not skip `.venv` directories, so pass a narrow path.
- **Options for Leland:** the `AskUserQuestion` tool.
- **Docs and Notes:** `docs/` (lowercase) and `Notes/`, per `AGENTS.md`.
