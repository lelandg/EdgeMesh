@AGENTS.md

# CLAUDE.md — Claude Code specifics

The line above imports `AGENTS.md`, the canonical guide for every agent. This
file adds only the Claude Code mechanisms that implement those conventions.

- **Date:** read "Today's date" from the `<env>` block before writing a date.
- **Version bump:** the `version-manager` skill (`/version-manager release <level>`).
- **Code map refresh:** the `update-code-map` skill.
- **Code review:** the `code-reviewer` agent for the structured review; then
  `/codex:review --base origin/main` for the independent second-family pass
  (read-only, committed work only). That command inherits the model in
  `~/.codex/config.toml`, which is GPT-6 Astra; Sol is review-only and is
  reachable only by editing that file.
- **Dependency audit:** the `scan-source` skill. It does not skip `.venv`
  directories, so pass a narrow path.
- **Options for Leland:** the `AskUserQuestion` tool.
- **Docs and Notes:** `docs/` (lowercase) and `Notes/`, per `AGENTS.md`.
