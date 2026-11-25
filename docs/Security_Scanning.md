# Security scanning — procedures and reasons

Detail moved out of `AGENTS.md` -> "Scanning & review" so that it loads only
during scanner or Dependabot work. Policy is in `~/.config/agents/AGENTS.md`
-> "Dependencies, advisories, and scanning" and
`~/.claude/instructions/security-scanning.md`.

- SHA-pinned actions get **no Dependabot alerts**: GitHub generates them only
  for actions on semantic versions (docs: "About Dependabot alerts"). So no
  security-update PR can ever fire for a pinned action, an
  `applies-to: security-updates` group over github-actions is dead config, and
  the weekly version update is their only automatic remediation -- which is why
  the github-actions block carries no semver-major ignore. DETECTION comes
  from the `zizmor` job in `security.yml`: it maps each SHA pin to its
  release tag and fails on a known-vulnerable, impostor or unpinned action.
  osv-scanner's `github/actions` plugin only lists the pinned actions: with
  osv-scanner 2.7.0 it did not flag `tj-actions/changed-files`
  (GHSA-mrrh-fwg8-r2c3) at a vulnerable SHA or tag. zizmor looks up a SHA
  pin by one tag: for a root action (owner/repo@sha) the longest tag
  without a `/`, for a sub-action the longest tag under its path or its
  last directory (else the longest of all). The job fails unless that tag, and any tag tied
  with it, is a plain release version (v1, v1.2.3): an untagged commit,
  a non-version tag or a path-prefixed tag matches no advisory. Pin
  released versions.
- Dependabot **alerts** are a per-repo GitHub setting; a config file does not
  turn them on. Without them the config above governs version updates only.
  Check it by HTTP STATUS, not by exit code -- `gh api` exits 0 on success
  and 1 on any failure, and never exits 404 or 403:
  `gh api -i -X GET repos/lelandg/EdgeMesh/vulnerability-alerts`
  204 = on. A 403 is inconclusive until you read the response: a body that
  mentions a rate limit, a `retry-after` header, or
  `x-ratelimit-remaining: 0` means a primary or secondary rate limit (wait,
  then retry); only a 403 with none of those, from a token whose
  `.permissions.admin` (below) is not `true`, means the token lacks admin.
  A 404 means off only when
  `gh api repos/lelandg/EdgeMesh --jq .permissions.admin`
  prints `true`; otherwise the token cannot see the setting. Neither of those
  is the same as off. Enable: same path with `-X PUT`.
- Alerts and automatic security **updates** are two switches, not one. Alerts
  only report; the half that opens PRs is `automated-security-fixes`. Its
  API answers differently: `gh api repos/lelandg/EdgeMesh/automated-security-fixes`
  returns 200 with `{"enabled": ..., "paused": ...}` (updates are on only
  when `enabled` is true and `paused` is false), or 404 when they are
  off. A 404 is also what a token without admin sees, so read it as off
  only when `.permissions.admin` (above) is `true`. Enable with `-X PUT`. It was left OFF while this repo's first-scan advisory backlog was
  open, because enabling it would have opened that backlog as PRs at once.
  PR #6 cleared the backlog, so it is due to be switched on.

Checked 2026-11-25: `gh api repos/lelandg/EdgeMesh/automated-security-fixes`
returned 200 with `{"enabled":false,"paused":false}` from an admin token, so
automatic security updates are still off.
