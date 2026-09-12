# Optional assistant integrations

The Assistant tab asks an installed Codex, Claude Code, or compatible Antigravity runtime for advice about images, depth, masks, and mesh settings. It does not apply generated settings, execute a returned command, download a model, or change a project. Responses may be wrong; inspect the proposal and use the normal controls to apply a change.

Choose a provider and its installed executable, choose account sign-in or an API key, and press **Connect / check status**. Sign in if needed. Write the question and press **Enter** to send; **Shift+Enter** inserts a line. **Escape** cancels the active request when this panel has focus. All controls have keyboard access and accessible names.

The prompt is sent to the selected provider. **Include current settings and mesh summary** starts off and must be enabled explicitly. **Preview context** shows the exact metadata supplied by the application. The panel never attaches image pixels or project files. It starts requests in an empty temporary directory. **Clear conversation** removes the visible prompt and response; each new request starts a fresh conversation.

## Authentication and compatibility

| Runtime | Account sign-in | API key | Required capability |
| --- | --- | --- | --- |
| Codex | Native app-server browser sign-in; EdgeMesh uses a separate Codex home and requires the OS keyring | Sent to app-server over stdin, with its in-memory credential store | Stdio app-server, account API, and restricted read-only sandbox policy |
| Claude Code | Native `auth login` and `auth status`; uses Claude Code's authentication | Session environment; minimal API-key mode prevents an account fallback | Restricted/safe modes, empty tool list, streaming JSON |
| Antigravity | A separate native sign-in window on Windows; credentials remain managed by Antigravity | Optional Antigravity SDK with the session's Gemini API key | Account mode needs compatible streaming CLI; key mode needs the `assistants` package extra |

No account token is copied between providers. Account access does not grant access to a separately billed media API. Provider subscription limits, account eligibility, model access, and API charges still apply. The editable model field must name a model available to that runtime/account; Codex initially uses `gpt-5.6-terra`.

EdgeMesh does not save entered API keys in preferences, project files, or application logs. A key can also be supplied through the selected provider's environment variable before starting EdgeMesh. **Forget key**, changing provider, or shutting down releases the panel's retained session credentials and connection. Do not put keys in prompts, model identifiers, executable paths, project files, or command arguments. Native runtime credential handling and the selected provider's retention policies also apply.

Codex protocol and credential storage are documented in [Codex App Server](https://learn.chatgpt.com/docs/app-server) and the [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference). Claude's supported modes and authentication commands are in its [CLI reference](https://code.claude.com/docs/en/cli-reference).

Antigravity's [installation and authentication guide](https://antigravity.google/docs/cli/install/) describes account/key access. Its [headless protocol](https://antigravity.google/docs/cli/headless/) and [permission rules](https://antigravity.google/docs/cli/permissions/) define the integration boundary. The panel uses an assistant-owned home under the EdgeMesh user-data directory, containing a nonsecret deny-all policy. It does not modify the user's normal Antigravity configuration. Existing incompatible CLIs show an update-required message and do not receive prompts. Native account sign-in remains available.

Antigravity API-key mode uses the official SDK and its bundled runtime, independently of the installed CLI. Install EdgeMesh with its `assistants` extra to include `google-antigravity==0.1.16`; the panel never installs it automatically. This pinned release was published September 2, 2026, more than seven days before verification. See the [PyPI release](https://pypi.org/project/google-antigravity/0.1.16/) and [SDK policies](https://antigravity.google/docs/sdk/policies/). It has an empty tool list and an explicit deny-all policy, with no MCP servers, skills, hooks, or subagents. Empty SDK session storage selects ephemeral trajectories; other runtime data is confined to the temporary request directory. The pinned Python SDK sends model configuration, including the key, over an authenticated loopback connection to its bundled runtime; the key is not a command argument. This does not establish provider-side retention or inspect every native-runtime storage path. The executable selector applies only to account mode for this provider.

On non-Windows systems, the panel currently directs Antigravity account sign-in to the native runtime; its SDK key mode and the other providers use the same in-app flow. Native Antigravity sign-in must use the assistant-owned home so it sees the same profile. An older native CLI can still be used interactively in the sign-in window, with its normal permission prompts; it receives no image, project files, or in-app prompt automatically.

## Runtime boundaries

Codex requests use an ephemeral conversation in an empty request directory with a read-only filesystem policy and sandbox network access disabled. Shell/exec, local images, image generation, browser/computer use, apps, MCP, plugins, skill discovery, memory, hooks, goals, and automation are disabled. The panel rejects all server requests to execute a host tool or approve an action. Claude requests use no built-in tools, no MCP servers, and restricted/safe operation. Antigravity requests require a strict, deny-all policy for file, command, browser, and MCP actions. There is no permission-bypass flag or automatic approval path.

Requests have a five-minute timeout, bounded prompt/output sizes, and JSON protocol validation. On Windows, an isolated Python supervisor joins a Job Object before launching a request runtime, so cancellation also owns children created by CLI and virtual-environment launchers. Cancellation stops the panel's owned request processes and descendants; it never kills another independently launched assistant. Native account-login windows are closed independently, so a browser opened for sign-in is not terminated as a request descendant. Runtime errors are logged as fixed diagnostic categories, without raw provider diagnostics, prompts, keys, or responses. The user-visible response is plain text, so HTML and links from an assistant are not executed.

**Check JSON proposal** accepts a small object with an explanation string and a settings object containing simple values. This validates its data structure only; it does not establish that setting names, ranges, or recommendations are correct, and it never applies the values.

Provider, executable, auth mode, model, splitter position, runtime target directory, and context-preview geometry are saved in `assistant.json` in EdgeMesh's user-data directory. Keys, prompts, responses, and the context opt-in are not persisted. Tests can inject `settings_path` and `state_dir` to keep application state isolated.

## Verification record

The local installed CLI help and Codex-generated protocol schemas were inspected on 2026-09-11. The installed Antigravity executable did not expose the streaming protocol required for in-app requests; its normal sign-in can still be opened, but prompts remain blocked until a compatible runtime is installed. No CLI was installed or upgraded by this change.

Focused tests cover simulated protocol handshakes, streaming completion, malformed messages, explicit context, credential non-persistence, npm shim resolution without a command shell, and cancellation. These tests do not spend provider credits or verify live account/model access. Live authenticated requests require the user's provider sign-in and available quota.
