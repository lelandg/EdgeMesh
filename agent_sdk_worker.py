"""Optional Antigravity API-key worker with no tools and no secret arguments."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys

SDK_VERSION = "0.1.16"
MAX_INPUT = 64_000
MAX_REQUEST = MAX_INPUT * 4 + 4096
MAX_OUTPUT = 900_000


def _emit(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def load_sdk():
    """Probe the pinned optional dependency without installing or authenticating."""
    from edgemesh_bootstrap.runtime import initialize_platform_runtime

    initialize_platform_runtime()
    if importlib.metadata.version("google-antigravity") != SDK_VERSION:
        raise ValueError("Unsupported SDK version")
    from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig
    from google.antigravity.hooks.policy import deny
    return Agent, CapabilitiesConfig, LocalAgentConfig, deny


def build_config(sdk, *, model="", api_key="", directory=None):
    _, CapabilitiesConfig, LocalAgentConfig, deny = sdk
    directory = Path(directory or Path.cwd()).resolve()
    options = {
        "api_key": api_key, "vertex": False,
        "capabilities": CapabilitiesConfig(enabled_tools=[], enable_subagents=False),
        "policies": [deny("*")], "tools": [], "mcp_servers": [],
        "hooks": [], "triggers": [], "skills_paths": [], "subagents": [],
        "workspaces": [str(directory)],
        "app_data_dir": str(directory / "runtime"), "save_dir": "",
    }
    if model:
        options["model"] = model
    return LocalAgentConfig(**options)


async def run_request(payload, sdk):
    if not isinstance(payload, dict) or set(payload) != {"prompt", "model"}:
        raise ValueError("Malformed request")
    prompt, model = payload["prompt"], payload["model"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > MAX_INPUT:
        raise ValueError("Invalid prompt")
    if not isinstance(model, str) or len(model) > 200:
        raise ValueError("Invalid model")
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or len(key) > 4096:
        raise ValueError("Missing API key")
    config = build_config(sdk, model=model, api_key=key)
    _emit({"event": "init", "init": {"permission_mode": "strict", "tools": []}})
    count = 0
    async with sdk[0](config) as agent:
        response = await agent.chat(prompt)
        async for token in response:
            if not isinstance(token, str):
                raise ValueError("Invalid stream token")
            count += len(token.encode())
            if count > MAX_OUTPUT:
                raise ValueError("Response too large")
            _emit({"event": "step_update", "step_update": {
                "step_type": "agent_response", "text_delta": token}})
    _emit({"event": "result", "result": {"status": "SUCCESS"}})


def main():
    # SDK diagnostic bodies may contain request data. The parent logs a fixed
    # failure category; no SDK exception body or raw diagnostic is emitted here.
    logging.disable(logging.CRITICAL)
    try:
        sdk = load_sdk()
        if sys.argv[1:] == ["--probe"]:
            build_config(sdk, api_key="probe-not-a-real-key")
            _emit({"available": True, "version": SDK_VERSION})
            return 0
        if sys.argv[1:]:
            raise ValueError("Unknown worker option")
        data = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(data) > MAX_REQUEST:
            raise ValueError("Input too large")
        asyncio.run(asyncio.wait_for(run_request(json.loads(data), sdk), timeout=280))
        return 0
    except (Exception, KeyboardInterrupt):
        _emit({"event": "result", "result": {"status": "ERROR"}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
