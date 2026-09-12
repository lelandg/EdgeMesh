"""SDK boundary tests with an injected SDK; no credentials or network required."""
import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_sdk_worker import build_config, run_request


class WorkerTests(unittest.TestCase):
    def test_configuration_disables_all_external_actions(self):
        capabilities = Mock(return_value="empty capabilities")
        configuration = Mock(return_value="configuration")
        deny = Mock(return_value="deny all")
        sdk = (Mock(), capabilities, configuration, deny)
        with tempfile.TemporaryDirectory() as directory:
            result = build_config(sdk, api_key="fixture-key", directory=directory)
            self.assertEqual(result, "configuration")
            options = configuration.call_args.kwargs
            self.assertEqual(options["workspaces"], [str(Path(directory).resolve())])
        capabilities.assert_called_once_with(enabled_tools=[], enable_subagents=False)
        deny.assert_called_once_with("*")
        self.assertEqual(options["policies"], ["deny all"])
        self.assertFalse(options["vertex"])
        self.assertEqual(options["save_dir"], "")
        for name in ("tools", "mcp_servers", "hooks", "triggers", "skills_paths", "subagents"):
            self.assertEqual(options[name], [])

    def test_sdk_response_stream_maps_to_bounded_runtime_protocol(self):
        class FakeAgent:
            def __init__(self, config):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def chat(self, prompt):
                async def tokens():
                    yield "More "
                    yield "depth."
                return tokens()

        sdk = (FakeAgent, Mock(), Mock(), Mock())
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fixture-key"}), patch("agent_sdk_worker._emit") as emit:
            asyncio.run(run_request({"prompt": "help", "model": ""}, sdk))
        events = [call.args[0] for call in emit.call_args_list]
        self.assertEqual(events[0]["init"]["tools"], [])
        self.assertEqual(events[-1]["result"]["status"], "SUCCESS")
        self.assertEqual("".join(item["step_update"]["text_delta"] for item in events if "step_update" in item), "More depth.")

    def test_sdk_rejects_extra_request_fields(self):
        with self.assertRaises(ValueError):
            asyncio.run(run_request({"prompt": "help", "model": "", "command": "unwanted"}, (None, None, None, None)))


if __name__ == "__main__":
    unittest.main()
