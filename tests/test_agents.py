import asyncio
import os
import unittest
from unittest.mock import patch

from magi import agents


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, model: str):
        self.model = model


class FakeChain:
    def __init__(self, side_effects):
        self._side_effects = side_effects

    async def ainvoke(self, _payload):
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class FakePrompt:
    def __init__(self, side_effects_by_model):
        self._side_effects_by_model = side_effects_by_model

    def __or__(self, llm):
        return FakeChain(self._side_effects_by_model[llm.model])


class TestAgents(unittest.TestCase):
    def test_make_llm_uses_expected_settings(self):
        os.environ["GITHUB_TOKEN"] = "token"
        with patch("magi.agents.ChatOpenAI") as mock_llm:
            result = agents._make_llm("model-x", 0.3)
        mock_llm.assert_called_once_with(
            api_key="token",
            base_url="https://models.inference.ai.azure.com",
            model="model-x",
            temperature=0.3,
        )
        self.assertIs(result, mock_llm.return_value)

    def test_ask_agent_with_fallback_success_first(self):
        side_effects = {"primary": [FakeResponse("OK\nVOTE: YES")]}
        fake_prompt = FakePrompt(side_effects)
        with patch("magi.agents.ChatPromptTemplate.from_messages", return_value=fake_prompt), \
             patch("magi.agents._make_llm", side_effect=lambda model, temp: FakeLLM(model)):
            response, elapsed, model_used = asyncio.run(
                agents.ask_agent_with_fallback("persona", "dilemma", [{"model": "primary", "temperature": 0.1}], "Agent")
            )
        self.assertEqual(response, "OK\nVOTE: YES")
        self.assertEqual(model_used, "primary")
        self.assertGreaterEqual(elapsed, 0)

    def test_ask_agent_with_fallback_rate_limit_then_success(self):
        side_effects = {
            "primary": [Exception("rate limit")],
            "fallback": [FakeResponse("Fallback\nVOTE: NO")],
        }
        fake_prompt = FakePrompt(side_effects)
        with patch("magi.agents.ChatPromptTemplate.from_messages", return_value=fake_prompt), \
             patch("magi.agents._make_llm", side_effect=lambda model, temp: FakeLLM(model)):
            response, _, model_used = asyncio.run(
                agents.ask_agent_with_fallback(
                    "persona",
                    "dilemma",
                    [
                        {"model": "primary", "temperature": 0.1},
                        {"model": "fallback", "temperature": 0.2},
                    ],
                    "Agent",
                )
            )
        self.assertEqual(response, "Fallback\nVOTE: NO")
        self.assertEqual(model_used, "fallback")

    def test_ask_agent_with_fallback_transient_retry(self):
        side_effects = {"primary": [Exception("boom"), FakeResponse("Recovered\nVOTE: YES")]}
        fake_prompt = FakePrompt(side_effects)

        async def fake_sleep(_delay):
            return None

        with patch("magi.agents.ChatPromptTemplate.from_messages", return_value=fake_prompt), \
             patch("magi.agents._make_llm", side_effect=lambda model, temp: FakeLLM(model)), \
             patch("magi.agents.asyncio.sleep", side_effect=fake_sleep):
            response, _, model_used = asyncio.run(
                agents.ask_agent_with_fallback("persona", "dilemma", [{"model": "primary", "temperature": 0.1}], "Agent")
            )
        self.assertEqual(response, "Recovered\nVOTE: YES")
        self.assertEqual(model_used, "primary")

    def test_ask_agent_with_fallback_exhausts_plans(self):
        side_effects = {"primary": [Exception("rate limit")]}
        fake_prompt = FakePrompt(side_effects)
        with patch("magi.agents.ChatPromptTemplate.from_messages", return_value=fake_prompt), \
             patch("magi.agents._make_llm", side_effect=lambda model, temp: FakeLLM(model)):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    agents.ask_agent_with_fallback("persona", "dilemma", [{"model": "primary", "temperature": 0.1}], "Agent")
                )
