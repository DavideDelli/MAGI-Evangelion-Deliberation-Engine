import asyncio
import unittest
from unittest.mock import patch

from magi import graph


class TestGraph(unittest.TestCase):
    def test_agent_nodes_return_expected_payloads(self):
        async def fake_ask(_persona, _dilemma, _configs, name):
            return f"{name} response\nVOTE: YES", 1.0, f"{name}-model"

        with patch("magi.graph.ask_agent_with_fallback", side_effect=fake_ask):
            melchior = asyncio.run(graph.melchior_node({"dilemma": "x"}))
            balthasar = asyncio.run(graph.balthasar_node({"dilemma": "x"}))
            casper = asyncio.run(graph.casper_node({"dilemma": "x"}))

        self.assertEqual(melchior["melchior_model_used"], "Melchior-model")
        self.assertEqual(balthasar["balthasar_elapsed"], 1.0)
        self.assertEqual(casper["casper_response"], "Casper response\nVOTE: YES")

    def test_arbitration_node_counts_votes(self):
        state = {
            "melchior_response": "Analysis\nVOTE: YES",
            "balthasar_response": "Analysis\nVOTE: NO",
            "casper_response": "Analysis\nVOTE: YES",
            "melchior_model_used": "m",
            "balthasar_model_used": "b",
            "casper_model_used": "c",
        }
        result = graph.arbitration_node(state)
        self.assertEqual(result["final_decision"], "APPROVED (2 to 1)")

    def test_arbitration_node_handles_missing_votes(self):
        state = {
            "melchior_response": "Analysis\nVOTE: YES",
            "balthasar_response": "Analysis",
            "casper_response": "Analysis\nVOTE: NO",
            "melchior_model_used": "m",
            "balthasar_model_used": "b",
            "casper_model_used": "c",
        }
        result = graph.arbitration_node(state)
        self.assertEqual(result["final_decision"], "TIE (YES=1, NO=1, ?=1)")

    def test_logging_node_calls_save_log(self):
        with patch("magi.graph.save_log") as mock_save:
            graph.logging_node({"dilemma": "x"})
        mock_save.assert_called_once()

    def test_build_graph_compiles(self):
        compiled = graph.build_graph()
        self.assertTrue(hasattr(compiled, "ainvoke"))
