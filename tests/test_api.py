import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from magi import api


class TestAPI(unittest.TestCase):
    def test_root_serves_frontend(self):
        client = TestClient(api.app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("MAGI Supercomputer System", response.text)

    def test_deliberate_endpoint_returns_votes(self):
        mock_result = {
            "melchior_response": "Logic\nVOTE: YES",
            "balthasar_response": "Heart\nVOTE: NO",
            "casper_response": "People\nVOTE: YES",
            "final_decision": "APPROVED (2 to 1)",
            "melchior_model_used": "Phi-4",
            "balthasar_model_used": "Llama-4",
            "casper_model_used": "Mistral",
        }
        mock_system = AsyncMock()
        mock_system.ainvoke.return_value = mock_result

        client = TestClient(api.app)
        with patch("magi.api.magi_system", mock_system):
            response = client.post("/api/delibera", json={"dilemma": "Test?"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["melchior_vote"], "YES")
        self.assertEqual(payload["balthasar_vote"], "NO")
        self.assertEqual(payload["casper_vote"], "YES")
        self.assertEqual(payload["final_decision"], "APPROVED (2 to 1)")
