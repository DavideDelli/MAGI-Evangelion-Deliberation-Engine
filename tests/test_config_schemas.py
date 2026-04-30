import unittest

from magi import config, schemas


class TestConfigSchemas(unittest.TestCase):
    def test_config_contains_fallbacks(self):
        self.assertGreaterEqual(len(config.MELCHIOR_CONFIGS), 2)
        self.assertGreaterEqual(len(config.BALTHASAR_CONFIGS), 2)
        self.assertGreaterEqual(len(config.CASPER_CONFIGS), 2)
        self.assertIn("model", config.MELCHIOR_CONFIGS[0])
        self.assertIn("temperature", config.BALTHASAR_CONFIGS[0])

    def test_dilemma_request_schema(self):
        request = schemas.DilemmaRequest(dilemma="Test dilemma")
        self.assertEqual(request.dilemma, "Test dilemma")
