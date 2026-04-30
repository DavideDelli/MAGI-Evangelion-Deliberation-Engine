import json
import unittest
from datetime import datetime
from pathlib import Path

from magi import utils
from magi.config import BALTHASAR_CONFIGS, CASPER_CONFIGS


class TestUtils(unittest.TestCase):
    def test_extract_vote_handles_variants(self):
        self.assertEqual(utils.extract_vote("Analysis\nVOTE: YES"), "YES")
        self.assertEqual(utils.extract_vote("vote: no"), "NO")
        self.assertEqual(utils.extract_vote("No explicit vote"), "?")

    def test_is_rate_limit_detects_messages(self):
        self.assertTrue(utils.is_rate_limit(Exception("rate limit exceeded")))
        self.assertTrue(utils.is_rate_limit(Exception("HTTP 429")))
        self.assertFalse(utils.is_rate_limit(Exception("other error")))

    def test_save_log_writes_files_and_flags_fallback(self):
        base_dir = Path(utils.__file__).resolve().parents[2]
        date_dir = datetime.now().strftime("%Y%m%d")
        json_dir = base_dir / "logs" / "json" / date_dir
        md_dir = base_dir / "logs" / "markdown" / date_dir
        existing_json = set(json_dir.glob("magi_run_*.json")) if json_dir.exists() else set()
        existing_md = set(md_dir.glob("magi_run_*.md")) if md_dir.exists() else set()

        state = {
            "dilemma": "Should we proceed?",
            "melchior_response": "Logic path.\nVOTE: YES",
            "balthasar_response": "Heart says no.\nVOTE: NO",
            "casper_response": "People matter.\nVOTE: YES",
            "melchior_elapsed": 1.23,
            "balthasar_elapsed": 2.34,
            "casper_elapsed": 3.45,
            "melchior_model_used": "Fallback-Model",
            "balthasar_model_used": BALTHASAR_CONFIGS[0]["model"],
            "casper_model_used": CASPER_CONFIGS[0]["model"],
            "final_decision": "APPROVED (2 to 1)",
        }

        utils.save_log(state)

        new_json = set(json_dir.glob("magi_run_*.json")) - existing_json
        new_md = set(md_dir.glob("magi_run_*.md")) - existing_md
        self.assertEqual(len(new_json), 1)
        self.assertEqual(len(new_md), 1)

        json_path = next(iter(new_json))
        md_path = next(iter(new_md))
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["dilemma"], "Should we proceed?")
            self.assertEqual(payload["final_decision"], "APPROVED (2 to 1)")

            with open(md_path, "r", encoding="utf-8") as handle:
                markdown = handle.read()
            self.assertIn("fallback", markdown)
            self.assertIn("MELCHIOR-1", markdown)
        finally:
            json_path.unlink(missing_ok=True)
            md_path.unlink(missing_ok=True)
            if json_dir.exists() and not any(json_dir.iterdir()):
                json_dir.rmdir()
            if md_dir.exists() and not any(md_dir.iterdir()):
                md_dir.rmdir()
