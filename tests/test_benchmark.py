import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from benchmark import codex_provider_overrides, codex_settings, load_cases, load_project_env, normalize_answer, run_cases, score
from paperqa_bridge import short_answer


class BenchmarkTests(unittest.TestCase):
    def test_project_env_loading_and_export_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text('# local key\nDEEPSEEK_API_KEY="from-file"\n', encoding="utf-8")
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
                load_project_env(env_file)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from-file")
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "from-shell"}):
                load_project_env(env_file)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from-shell")

    def test_codex_provider_uses_environment_key(self):
        config = codex_settings()
        overrides = codex_provider_overrides(config)
        self.assertIn('model_providers.deepseek.env_key="DEEPSEEK_API_KEY"', overrides)
        self.assertFalse(any("bearer_token" in item for item in overrides))

    def test_case_loading_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "cases.json"
            manifest.write_text(json.dumps([{
                "id": "case-one", "document": "one.pdf", "question": "Question?",
                "reference_answer": "Answer",
            }]), encoding="utf-8")
            cases = load_cases(manifest)
            self.assertEqual(cases[0].id, "case-one")
            manifest.write_text(json.dumps([{
                "id": "case-one", "document": "../wrong.pdf", "question": "Question?",
                "reference_answer": "Answer",
            }]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(manifest)

    def test_normalization_and_exact_match(self):
        self.assertEqual(normalize_answer('  “Rick   SCOTT.”\n'), "rick scott")
        self.assertTrue(score("  RICK  Scott.  ", "Rick Scott"))
        self.assertFalse(score("Governor Rick Scott", "Rick Scott"))
        self.assertEqual(short_answer("01983 873655 (pqac-d79ef6fa)"), "01983 873655")
        self.assertEqual(short_answer("Rick Scott (pqac-d79ef6fa, pqac-0f650d59)"), "Rick Scott")

    def test_records_answers_and_runner_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "dataset" / "data" / "documents"
            documents.mkdir(parents=True)
            (documents / "one.pdf").write_bytes(b"%PDF-1.4\n")
            manifest = root / "cases.json"
            manifest.write_text(json.dumps([{
                "id": "case-one", "document": "one.pdf", "question": "Question?",
                "reference_answer": "Answer",
            }]), encoding="utf-8")
            cases = load_cases(manifest)

            def good(case, pdf, case_dir):
                self.assertEqual(pdf.name, "one.pdf")
                return " Answer. ", {"preset": "stub"}, "Answer (cited source)"

            def bad(case, pdf, case_dir):
                raise RuntimeError("model unavailable")

            output = root / "work" / "results.jsonl"
            progress = io.StringIO()
            with redirect_stdout(progress):
                records = run_cases(cases, root / "dataset", root / "work", output, {
                    "paperqa": good, "codex": bad,
                }, planned_settings={"codex": {"sandbox": "read-only"}})
            self.assertEqual(records[0]["exact_match"], True)
            self.assertIsNone(records[0]["error"])
            self.assertEqual(records[0]["settings"], {"preset": "stub"})
            self.assertEqual(records[0]["full_answer"], "Answer (cited source)")
            self.assertIsNone(records[1]["exact_match"])
            self.assertIn("model unavailable", records[1]["error"])
            self.assertEqual(records[1]["settings"], {"sandbox": "read-only"})
            saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(saved, records)
            self.assertIn("[1/1] case-one / paperqa running...", progress.getvalue())
            self.assertIn("[1/1] case-one / paperqa done (", progress.getvalue())
            self.assertIn("[1/1] case-one / codex failed (", progress.getvalue())


if __name__ == "__main__":
    unittest.main()
