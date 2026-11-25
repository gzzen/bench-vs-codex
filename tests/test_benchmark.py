import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from benchmark import (
    Case, codex_provider_overrides, codex_settings, extract_answer, judge_semantic_match,
    load_cases, load_project_env, normalize_answer, print_table, run_cases, run_codex,
    run_paperqa, score,
)


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

    def test_summarizer_receives_only_question_and_response(self):
        reply = {"choices": [{"message": {"content": "01983 873655"}}]}
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as request_call:
                answer, settings = extract_answer("What is the phone?", "01983 873655 (citation)")
        self.assertEqual(answer, "01983 873655")
        self.assertEqual(settings["model"], "deepseek-flash")
        payload = json.loads(request_call.call_args.args[0].data)
        self.assertEqual(json.loads(payload["messages"][1]["content"]), {
            "question": "What is the phone?", "response": "01983 873655 (citation)",
        })

    def test_semantic_judge_verdicts_and_payload(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            for response, expected in (("YES", True), ("No.", False)):
                reply = {"choices": [{"message": {"content": response}}]}
                with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as call:
                    verdict, settings = judge_semantic_match("Who?", "Robert", "Bob")
                self.assertIs(verdict, expected)
                self.assertEqual(settings["model"], "deepseek-flash")
                payload = json.loads(call.call_args.args[0].data)
                self.assertEqual(json.loads(payload["messages"][1]["content"]), {
                    "question": "Who?", "candidate_answer": "Robert", "reference_answer": "Bob",
                })
            invalid = {"choices": [{"message": {"content": "Maybe"}}]}
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(invalid).encode())):
                with self.assertRaisesRegex(ValueError, "invalid verdict"):
                    judge_semantic_match("Who?", "Robert", "Bob")

    def test_both_runners_receive_exact_manifest_question(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "one.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            case = Case("case-one", "one.pdf", "  Exact question?  ", "Answer")
            checkout = root / "paper-qa"
            python = checkout / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            paperqa_result = json.dumps({
                "answer": "Answer (citation)", "formatted_answer": "Answer (citation)",
                "model": "deepseek-flash", "summary_model": "deepseek-flash",
                "agent_model": "deepseek-flash", "embedding": "sparse", "status": "success",
            })
            with patch("benchmark._run_command", return_value=paperqa_result) as command_call:
                run_paperqa(case, pdf, root / "paperqa-case", checkout)
            command = command_call.call_args.args[0]
            self.assertEqual(command[command.index("--question") + 1], case.question)

            codex_case_dir = root / "codex-case"
            codex_case_dir.mkdir()

            def fake_codex_command(command, cwd, log_prefix, *, stdin):
                (codex_case_dir / "codex-answer.txt").write_text("Answer", encoding="utf-8")
                self.assertEqual(stdin, case.question)
                return ""

            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
                with patch("benchmark._run_command", side_effect=fake_codex_command):
                    run_codex(case, pdf, codex_case_dir, "codex")

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
                return "Answer (citation)", {"preset": "stub"}, "Answer (cited source)"

            def bad(case, pdf, case_dir):
                raise RuntimeError("model unavailable")

            seen = []

            def summarize(question, response):
                seen.append((question, response))
                return "Answer", {"model": "stub"}

            judged = []

            def judge(question, answer, reference):
                judged.append((question, answer, reference))
                return True, {"model": "judge-stub"}

            output = root / "work" / "results.jsonl"
            progress = io.StringIO()
            with redirect_stdout(progress):
                records = run_cases(cases, root / "dataset", root / "work", output, {
                    "paperqa": good, "codex": bad,
                }, summarize, judge, planned_settings={"codex": {"sandbox": "read-only"}})
            self.assertEqual(records[0]["exact_match"], True)
            self.assertIsNone(records[0]["error"])
            self.assertEqual(records[0]["settings"], {"preset": "stub"})
            self.assertEqual(records[0]["raw_answer"], "Answer (citation)")
            self.assertEqual(records[0]["answer"], "Answer")
            self.assertEqual(records[0]["full_answer"], "Answer (cited source)")
            self.assertEqual(records[0]["summarizer_settings"]["model"], "stub")
            self.assertIs(records[0]["semantic_match"], True)
            self.assertEqual(records[0]["judge_settings"]["model"], "judge-stub")
            self.assertIsNone(records[1]["exact_match"])
            self.assertIsNone(records[1]["semantic_match"])
            self.assertIn("model unavailable", records[1]["error"])
            self.assertEqual(records[1]["settings"], {"sandbox": "read-only"})
            self.assertEqual(seen, [("Question?", "Answer (citation)")])
            self.assertEqual(judged, [("Question?", "Answer", "Answer")])
            saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(saved, records)
            self.assertIn("[1/1] case-one / paperqa running...", progress.getvalue())
            self.assertIn("[1/1] case-one / paperqa extracting...", progress.getvalue())
            self.assertIn("[1/1] case-one / paperqa judging...", progress.getvalue())
            self.assertIn("[1/1] case-one / paperqa done (", progress.getvalue())
            self.assertIn("[1/1] case-one / codex failed (", progress.getvalue())

            def failed_summarizer(question, response):
                raise RuntimeError("extractor unavailable")

            with redirect_stdout(io.StringIO()):
                failed_records = run_cases(
                    cases, root / "dataset", root / "failed-work", root / "failed-work" / "results.jsonl",
                    {"paperqa": good}, failed_summarizer, judge,
                )
            self.assertEqual(failed_records[0]["raw_answer"], "Answer (citation)")
            self.assertIsNone(failed_records[0]["answer"])
            self.assertIn("extractor unavailable", failed_records[0]["summarizer_error"])
            self.assertIsNone(failed_records[0]["exact_match"])
            self.assertIsNone(failed_records[0]["semantic_match"])

            def failed_judge(question, answer, reference):
                raise RuntimeError("judge unavailable")

            with redirect_stdout(io.StringIO()):
                judge_failed_records = run_cases(
                    cases, root / "dataset", root / "judge-failed", root / "judge-failed" / "results.jsonl",
                    {"paperqa": good}, summarize, failed_judge,
                )
            self.assertIsNone(judge_failed_records[0]["semantic_match"])
            self.assertIn("judge unavailable", judge_failed_records[0]["judge_error"])
            table = io.StringIO()
            with redirect_stdout(table):
                print_table(records)
            self.assertIn(" EM    SM ", table.getvalue())


if __name__ == "__main__":
    unittest.main()
