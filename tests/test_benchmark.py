import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from benchmark import main, print_table, run_cases, select_cases
from benchmark_data import (
    Case, filter_cases, load_cases, load_run_config, questions_path,
)
from benchmark_evaluation import (
    extract_answer, judge_semantic_match, load_project_env, normalize_answer, score,
)
from benchmark_systems import (
    PaperQAEmptyDocsError, codex_provider_overrides, codex_settings, run_codex,
    run_paperqa,
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

    def test_mmlongbench_loading_and_evidence_filters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            samples = [
                {
                    "doc_id": "one.pdf", "doc_type": "Report", "question": "Text?",
                    "answer": "A", "evidence_pages": "[1]",
                    "evidence_sources": "['Pure-text (Plain-text)']",
                    "answer_format": "Str",
                },
                {
                    "doc_id": "one.pdf", "doc_type": "Report", "question": "Chart?",
                    "answer": "B", "evidence_pages": "[2]",
                    "evidence_sources": "['Chart']",
                    "answer_format": "Str",
                },
                {
                    "doc_id": "two.pdf", "doc_type": "Brochure", "question": "Figure?",
                    "answer": "C", "evidence_pages": "[3, 4]",
                    "evidence_sources": "['Figure', 'Table']",
                    "answer_format": "Str",
                },
            ]
            (data / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
            cases = load_cases(questions_path(data))
            self.assertEqual([case.id for case in cases], [
                "mmlongbench-0001", "mmlongbench-0002", "mmlongbench-0003",
            ])
            self.assertEqual(cases[2].evidence_sources, ("Figure", "Table"))
            self.assertEqual(cases[2].evidence_pages, (3, 4))
            self.assertEqual(cases[1].reference_answer, "B")

            config_path = root / "run_configs.json"
            config_path.write_text(json.dumps({"profiles": {
                "no-visuals": {
                    "systems": ["codex"],
                    "limit": 1,
                    "filters": {"exclude_evidence_sources": ["figure", "CHART"]},
                },
                "tables": {
                    "random": True,
                    "filters": {"include_evidence_sources": ["Table"]},
                },
            }}), encoding="utf-8")
            config = load_run_config(config_path, "no-visuals")
            self.assertEqual(config.systems, ("codex",))
            self.assertEqual(config.limit, 1)
            filtered = filter_cases(cases, config.filters)
            self.assertEqual([case.id for case in filtered], ["mmlongbench-0001"])
            table_config = load_run_config(config_path, "tables")
            self.assertTrue(table_config.randomize)
            table_cases = filter_cases(cases, table_config.filters)
            self.assertEqual([case.id for case in table_cases], ["mmlongbench-0003"])

    def test_dataset_dry_run_uses_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            documents = data / "documents"
            documents.mkdir(parents=True)
            (documents / "one.pdf").write_bytes(b"%PDF-1.4\n")
            (data / "samples.json").write_text(json.dumps([{
                "doc_id": "one.pdf", "doc_type": "Report", "question": "Plain text?",
                "answer": "A", "evidence_pages": "[1]",
                "evidence_sources": "['Pure-text (Plain-text)']",
                "answer_format": "Str",
            }, {
                "doc_id": "one.pdf", "doc_type": "Report", "question": "Chart?",
                "answer": "B", "evidence_pages": "[2]",
                "evidence_sources": "['Chart']", "answer_format": "Str",
            }]), encoding="utf-8")
            config = root / "run_configs.json"
            config.write_text(json.dumps({"profiles": {
                "skip-visuals": {
                    "filters": {"exclude_evidence_sources": ["Figure", "Chart"]},
                },
                "all": {"filters": {}},
            }}), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = main([
                    "--data", str(data), "--config", str(config),
                    "--profile", "skip-visuals", "--dry-run",
                ])
            self.assertEqual(status, 0)
            self.assertIn("Plain text?", output.getvalue())
            self.assertNotIn("Chart?", output.getvalue())
            self.assertIn("Profile skip-visuals: validated 1 PDF paths", output.getvalue())

            with patch("benchmark.random.shuffle", side_effect=lambda values: values.reverse()):
                with redirect_stdout(output := io.StringIO()):
                    status = main([
                        "--data", str(data), "--config", str(config),
                        "--profile", "all", "--limit", "1", "--random", "--dry-run",
                    ])
            self.assertEqual(status, 0)
            self.assertIn("Chart?", output.getvalue())
            self.assertNotIn("Plain text?", output.getvalue())
            self.assertIn("selection=random", output.getvalue())

    def test_select_cases_randomizes_before_limit(self):
        cases = [
            Case("case-a", "one.pdf", "A?", "A"),
            Case("case-b", "one.pdf", "B?", "B"),
            Case("case-c", "one.pdf", "C?", "C"),
        ]
        self.assertEqual(
            [case.id for case in select_cases(cases, 2)], ["case-a", "case-b"],
        )
        with patch("benchmark.random.shuffle", side_effect=lambda values: values.reverse()):
            selected = select_cases(cases, 2, randomize=True)
        self.assertEqual([case.id for case in selected], ["case-c", "case-b"])

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

    def test_both_runners_receive_exact_question(self):
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
            with patch("benchmark_systems._run_command", return_value=paperqa_result) as command_call:
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
                with patch("benchmark_systems._run_command", side_effect=fake_codex_command):
                    run_codex(case, pdf, codex_case_dir, "codex")

    def test_paperqa_no_papers_is_an_execution_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "one.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            case = Case("case-one", "one.pdf", "Question?", "Answer")
            checkout = root / "paper-qa"
            python = checkout / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            no_papers = "I cannot answer this question due to having no papers."
            paperqa_result = json.dumps({
                "answer": no_papers, "formatted_answer": no_papers,
                "model": "deepseek-flash", "summary_model": "deepseek-flash",
                "agent_model": "deepseek-flash", "embedding": "sparse", "status": "unsure",
            })
            with patch("benchmark_systems._run_command", return_value=paperqa_result):
                with self.assertRaisesRegex(PaperQAEmptyDocsError, "no readable papers"):
                    run_paperqa(case, pdf, root / "paperqa-case", checkout)

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
