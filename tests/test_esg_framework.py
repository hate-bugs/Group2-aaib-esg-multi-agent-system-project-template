from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from project.esg.config import ESGExperimentConfig
from project.esg.data import load_reports
from project.esg.runner import run_experiments
from project.llm_config import llm, llm_thinking


class ESGFrameworkTests(unittest.TestCase):
    def test_llm_config_allows_missing_env(self) -> None:
        self.assertIsNone(llm)
        self.assertIsNone(llm_thinking)

    def test_load_reports_accepts_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "reports.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["report_id", "company_name", "preprocessed_content", "environmental_score", "social_score", "governance_score", "esg_score"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "report_id": "r1",
                        "company_name": "Test Health",
                        "preprocessed_content": "environmental strategy target board oversight community transparency",
                        "environmental_score": "70",
                        "social_score": "65",
                        "governance_score": "72",
                        "esg_score": "69",
                    }
                )
            reports = load_reports(dataset_path, sample_size=5)
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0].report_id, "r1")
            self.assertEqual(reports[0].company_name, "Test Health")
            self.assertAlmostEqual(float(reports[0].actual_scores["total"]), 69.0)

    def test_run_experiments_generates_summary(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = ESGExperimentConfig(
                dataset_path=root / "sustainability-reports-2026-05-29.csv",
                sample_size=2,
                trials=2,
                output_dir=tmp_path / "outputs",
                chunk_dir=tmp_path / "indexes",
                sample_output_path=tmp_path / "sample.json",
            )
            output = run_experiments(config)
            self.assertIn("summary", output)
            self.assertIn("patterns", output["summary"])
            self.assertEqual(set(output["summary"]["patterns"].keys()), {"parallel", "hierarchical", "review"})
            result_path = tmp_path / "outputs" / "first10_metrics_summary.json"
            self.assertTrue(result_path.exists())
            self.assertTrue((tmp_path / "sample.json").exists())
            self.assertEqual(len(output["results"]), 6)


if __name__ == "__main__":
    unittest.main()
