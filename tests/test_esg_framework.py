import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project.esg_framework.data import filter_healthcare_reports, load_reports, select_sample
from project.esg_framework.runner import run_experiment


class TestESGFramework(unittest.TestCase):
    def setUp(self):
        self.dataset = Path(__file__).resolve().parents[1] / "knowledge" / "sustainability-reports-preprocessed.csv"

    def test_load_and_select_reports(self):
        records = load_reports(self.dataset)
        self.assertGreater(len(records), 0)
        healthcare = filter_healthcare_reports(records)
        sample = select_sample(healthcare, 10)
        self.assertLessEqual(len(sample), 10)
        self.assertGreater(len(sample), 0)

    def test_run_experiment_outputs_all_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.json"
            payload = run_experiment(
                dataset_path=self.dataset,
                sample_size=2,
                trials=1,
                output_path=output_file,
                chunk_store_dir=tmpdir,
            )

            self.assertEqual(
                sorted(payload["patterns"]),
                ["handoff_hierarchical", "parallel_concurrent", "review_critique"],
            )
            self.assertIn("summary", payload)
            self.assertTrue(output_file.exists())

            data = json.loads(output_file.read_text(encoding="utf-8"))
            for pattern in payload["patterns"]:
                self.assertIn(pattern, data["summary"])
                self.assertIn("accuracy", data["summary"][pattern])
                self.assertIn("predicted_total_avg", data["summary"][pattern])
                self.assertIn("actual_total_avg", data["summary"][pattern])

            for pattern in payload["patterns"]:
                self.assertIn(pattern, data["results"])
                self.assertGreater(len(data["results"][pattern]), 0)
                first = data["results"][pattern][0]
                for domain in ["environmental", "social", "governance"]:
                    self.assertIn("estimated_score", first["domain_scores"][domain])
                    self.assertIn("confidence", first["domain_scores"][domain])
                    self.assertIn("rationale", first["domain_scores"][domain])
                    self.assertTrue(first["domain_scores"][domain]["rationale"])


if __name__ == "__main__":
    unittest.main()
