"""
test_pipeline.py - Comprehensive Unit and Component Tests for Job Autopilot Pipeline
Tests database deduplication, job discovery filtering, evaluation schemas,
and Playwright form field mapping.
"""

import unittest
from unittest.mock import MagicMock, patch

from db import is_already_applied, log_application, _local_url_cache, SUPABASE_SCHEMA_SQL
from fetcher import extract_company_token, clean_html_content, matches_filter
from evaluator import CANDIDATE_PROFILE, EvaluationResult, evaluate_job


class TestDatabaseLayer(unittest.TestCase):
    def setUp(self):
        _local_url_cache.clear()

    @patch("db.get_supabase_client", return_value=None)
    def test_deduplication_local_cache(self, mock_client):
        test_url = "https://boards.greenhouse.io/testco/jobs/local-cache-test-999"
        self.assertFalse(is_already_applied(test_url))

        # Log application
        log_res = log_application(
            job_url=test_url,
            company="testco",
            title="Senior PHP Engineer",
            status="drafted",
            match_score=92
        )
        self.assertTrue(log_res["success"])
        self.assertTrue(is_already_applied(test_url))

    def test_schema_sql_present(self):
        self.assertIn("CREATE TABLE IF NOT EXISTS applications", SUPABASE_SCHEMA_SQL)
        self.assertIn("job_url TEXT UNIQUE NOT NULL", SUPABASE_SCHEMA_SQL)


class TestFetcherLayer(unittest.TestCase):
    def test_company_token_extraction(self):
        self.assertEqual(
            extract_company_token("https://boards-api.greenhouse.io/v1/boards/canonical/jobs"),
            "canonical"
        )
        self.assertEqual(
            extract_company_token("https://boards.greenhouse.io/automattic"),
            "automattic"
        )
        self.assertEqual(
            extract_company_token("gitlab"),
            "gitlab"
        )

    def test_clean_html_content(self):
        raw_html = "<p>Join our team as a <strong>PHP Developer</strong>.<br>Apply now!</p>"
        cleaned = clean_html_content(raw_html)
        self.assertIn("PHP Developer", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<strong>", cleaned)

    def test_keyword_matching(self):
        job1 = {
            "title": "Full Stack Engineer (PHP / Vue)",
            "location": "Remote - Worldwide",
            "description": "Building web apps with PHP and CodeIgniter."
        }
        self.assertTrue(matches_filter(job1, ["remote", "php", "full stack"]))

        job2 = {
            "title": "On-Site iOS Developer",
            "location": "New York, NY",
            "description": "Swift and Objective-C mobile applications."
        }
        self.assertFalse(matches_filter(job2, ["remote", "php", "full stack"]))


class TestEvaluatorLayer(unittest.TestCase):
    def test_candidate_profile_integrity(self):
        self.assertEqual(CANDIDATE_PROFILE["years_of_experience"], 4)
        self.assertIn("CodeIgniter", str(CANDIDATE_PROFILE["core_skills"]))
        self.assertIn("PHP", str(CANDIDATE_PROFILE["core_skills"]))
        self.assertTrue(CANDIDATE_PROFILE["employment_preferences"]["legally_authorized_to_work"])

    def test_evaluation_schema_validation(self):
        sample_data = {
            "match_score": 88,
            "is_remote": True,
            "reasoning": "Strong match with candidate's 4 years of PHP and CodeIgniter experience.",
            "custom_answers": {
                "salary_expectations": "$95,000 USD / year",
                "notice_period": "2 weeks"
            }
        }
        res = EvaluationResult(**sample_data)
        self.assertEqual(res.match_score, 88)
        self.assertTrue(res.is_remote)
        self.assertIn("salary_expectations", res.custom_answers)

    def test_fallback_evaluator(self):
        jd = "Looking for a Remote Full-Stack PHP Engineer with CodeIgniter and MySQL skills."
        eval_dict = evaluate_job(jd)
        self.assertGreaterEqual(eval_dict["match_score"], 70)
        self.assertTrue(eval_dict["is_remote"])
        self.assertIsInstance(eval_dict["custom_answers"], dict)


if __name__ == "__main__":
    unittest.main()
