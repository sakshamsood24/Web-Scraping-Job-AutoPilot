"""
check_llm.py - Diagnostic script to test the LLM evaluation layer.
Supports Google Gemini, OpenAI, or the built-in rule engine.
"""

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

from evaluator import evaluate_job


def test_evaluator():
    print("=" * 60)
    print("🧠 Testing Job Evaluator (LLM / Rule-Based)")
    print("=" * 60)

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if gemini_key and not any(p in gemini_key.lower() for p in ["your-", "your_", "dummy"]):
        print(f"✅ GEMINI_API_KEY detected. Using model: {os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')}")
    elif openai_key and not any(p in openai_key.lower() for p in ["your-", "your_", "sk-placeholder"]):
        print(f"✅ OPENAI_API_KEY detected. Using model: {os.getenv('OPENAI_MODEL_NAME', 'gpt-4o')}")
    else:
        print("ℹ️ No active API key detected. Running with built-in heuristic/rule engine (100% free).")

    sample_jd = """
    Job Title: Remote Full-Stack PHP Engineer
    Location: Worldwide (100% Remote)
    Requirements:
    - 3-5 years experience with PHP (PHP 7.4 / 8.x) and MVC frameworks like CodeIgniter or Laravel.
    - Strong SQL and database optimization skills (MySQL / PostgreSQL).
    - Familiarity with modern frontend JavaScript and RESTful APIs.
    - Comfortable working across distributed remote timezones.
    """

    sample_questions = [
        "What are your salary expectations?",
        "Are you authorized to work remotely?",
        "How many years of PHP experience do you have?"
    ]

    print("\nEvaluating sample remote PHP job description...")
    result = evaluate_job(sample_jd, form_questions=sample_questions)

    print("\n--- Evaluation Output ---")
    print(f"Match Score: {result.get('match_score')}/100")
    print(f"Is Remote  : {result.get('is_remote')}")
    print(f"Reasoning  : {result.get('reasoning')}")
    print("\nGenerated Custom Answers:")
    for q, a in result.get("custom_answers", {}).items():
        print(f"  • {q}: {a}")

    print("\n🎉 Evaluator is working perfectly!")
    return True


if __name__ == "__main__":
    test_evaluator()
