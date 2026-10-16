"""
main.py - Orchestrator for Headless Autonomous Job Application Pipeline ("Autopilot")
Discovers remote Full-Stack PHP/CodeIgniter roles, evaluates them via LangChain LLM,
logs to Supabase PostgreSQL, and automates form filling headlessly via Playwright.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import pipeline modules
from db import is_already_applied, get_all_applied_urls, log_application, get_supabase_client, SUPABASE_SCHEMA_SQL
from fetcher import (
    fetch_greenhouse_jobs,
    fetch_worldwide_remote_jobs,
    DEFAULT_KEYWORDS,
    POPULAR_TECH_COMPANIES,
)
from evaluator import evaluate_job
from executor import fill_job_application_sync

# Setup colored/structured console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("JobAutopilot.Main")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the autopilot pipeline."""
    parser = argparse.ArgumentParser(
        description="Headless Autonomous Job Application Pipeline (Autopilot)"
    )
    parser.add_argument(
        "--boards",
        type=str,
        default=os.getenv("DEFAULT_GREENHOUSE_BOARDS", "canonical"),
        help="Comma-separated Greenhouse board tokens or full URLs (e.g., 'canonical,automattic')"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated keywords to filter roles (default: 'remote,php,full stack')"
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=int(os.getenv("MIN_MATCH_SCORE", "70")),
        help="Minimum LLM match score (0-100) required to initiate application (default: 70)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=os.getenv("RESUME_PATH", "assets/Saksham_Sood_Resume(1).pdf"),
        help="Path to resume file (PDF/DOCX) for programmatic upload"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=os.getenv("HEADLESS", "true").lower() in ("true", "1", "yes"),
        help="Run Playwright Chromium browser in headless mode (default: True)"
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run Playwright with visible browser UI for visual debugging"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of candidate jobs to evaluate per run (default: 5)"
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the Supabase SQL schema and exit"
    )
    return parser.parse_args()


def run_autopilot(
    boards: List[str],
    keywords: List[str],
    min_score: int,
    resume_path: str,
    headless: bool = True,
    job_limit: int = 5,
    loop_id: Optional[int] = None,
    loop_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Executes the end-to-end autopilot workflow:
    1. Discovery: Scrape Greenhouse job boards with keyword filtering
    2. Deduplication: Check Supabase DB to avoid double-applying
    3. Evaluation: Score role and generate answers using LangChain ChatOpenAI
    4. Execution: Headlessly autofill application and upload resume with Playwright
    5. Persistence: Record status and screenshots in Supabase PostgreSQL
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Job Application Autopilot Pipeline")
    logger.info(f"Target Boards: {boards}")
    logger.info(f"Keywords: {keywords}")
    logger.info(f"Min Match Score: {min_score} | Resume: {resume_path} | Headless: {headless}")
    logger.info("=" * 60)

    # 1. Verify DB status
    client = get_supabase_client()
    if client is None:
        logger.warning(
            "Supabase client running in offline/memory cache mode. "
            "To connect PostgreSQL, set SUPABASE_URL and SUPABASE_KEY in your .env file."
        )

    # Expand "all", "auto", "worldwide", "global", or "any" to all popular tech companies + global remote feeds
    expanded_boards: List[str] = []
    include_worldwide_feed = False
    for b in boards:
        b_clean = b.strip().lower()
        if b_clean in ("all", "auto", "popular", "any", "top", "worldwide", "global", "remote"):
            expanded_boards.extend(POPULAR_TECH_COMPANIES)
            include_worldwide_feed = True
        elif b_clean:
            expanded_boards.append(b.strip())
    
    unique_boards = list(dict.fromkeys(expanded_boards)) if expanded_boards else POPULAR_TECH_COMPANIES
    if not expanded_boards:
        include_worldwide_feed = True

    # 2. Discovery Phase
    all_discovered_jobs: List[Dict[str, Any]] = []

    # A. Scrape Greenhouse company boards
    for board_token in unique_boards:
        try:
            jobs = fetch_greenhouse_jobs(board_token, keywords=keywords, filter_by_keywords=True)
            all_discovered_jobs.extend(jobs)
        except Exception as e:
            logger.error(f"Error fetching jobs from board '{board_token}': {e}")

    # B. If worldwide/global is enabled, also scrape worldwide remote job feeds (Jobicy, Remotive)
    if include_worldwide_feed:
        try:
            worldwide_jobs = fetch_worldwide_remote_jobs(keywords=keywords, limit=40)
            all_discovered_jobs.extend(worldwide_jobs)
            logger.info(f"Discovered {len(worldwide_jobs)} additional candidate jobs from global worldwide feeds.")
        except Exception as e:
            logger.error(f"Error fetching worldwide remote feeds: {e}")

    logger.info(f"Discovered {len(all_discovered_jobs)} total candidate job postings across {len(unique_boards)} boards and worldwide feeds.")

    # Deduplication using batch pre-fetched URLs from Supabase
    applied_urls = get_all_applied_urls()
    unapplied_jobs = [j for j in all_discovered_jobs if j["url"] not in applied_urls]
    logger.info(f"Deduplication: {len(all_discovered_jobs) - len(unapplied_jobs)} already in database, {len(unapplied_jobs)} unapplied fresh jobs remaining.")

    if not unapplied_jobs:
        logger.info("ℹ️ No new unapplied jobs found matching the criteria in this run.")
        return []

    # Prioritize and interleave across companies to prevent single-company starvation
    by_company: Dict[str, List[Dict[str, Any]]] = {}
    for j in unapplied_jobs:
        by_company.setdefault(j["company"], []).append(j)

    interleaved_jobs: List[Dict[str, Any]] = []
    max_len = max(len(v) for v in by_company.values())
    for i in range(max_len):
        for comp_name, comp_jobs in by_company.items():
            if i < len(comp_jobs):
                interleaved_jobs.append(comp_jobs[i])

    # Apply limit
    jobs_to_process = interleaved_jobs[:job_limit]
    logger.info(f"Processing up to {len(jobs_to_process)} fresh jobs in this batch.")

    pipeline_summary: List[Dict[str, Any]] = []

    # 3. Processing Loop
    for idx, job in enumerate(jobs_to_process, 1):
        job_url = job["url"]
        job_title = job["title"]
        company = job["company"]
        logger.info("-" * 60)
        logger.info(f"[{idx}/{len(jobs_to_process)}] Inspecting: '{job_title}' at '{company}'")
        logger.info(f"URL: {job_url}")

        # Step A: Double check deduplication
        if is_already_applied(job_url, known_urls=applied_urls):
            logger.info(f"⏩ [ALREADY PROCESSED] Job '{job_url}' already exists in database. Skipping.")
            continue

        # Step B: LLM Evaluation Layer
        logger.info(f"🧠 Evaluating Job Description for '{job_title}'...")
        eval_result = evaluate_job(
            job_description=job["description"] or f"Title: {job_title}\nLocation: {job['location']}"
        )
        match_score = eval_result.get("match_score", 0)
        is_remote = eval_result.get("is_remote", False)
        reasoning = eval_result.get("reasoning", "")
        custom_answers = eval_result.get("custom_answers", {})

        logger.info(f"Score: {match_score}/100 | Is Remote: {is_remote} | Reasoning: {reasoning}")

        # Step C: Evaluate qualification against thresholds
        if not is_remote:
            logger.info(f"❌ [NOT REMOTE] Position '{job_title}' is not remote. Skipping application.")
            status = "skipped_not_remote"
            metadata = {"reasoning": reasoning}
            if loop_id:
                metadata["loop_id"] = loop_id
            if loop_name:
                metadata["loop_name"] = loop_name
            log_application(
                job_url=job_url,
                company=company,
                title=job_title,
                status=status,
                match_score=match_score,
                metadata=metadata
            )
            pipeline_summary.append({
                "job_title": job_title,
                "company": company,
                "url": job_url,
                "status": status,
                "match_score": match_score,
                "screenshot": None,
            })
            continue

        if match_score < min_score:
            logger.info(f"⚠️ [SCORE BELOW THRESHOLD] Score {match_score} < {min_score}. Skipping application.")
            status = "skipped_low_match"
            metadata = {"reasoning": reasoning}
            if loop_id:
                metadata["loop_id"] = loop_id
            if loop_name:
                metadata["loop_name"] = loop_name
            log_application(
                job_url=job_url,
                company=company,
                title=job_title,
                status=status,
                match_score=match_score,
                metadata=metadata
            )
            pipeline_summary.append({
                "job_title": job_title,
                "company": company,
                "url": job_url,
                "status": status,
                "match_score": match_score,
                "screenshot": None,
            })
            continue

        # Step D: Playwright Execution Layer (Safe V1: Autofill & Screenshot, NO Submit)
        logger.info(f"🎯 [MATCH ACCEPTED] Score {match_score} >= {min_score}. Launching Playwright autofill...")
        exec_result = fill_job_application_sync(
            job_url=job_url,
            custom_answers=custom_answers,
            resume_path=resume_path,
            headless=headless,
        )

        screenshot_path = exec_result.get("screenshot_path")
        fields_filled = exec_result.get("fields_filled", [])

        if exec_result.get("success"):
            status = "drafted"
            logger.info(f"✅ Form successfully drafted! Screenshot saved at: {screenshot_path}")
            metadata = {
                "screenshot_path": screenshot_path,
                "fields_filled": fields_filled,
                "reasoning": reasoning,
                "custom_answers": custom_answers,
            }
            if loop_id:
                metadata["loop_id"] = loop_id
            if loop_name:
                metadata["loop_name"] = loop_name
            log_application(
                job_url=job_url,
                company=company,
                title=job_title,
                status=status,
                match_score=match_score,
                metadata=metadata
            )
        else:
            status = "failed"
            error_msg = exec_result.get("error", "Unknown execution failure")
            logger.error(f"❌ Execution failed for {job_url}: {error_msg}")
            metadata = {"error": error_msg, "screenshot_path": screenshot_path}
            if loop_id:
                metadata["loop_id"] = loop_id
            if loop_name:
                metadata["loop_name"] = loop_name
            log_application(
                job_url=job_url,
                company=company,
                title=job_title,
                status=status,
                match_score=match_score,
                metadata=metadata
            )

        pipeline_summary.append({
            "job_title": job_title,
            "company": company,
            "url": job_url,
            "status": status,
            "match_score": match_score,
            "screenshot": screenshot_path,
        })

    # 4. Print Executive Run Summary
    logger.info("=" * 60)
    logger.info("📊 AUTOPILOT PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    for idx, item in enumerate(pipeline_summary, 1):
        logger.info(
            f"{idx}. [{item['status'].upper()}] {item['company']} - {item['job_title']} "
            f"| Score: {item['match_score']} | Screenshot: {item['screenshot']}"
        )
    logger.info("=" * 60)

    return pipeline_summary


def main():
    """Main CLI entrypoint."""
    args = parse_arguments()

    if args.print_schema:
        print("\n--- Supabase SQL Migration Schema ---")
        print(SUPABASE_SCHEMA_SQL)
        sys.exit(0)

    boards_list = [b.strip() for b in args.boards.split(",") if b.strip()]
    keywords_list = [k.strip() for k in args.keywords.split(",") if k.strip()]

    run_autopilot(
        boards=boards_list,
        keywords=keywords_list,
        min_score=args.min_score,
        resume_path=args.resume,
        headless=args.headless,
        job_limit=args.limit,
    )


if __name__ == "__main__":
    main()
