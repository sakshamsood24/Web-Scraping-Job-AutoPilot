"""
fetcher.py - Discovery Layer for Job Autopilot Pipeline
Fetches job postings from Greenhouse Board APIs and filters opportunities
based on target keywords (e.g., 'remote', 'php', 'full stack').
"""

import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("JobAutopilot.Fetcher")

DEFAULT_KEYWORDS = ["remote", "php", "full stack"]
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"

# Curated list of verified active companies hiring on Greenhouse (3,000+ live jobs)
POPULAR_TECH_COMPANIES = [
    "canonical",
    "gitlab",
    "elastic",
    "cloudflare",
    "stripe",
    "mongodb",
    "reddit",
    "instacart",
    "gusto",
    "figma",
    "twitch"
]


def extract_company_token(board_input: str) -> str:
    """
    Extracts the company token from a given Greenhouse URL or identifier.
    
    Supports:
    - 'https://boards-api.greenhouse.io/v1/boards/{company}/jobs'
    - 'https://boards.greenhouse.io/{company}'
    - 'https://boards.greenhouse.io/{company}/jobs'
    - '{company}'
    """
    cleaned = board_input.strip().rstrip("/")
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        parts = [p for p in parsed.path.split("/") if p]
        
        # Format: /v1/boards/{company}/jobs or /v1/boards/{company}
        if "boards" in parts:
            idx = parts.index("boards")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        
        # Format: boards.greenhouse.io/{company}
        if parts:
            return parts[0]
            
    # Default to raw token string
    return cleaned


def clean_html_content(raw_html: Optional[str]) -> str:
    """
    Cleans raw HTML job description into well-formatted plain text.
    Handles HTML entities and normalizes whitespace.
    """
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        # Collapse excessive newlines
        cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(cleaned_lines)
    except Exception as e:
        logger.warning(f"Failed to parse HTML with BeautifulSoup ({e}); falling back to regex clean.")
        return re.sub(r"<[^>]+>", " ", raw_html).strip()


def matches_filter(job: Dict[str, Any], keywords: List[str]) -> tuple[bool, int]:
    """
    Checks if a job posting contains target keywords.
    Distinguishes role/skill keywords from location modifiers ('remote').
    Returns (is_matched: bool, relevance_score: int).
    """
    if not keywords:
        return True, 1

    title_lower = (job.get("title") or "").lower()
    location_lower = (job.get("location") or "").lower()
    desc_lower = (job.get("description") or "").lower()
    full_text = f"{title_lower} {location_lower} {desc_lower}"

    location_modifiers = {"remote", "hybrid", "onsite", "wfh", "anywhere"}
    role_kws = [k.strip().lower() for k in keywords if k.strip().lower() not in location_modifiers]
    has_remote_pref = any(k.strip().lower() in ("remote", "wfh", "anywhere") for k in keywords)

    # 1. Location / Remote Check
    if has_remote_pref:
        is_remote = any(
            term in full_text
            for term in ["remote", "worldwide", "anywhere", "telecommute", "home based", "virtual", "work from home"]
        )
        if not is_remote:
            return False, 0

    # 2. Role Keywords Check
    if not role_kws:
        return True, 1

    matched_in_title = any(re.search(r"\b" + re.escape(kw) + r"\b", title_lower, re.IGNORECASE) for kw in role_kws)
    matched_in_desc = any(re.search(r"\b" + re.escape(kw) + r"\b", desc_lower, re.IGNORECASE) for kw in role_kws)

    if matched_in_title:
        # Highest relevance: Title directly matches tech keywords
        return True, 10
    elif matched_in_desc:
        # Good relevance: Description contains tech stack
        return True, 5

    return False, 0


def fetch_greenhouse_jobs(
    board_url_or_token: str,
    keywords: Optional[List[str]] = None,
    filter_by_keywords: bool = True,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """
    Fetches job postings from a Greenhouse board and filters them based on keywords.
    
    Args:
        board_url_or_token: Full Greenhouse URL or company slug (e.g. 'canonical').
        keywords: List of keywords to filter by (default: ['remote', 'php', 'full stack']).
        filter_by_keywords: Whether to filter out jobs missing target keywords.
        timeout: Request timeout in seconds.
        
    Returns:
        List[Dict[str, Any]]: Filtered list of job postings sorted by relevance.
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    company_token = extract_company_token(board_url_or_token)
    api_url = f"https://boards-api.greenhouse.io/v1/boards/{company_token}/jobs?content=true"

    logger.info(f"Fetching jobs from Greenhouse API: {api_url} (Company: '{company_token}')")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=timeout)
        if response.status_code == 404:
            logger.error(f"Greenhouse board not found for company '{company_token}' (HTTP 404)")
            return []
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Network error fetching jobs from Greenhouse for '{company_token}': {e}")
        return []

    raw_jobs = data.get("jobs", [])
    logger.info(f"Retrieved {len(raw_jobs)} total raw jobs from '{company_token}'")

    parsed_jobs: List[Dict[str, Any]] = []

    for item in raw_jobs:
        job_id = str(item.get("id", ""))
        title = item.get("title", "").strip()
        location_obj = item.get("location") or {}
        location_name = location_obj.get("name", "").strip() if isinstance(location_obj, dict) else str(location_obj)
        abs_url = item.get("absolute_url") or f"https://boards.greenhouse.io/{company_token}/jobs/{job_id}"
        raw_html = item.get("content", "")
        clean_desc = clean_html_content(raw_html)
        updated_at = item.get("updated_at", "")

        job_record = {
            "id": job_id,
            "company": company_token,
            "title": title,
            "location": location_name,
            "url": abs_url,
            "description": clean_desc,
            "content_html": raw_html,
            "updated_at": updated_at,
            "relevance_score": 1,
        }

        # Apply keyword filtering if enabled
        if filter_by_keywords:
            matched, score = matches_filter(job_record, keywords)
            if matched:
                job_record["relevance_score"] = score
                parsed_jobs.append(job_record)
                logger.debug(f"Matched job: [{title}] ({location_name}) score={score}")
            else:
                logger.debug(f"Filtered out: [{title}] ({location_name})")
        else:
            parsed_jobs.append(job_record)

    # Sort parsed jobs by relevance score descending (title matches first)
    parsed_jobs.sort(key=lambda j: j.get("relevance_score", 0), reverse=True)
    logger.info(f"Discovered {len(parsed_jobs)} matching jobs for keywords {keywords} in '{company_token}'")
    return parsed_jobs


if __name__ == "__main__":
    # Quick standalone test against an open Greenhouse board
    import sys
    test_board = sys.argv[1] if len(sys.argv) > 1 else "automattic"
    logging.basicConfig(level=logging.INFO)
    results = fetch_greenhouse_jobs(test_board)
    print(f"\n--- Found {len(results)} jobs for '{test_board}' ---")
    for j in results[:5]:
        print(f"* {j['title']} | {j['location']} | {j['url']}")
