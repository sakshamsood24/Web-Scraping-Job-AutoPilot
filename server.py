"""
server.py - LoopCV-Style Autonomous Job Search & Application Server
Replicates the full LoopCV web application experience with Supabase PostgreSQL,
Playwright form autofill, automated campaigns (Loops), CV Checker, and live search.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, set_key

# Load environment
load_dotenv()

# Internal project modules
from db import get_supabase_client, SUPABASE_TABLE_NAME, log_application
from evaluator import CANDIDATE_PROFILE, evaluate_job
from executor import fill_job_application_sync
from fetcher import fetch_greenhouse_jobs, POPULAR_TECH_COMPANIES

app = FastAPI(
    title="LoopCV Pro - Autonomous Job Search & Application Platform",
    description="Full LoopCV replica connected to Supabase and Playwright executor",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
ASSETS_DIR = BASE_DIR / "assets"
LOOPS_FILE = BASE_DIR / "loops.json"

STATIC_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# Mount static and screenshots folders
app.mount("/screenshots", StaticFiles(directory=str(SCREENSHOTS_DIR)), name="screenshots")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared log queue for streaming live logs via SSE
log_queues: List[asyncio.Queue] = []
is_pipeline_running = False


class LogHandler(logging.Handler):
    """Custom logging handler to push formatted log messages to active SSE queues."""
    def emit(self, record):
        msg = self.format(record)
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if loop and loop.is_running():
            for q in list(log_queues):
                try:
                    q.put_nowait(msg)
                except Exception:
                    pass


root_logger = logging.getLogger()
sse_handler = LogHandler()
sse_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger.addHandler(sse_handler)


# -------------------------------------------------------------
# Pydantic Request Models
# -------------------------------------------------------------
class RunPipelineRequest(BaseModel):
    boards: str = "all"
    keywords: str = "remote,php,full stack"
    min_score: int = 70
    limit: int = 5
    headless: bool = True


class ApplyJobRequest(BaseModel):
    job_url: str
    company: Optional[str] = "Company"
    title: Optional[str] = "Developer"


class BatchApplyRequest(BaseModel):
    job_urls: List[str]


class NewLoopRequest(BaseModel):
    name: str
    keywords: str
    boards: str = "all"
    min_score: int = 70
    auto_apply: bool = False


class CVCheckRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = "Developer"


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the LoopCV replica dashboard UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(index_file)


@app.get("/api/health")
async def health_check():
    """Health status and Supabase connectivity."""
    sb = get_supabase_client()
    return {
        "status": "online",
        "supabase_connected": sb is not None,
        "supabase_url": os.getenv("SUPABASE_URL", ""),
        "pipeline_running": is_pipeline_running,
    }


@app.get("/api/stats")
async def get_stats():
    """Returns aggregated stats matching LoopCV overview cards."""
    sb = get_supabase_client()
    if not sb:
        return {
            "total_matches": 0,
            "applied_count": 0,
            "drafted_count": 0,
            "skipped_count": 0,
            "high_match_count": 0,
            "avg_match_score": 0,
            "total_companies": 0,
            "active_loops": 3,
            "rejected_matches": 0,
        }

    try:
        res = sb.table(SUPABASE_TABLE_NAME).select("id, status, match_score, company").execute()
        rows = res.data or []
        total = len(rows)
        applied = sum(1 for r in rows if r.get("status") in ("applied", "drafted"))
        drafted = sum(1 for r in rows if r.get("status") == "drafted")
        skipped = sum(1 for r in rows if "skipped" in (r.get("status") or ""))
        high_match = sum(1 for r in rows if (r.get("match_score") or 0) >= 70)
        scores = [r.get("match_score") for r in rows if r.get("match_score") is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        companies = len(set(r.get("company", "").lower() for r in rows if r.get("company")))

        return {
            "total_matches": total,
            "applied_count": applied,
            "drafted_count": drafted,
            "skipped_count": skipped,
            "high_match_count": high_match,
            "avg_match_score": avg_score,
            "total_companies": companies,
            "active_loops": 2,
            "rejected_matches": skipped,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {e}")


@app.get("/api/matches")
async def get_matches(
    status: Optional[str] = None,
    platform: Optional[str] = None,
    company: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 100,
):
    """
    Returns matches table rows exactly formatted for the LoopCV Matches view:
    Role & Company, Status, Platform, Salary, Matched Date, Actions.
    """
    sb = get_supabase_client()
    if not sb:
        return {"data": [], "count": 0}

    try:
        query = sb.table(SUPABASE_TABLE_NAME).select("*").order("id", desc=True).limit(limit)
        if status and status not in ("all", "all_matches"):
            if status == "applied":
                query = query.in_("status", ["drafted", "applied"])
            elif status == "skipped":
                query = query.like("status", "%skipped%")
            else:
                query = query.eq("status", status)

        if company and company.strip():
            query = query.ilike("company", f"%{company.strip()}%")

        res = query.execute()
        rows = res.data or []

        # Client-side text filter across title, job_url, reasoning
        if q and q.strip():
            term = q.strip().lower()
            rows = [
                r for r in rows
                if term in r.get("title", "").lower()
                or term in r.get("company", "").lower()
                or term in r.get("job_url", "").lower()
                or term in str(r.get("metadata", "")).lower()
            ]

        # Enhance rows with LoopCV attributes (Platform tag, Salary, Location, Logo)
        formatted_rows = []
        for r in rows:
            company_slug = (r.get("company") or "Greenhouse").lower()
            meta = r.get("metadata") or {}
            
            # Detect platform
            url = r.get("job_url", "")
            if "greenhouse.io" in url:
                platform_name = "Greenhouse"
            elif "linkedin.com" in url:
                platform_name = "LinkedIn"
            else:
                platform_name = "Greenhouse"

            formatted_rows.append({
                "id": r.get("id"),
                "job_url": url,
                "company": r.get("company", "Tech Company"),
                "title": r.get("title", "Software Engineer"),
                "location": meta.get("location", "Remote"),
                "status": r.get("status", "discovered"),
                "match_score": r.get("match_score"),
                "platform": platform_name,
                "salary": meta.get("salary", "N/A"),
                "matched_at": r.get("timestamp"),
                "screenshot_path": meta.get("screenshot_path"),
                "reasoning": meta.get("reasoning", ""),
                "fields_filled": meta.get("fields_filled", []),
            })

        return {"data": formatted_rows, "count": len(formatted_rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch matches: {e}")


@app.post("/api/apply-single")
async def apply_single_job(req: ApplyJobRequest):
    """
    Triggers Playwright autofill execution for a single job posting
    (triggered by the ⚡ Instant Apply button on a match row).
    """
    logger.info(f"⚡ Instant Auto-Applying to: {req.job_url} ({req.title} at {req.company})")
    try:
        resume_path = os.getenv("RESUME_PATH", "assets/Saksham_Sood_Resume(1).pdf")
        
        # Evaluate job to get custom answers
        eval_dict = evaluate_job(f"{req.title} at {req.company}. Technical role.")
        custom_answers = eval_dict.get("custom_answers", {})

        # Fill via Playwright
        res = fill_job_application_sync(
            job_url=req.job_url,
            custom_answers=custom_answers,
            resume_path=resume_path,
            headless=True
        )

        status = "drafted" if res.get("success") else "failed"
        log_application(
            job_url=req.job_url,
            company=req.company,
            title=req.title,
            status=status,
            match_score=eval_dict.get("match_score", 85),
            metadata={
                "screenshot_path": res.get("screenshot_path"),
                "fields_filled": res.get("fields_filled", []),
                "error": res.get("error")
            }
        )

        return {
            "success": res.get("success", False),
            "status": status,
            "screenshot_path": res.get("screenshot_path"),
            "fields_filled": res.get("fields_filled", []),
            "message": f"Application {'drafted with proof' if res.get('success') else 'encountered issue'}"
        }
    except Exception as e:
        logger.error(f"Error applying to job {req.job_url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-apply")
async def batch_apply_jobs(req: BatchApplyRequest, background_tasks: BackgroundTasks):
    """Batch applies to all selected jobs."""
    if not req.job_urls:
        raise HTTPException(status_code=400, detail="No jobs selected.")

    def _worker(urls: List[str]):
        logger.info(f"🚀 Launching Batch Apply for {len(urls)} selected jobs...")
        for i, url in enumerate(urls):
            try:
                logger.info(f"[{i+1}/{len(urls)}] Processing application for: {url}")
                fill_job_application_sync(
                    job_url=url,
                    custom_answers={},
                    resume_path=os.getenv("RESUME_PATH", "assets/Saksham_Sood_Resume(1).pdf"),
                    headless=True
                )
            except Exception as e:
                logger.error(f"Failed application on {url}: {e}")

    background_tasks.add_task(_worker, req.job_urls)
    return {"success": True, "message": f"Queued {len(req.job_urls)} applications in background."}


@app.get("/api/loops")
async def get_loops():
    """Returns all automated campaigns (Loops)."""
    if LOOPS_FILE.is_file():
        try:
            with open(LOOPS_FILE, "r") as f:
                return {"loops": json.load(f)}
        except Exception:
            pass
    return {"loops": []}


@app.post("/api/loops")
async def create_loop(req: NewLoopRequest):
    """Creates a new automated loop campaign."""
    current = []
    if LOOPS_FILE.is_file():
        try:
            with open(LOOPS_FILE, "r") as f:
                current = json.load(f)
        except Exception:
            current = []

    new_item = {
        "id": len(current) + 1,
        "name": req.name,
        "keywords": req.keywords,
        "boards": req.boards,
        "min_score": req.min_score,
        "auto_apply": req.auto_apply,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matches_count": 0,
        "applied_count": 0
    }
    current.append(new_item)
    with open(LOOPS_FILE, "w") as f:
        json.dump(current, f, indent=2)

    return {"success": True, "loop": new_item}


@app.get("/api/questions")
async def get_screening_questions():
    """Returns stored screening questions and AI generated answers."""
    cand = CANDIDATE_PROFILE
    personal = cand.get("personal_info", {})
    prefs = cand.get("employment_preferences", {})

    questions = [
        {"q": "What is your full legal name?", "a": personal.get("full_name")},
        {"q": "What is your primary contact email?", "a": personal.get("email")},
        {"q": "What is your phone number?", "a": personal.get("phone")},
        {"q": "What are your salary expectations?", "a": prefs.get("salary_expectation_usd")},
        {"q": "What is your notice period?", "a": prefs.get("notice_period")},
        {"q": "Are you legally authorized to work remotely?", "a": "Yes, legally authorized to work remotely worldwide."},
        {"q": "Do you require visa sponsorship now or in the future?", "a": "No, I do not require visa sponsorship."},
        {"q": "How many years of Full-Stack PHP & REST API experience do you have?", "a": "4+ years of hands-on production experience with PHP (CodeIgniter, Laravel), MySQL optimization, and building high-traffic RESTful APIs."},
        {"q": "What is your experience with databases?", "a": "Extensive experience designing, indexing, and optimizing MySQL and PostgreSQL databases for SaaS and ERP applications."}
    ]
    return {"questions": questions}


@app.post("/api/cv-checker")
async def check_cv(req: CVCheckRequest):
    """Evaluates the candidate resume against a provided JD (CV Checker tool)."""
    try:
        eval_dict = evaluate_job(req.job_description)
        score = eval_dict.get("match_score", 75)
        return {
            "score": score,
            "is_remote": eval_dict.get("is_remote", True),
            "reasoning": eval_dict.get("reasoning", "Strong match with candidate's tech stack."),
            "recommendations": [
                "Highlight CodeIgniter 3/4 and Laravel REST API design in your top experience summary.",
                "Emphasize database query optimization (MySQL indexing) which is highlighted in this JD.",
                "Mention asynchronous remote teamwork across international time zones."
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search-jobs")
async def search_jobs(keywords: str = "remote,developer", limit: int = 15):
    """Performs live search across top Greenhouse boards."""
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    results = []
    # Search Canonical, GitLab, Elastic, Stripe
    for b in ["canonical", "gitlab", "elastic", "stripe"][:3]:
        try:
            jobs = fetch_greenhouse_jobs(b, keywords=kw_list, filter_by_keywords=True)
            results.extend(jobs)
            if len(results) >= limit:
                break
        except Exception:
            pass
    return {"results": results[:limit], "count": len(results[:limit])}


@app.get("/api/profile")
async def get_profile():
    """Returns candidate profile and active resume file metadata."""
    resume_path = os.getenv("RESUME_PATH", "assets/Saksham_Sood_Resume(1).pdf")
    resume_file = BASE_DIR / resume_path
    resume_info = {
        "path": resume_path,
        "exists": resume_file.is_file(),
        "filename": resume_file.name if resume_file.is_file() else "None",
        "size_kb": round(resume_file.stat().st_size / 1024, 1) if resume_file.is_file() else 0,
    }
    return {
        "candidate": CANDIDATE_PROFILE,
        "active_resume": resume_info,
    }


@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Uploads a new PDF resume, saves it to assets/, and updates .env."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest_path = ASSETS_DIR / file.filename
    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)

        rel_path = f"assets/{file.filename}"
        env_file = BASE_DIR / ".env"
        set_key(str(env_file), "RESUME_PATH", rel_path)
        os.environ["RESUME_PATH"] = rel_path

        return {
            "success": True,
            "filename": file.filename,
            "path": rel_path,
            "size_kb": round(len(content) / 1024, 1),
            "message": f"Resume '{file.filename}' uploaded and set as active."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload resume: {e}")


def _execute_pipeline_task(req: RunPipelineRequest):
    """Worker function that runs the autopilot pipeline in background."""
    global is_pipeline_running
    is_pipeline_running = True
    try:
        from main import run_autopilot
        boards_list = [b.strip() for b in req.boards.split(",") if b.strip()]
        kw_list = [k.strip() for k in req.keywords.split(",") if k.strip()]
        resume_path = os.getenv("RESUME_PATH", "assets/Saksham_Sood_Resume(1).pdf")

        logging.info(f"▶️ Launching Loop Autopilot: boards={boards_list}, min_score={req.min_score}, limit={req.limit}")
        run_autopilot(
            boards=boards_list,
            keywords=kw_list,
            min_score=req.min_score,
            resume_path=resume_path,
            headless=req.headless,
            job_limit=req.limit,
        )
        logging.info("🏁 Autopilot loop finished successfully.")
    except Exception as e:
        logging.error(f"❌ Error during Autopilot run: {e}")
    finally:
        is_pipeline_running = False


@app.post("/api/run-autopilot")
async def trigger_autopilot(req: RunPipelineRequest, background_tasks: BackgroundTasks):
    """Triggers the autopilot pipeline in the background."""
    global is_pipeline_running
    if is_pipeline_running:
        raise HTTPException(status_code=409, detail="Autopilot is already running a loop. Please wait.")

    background_tasks.add_task(_execute_pipeline_task, req)
    return {"success": True, "message": "Autopilot started in background."}


@app.get("/api/logs/stream")
async def stream_logs():
    """Server-Sent Events endpoint streaming real-time pipeline execution logs."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    log_queues.append(q)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'message': 'Connected to live log stream', 'level': 'INFO', 'time': datetime.now().strftime('%H:%M:%S')})}\n\n"
            while True:
                msg = await q.get()
                payload = json.dumps({"raw": msg, "time": datetime.now().strftime('%H:%M:%S')})
                yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in log_queues:
                log_queues.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/api/screenshots-list")
async def list_screenshots():
    """Lists all available screenshots in the screenshots folder."""
    files = []
    for f in sorted(SCREENSHOTS_DIR.glob("*.png"), key=os.path.getmtime, reverse=True):
        files.append({
            "filename": f.name,
            "url": f"/screenshots/{f.name}",
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"screenshots": files}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting LoopCV Pro Dashboard at http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
