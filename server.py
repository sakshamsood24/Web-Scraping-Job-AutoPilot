"""
server.py - FastAPI Backend & Live Dashboard Server for Job Autopilot
Provides REST APIs and Server-Sent Events (SSE) for live monitoring,
Supabase synchronization, candidate profile management, and pipeline orchestration.
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
from db import get_supabase_client, SUPABASE_TABLE_NAME
from evaluator import CANDIDATE_PROFILE

app = FastAPI(
    title="Job Autopilot Live Dashboard",
    description="Live control center & analytics for the autonomous job application pipeline",
    version="1.0.0"
)

# Enable CORS for local development flexibility
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
STATIC_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# Mount screenshots directory to serve visual proof images directly
app.mount("/screenshots", StaticFiles(directory=str(SCREENSHOTS_DIR)), name="screenshots")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared log queue for streaming live logs to frontend via SSE
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


# Attach custom handler to root logger
root_logger = logging.getLogger()
sse_handler = LogHandler()
sse_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger.addHandler(sse_handler)


class RunPipelineRequest(BaseModel):
    boards: str = "canonical"
    keywords: str = "remote,php,full stack"
    min_score: int = 70
    limit: int = 5
    headless: bool = True


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main single-page application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(index_file)


@app.get("/api/health")
async def health_check():
    """Check health and Supabase connectivity."""
    sb = get_supabase_client()
    return {
        "status": "online",
        "supabase_connected": sb is not None,
        "supabase_url": os.getenv("SUPABASE_URL", ""),
        "pipeline_running": is_pipeline_running,
    }


@app.get("/api/stats")
async def get_stats():
    """Returns aggregated stats from Supabase applications table."""
    sb = get_supabase_client()
    if not sb:
        return {
            "total_jobs": 0,
            "drafted_count": 0,
            "skipped_count": 0,
            "high_match_count": 0,
            "avg_match_score": 0,
            "total_companies": 0,
        }

    try:
        res = sb.table(SUPABASE_TABLE_NAME).select("id, status, match_score, company").execute()
        rows = res.data or []
        total = len(rows)
        drafted = sum(1 for r in rows if r.get("status") == "drafted")
        skipped = sum(1 for r in rows if "skipped" in (r.get("status") or ""))
        high_match = sum(1 for r in rows if (r.get("match_score") or 0) >= 70)
        scores = [r.get("match_score") for r in rows if r.get("match_score") is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        companies = len(set(r.get("company", "").lower() for r in rows if r.get("company")))

        return {
            "total_jobs": total,
            "drafted_count": drafted,
            "skipped_count": skipped,
            "high_match_count": high_match,
            "avg_match_score": avg_score,
            "total_companies": companies,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {e}")


@app.get("/api/applications")
async def get_applications(
    status: Optional[str] = None,
    company: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 100,
):
    """Returns application records with filtering and sorting."""
    sb = get_supabase_client()
    if not sb:
        return {"data": [], "count": 0}

    try:
        query = sb.table(SUPABASE_TABLE_NAME).select("*").order("id", desc=True).limit(limit)
        if status and status != "all":
            query = query.eq("status", status)
        if company and company != "all":
            query = query.ilike("company", f"%{company}%")
        
        res = query.execute()
        rows = res.data or []

        # Client-side keyword search across title, url, reasoning
        if q:
            q_lower = q.lower()
            rows = [
                r for r in rows
                if q_lower in r.get("title", "").lower()
                or q_lower in r.get("company", "").lower()
                or q_lower in r.get("job_url", "").lower()
                or q_lower in str(r.get("metadata", "")).lower()
            ]

        return {"data": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch applications: {e}")


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

    # Available resumes in assets folder
    available_resumes = [
        {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in ASSETS_DIR.glob("*.pdf")
    ]

    return {
        "candidate": CANDIDATE_PROFILE,
        "active_resume": resume_info,
        "available_resumes": available_resumes,
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

        # Update .env
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


@app.post("/api/select-resume")
async def select_resume(filename: str = Form(...)):
    """Selects an already uploaded resume from assets/ as active."""
    target = ASSETS_DIR / filename
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Resume file not found.")

    rel_path = f"assets/{filename}"
    env_file = BASE_DIR / ".env"
    set_key(str(env_file), "RESUME_PATH", rel_path)
    os.environ["RESUME_PATH"] = rel_path
    return {"success": True, "active_resume": rel_path}


def _execute_pipeline_task(req: RunPipelineRequest):
    """Worker function that runs the autopilot pipeline in background."""
    global is_pipeline_running
    is_pipeline_running = True
    try:
        from main import run_autopilot
        boards_list = [b.strip() for b in req.boards.split(",") if b.strip()]
        kw_list = [k.strip() for k in req.keywords.split(",") if k.strip()]
        resume_path = os.getenv("RESUME_PATH", "assets/Saksham_Sood_Resume(1).pdf")

        logging.info(f"▶️ Launching Autopilot: boards={boards_list}, min_score={req.min_score}, limit={req.limit}")
        run_autopilot(
            boards=boards_list,
            keywords=kw_list,
            min_score=req.min_score,
            resume_path=resume_path,
            headless=req.headless,
            job_limit=req.limit,
        )
        logging.info("🏁 Autopilot run finished successfully.")
    except Exception as e:
        logging.error(f"❌ Error during Autopilot run: {e}")
    finally:
        is_pipeline_running = False


@app.post("/api/run-autopilot")
async def trigger_autopilot(req: RunPipelineRequest, background_tasks: BackgroundTasks):
    """Triggers the autopilot pipeline in the background."""
    global is_pipeline_running
    if is_pipeline_running:
        raise HTTPException(status_code=409, detail="Autopilot is already running a batch. Please wait.")

    background_tasks.add_task(_execute_pipeline_task, req)
    return {"success": True, "message": "Autopilot started in background."}


@app.get("/api/logs/stream")
async def stream_logs():
    """Server-Sent Events endpoint streaming real-time pipeline execution logs."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    log_queues.append(q)

    async def event_generator():
        try:
            # Send initial connected ping
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
    print(f"\n🚀 Starting Job Autopilot Dashboard at http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
