# Headless Autonomous Job Application Pipeline ("Autopilot")

An autonomous, end-to-end Python pipeline engineered to discover remote **Full-Stack PHP / CodeIgniter** roles, evaluate them using **LangChain & OpenAI**, log applications to a **Supabase PostgreSQL** database, and headlessly autofill Greenhouse job applications using **Playwright** with stealth evasions.

---

## 🏗️ Architecture & Component Overview

```
                      +-----------------------------------+
                      |       Greenhouse API Boards       |
                      +-----------------+-----------------+
                                        |
                                        v
                            [ fetcher.py: Discovery ]
                            * Greenhouse REST endpoint
                            * Keyword filter: remote, php,
                              full stack
                                        |
                                        v
   +--------------------+   [ main.py: Orchestrator ]   +-----------------------+
   |   Supabase (DB)    |<----------------------------->| evaluator.py (LLM)    |
   |   applications     |   1. Deduplication check      | * LangChain + OpenAI  |
   |   PostgreSQL table |   2. Log status & scores      | * Candidate Profile   |
   +--------------------+                               | * Match score & Q&A   |
                                        |               +-----------------------+
                                        v
                          [ executor.py: Playwright ]
                          * Stealth plugin evasions
                          * Form fields mapping
                          * page.set_input_files() [Resume]
                          * Full-page verification screenshot
                          * ⚠️ V1 Safety: NO final submit
```

### Module Responsibilities:
1. **`db.py` (Database Layer)**:
   - Connects to Supabase PostgreSQL using the official `supabase` Python client.
   - Manages table `applications` (`id`, `job_url`, `company`, `title`, `status`, `match_score`, `timestamp`, `metadata`).
   - Deduplication check (`is_already_applied`) to prevent double-applying.
   - Resilient offline fallback cache so local development is frictionless before database configuration.

2. **`fetcher.py` (Discovery Layer)**:
   - Connects to public Greenhouse REST APIs (`https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true`).
   - Normalizes board URLs and company tokens.
   - Converts raw HTML job descriptions into clean text via BeautifulSoup.
   - Filters positions by target keywords: `"remote"`, `"php"`, or `"full stack"`.

3. **`evaluator.py` (Evaluation Layer)**:
   - Contains candidate profile for Alex Morgan (4 years Full-Stack PHP developer with CodeIgniter 3/4, Laravel, MySQL, REST APIs, permanent remote worldwide).
   - Supports **Google Gemini API** (Free Tier via `langchain-google-genai`), **OpenAI**, or built-in heuristic rule-based engine.
   - Evaluates match scores (0-100), checks remote suitability, and generates first-person answers to application form questions.

4. **`executor.py` (Execution Layer)**:
   - Launches headless Chromium with `playwright-stealth` evasions.
   - Integrated **2Captcha** API to automatically detect and solve reCAPTCHA and hCaptcha challenges.
   - Detects anti-bot/CAPTCHA/Cloudflare challenges and exits gracefully with diagnostic screenshots.
   - Fills personal contact fields (`first_name`, `last_name`, `email`, `phone`).
   - **Resume Upload**: Uses `page.set_input_files()` to attach candidate PDF files programmatically.
   - Maps LLM answers to custom questions, textareas, and dropdown selects.
   - **V1 Safety**: **Never clicks final submit**. Instead, takes a high-resolution full-page screenshot saved locally to `screenshots/`.

5. **`ingest_resume.py` & `resume_qa.py` (Resume RAG Engine)**:
   - Ingests candidate PDF resumes using `PyMuPDFLoader` and splits them into overlapping semantic chunks.
   - Stores chunk embeddings in Supabase PostgreSQL using `PGVector` and Gemini `text-embedding-004`.
   - Generates truthful, first-person answers to application screening questions grounded strictly in the resume context.

6. **`main.py` (Orchestrator)**:
   - Coordinates the entire lifecycle, CLI arguments, error recovery, and prints executive run summaries.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 3. Environment Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Edit `.env` with your credentials:
```ini
# Google Gemini API (100% Free tier: https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL_NAME=gemini-2.5-flash

# Supabase PostgreSQL
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_role_key

# Autopilot Settings
RESUME_PATH=assets/resume_alex_morgan.pdf
MIN_MATCH_SCORE=70
HEADLESS=true
DEFAULT_GREENHOUSE_BOARDS=canonical,automattic,gitlab
```

---

## 🗄️ Supabase PostgreSQL Setup

Run the following SQL snippet in your Supabase SQL Editor:
```sql
CREATE TABLE IF NOT EXISTS applications (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    job_url TEXT UNIQUE NOT NULL,
    company TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    match_score INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_applications_job_url ON applications(job_url);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
```
*(You can also print this schema directly by running `python3 main.py --print-schema`)*.

---

## 💻 Usage Guide

### 🌟 Launch Live Web Dashboard (Recommended)
Launch the interactive command center UI to monitor Supabase applications, inspect match scores, view filled application screenshots, upload resumes, and trigger runs:
```bash
source .venv/bin/activate
python3 server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

### Run Autopilot from Terminal (Standard Run)
```bash
source .venv/bin/activate
python3 main.py --boards canonical --limit 5 --min-score 75
```

### Run with Visual Browser (Debug Mode)
```bash
python3 main.py --boards canonical --limit 1 --no-headless
```

### Inspect Output Screenshots
Screenshots of drafted applications are stored in:
```bash
ls -lh screenshots/
```

### Run Automated Tests
```bash
python3 test_pipeline.py
```

---

## 🛡️ Safety & Anti-Detection
- **Stealth Evasions**: Obfuscates `navigator.webdriver`, vendor overrides, and automation flags.
- **Safety Lock (V1)**: Code strictly terminates before triggering `#submit_app`.
- **Deduplication Protection**: Checks database before page navigation to ensure no job URL is processed twice.
