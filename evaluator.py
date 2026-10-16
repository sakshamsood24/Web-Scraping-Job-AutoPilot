"""
evaluator.py - Evaluation Layer for Job Autopilot Pipeline
Uses LangChain and OpenAI to evaluate Job Descriptions against a hardcoded
Full-Stack PHP / CodeIgniter candidate profile, computing match scores and
generating tailored answers for application form fields.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

logger = logging.getLogger("JobAutopilot.Evaluator")

# Candidate Profile for Saksham Sood
CANDIDATE_PROFILE = {
    "personal_info": {
        "first_name": "Saksham",
        "last_name": "Sood",
        "full_name": "Saksham Sood",
        "email": "sakshamsood007@gmail.com",
        "phone": "+919882315768",
        "location": "Remote / Global Worldwide",
        "linkedin": "https://www.linkedin.com/in/sakshamsood24/",
        "github": "https://github.com/sakshamsood24",
        "portfolio": "https://github.com/sakshamsood24"
    },
    "professional_summary": (
        "Full Stack Developer with 4+ years of experience building scalable ERP and SaaS applications. "
        "Strong expertise in RESTful API development, database optimization, payment gateway integration, "
        "PHP (CodeIgniter, Laravel), Python, JavaScript, and high-concurrency systems."
    ),
    "years_of_experience": 4,
    "core_skills": [
        "PHP (CodeIgniter, Laravel)",
        "Python",
        "JavaScript (ES6+), React.js, jQuery",
        "MySQL & PostgreSQL database optimization, indexing, and data modeling",
        "RESTful API Development & Payment Gateway Integrations",
        "ASP.NET, Blazor, C#",
        "Linux (CLI proficient, server management), Docker",
        "Git, GitHub Actions, Agile / Kanban"
    ],
    "employment_preferences": {
        "desired_role": "Full-Stack Developer / Backend Engineer",
        "workplace_type": "100% Remote Global / Worldwide",
        "contract_type": "Permanent, Full-Time",
        "notice_period": "1 month",
        "sponsorship_required": False,
        "legally_authorized_to_work": True,
        "salary_expectation_usd": "$10,000 - $110,000 USD per annum"
    }
}


class EvaluationResult(BaseModel):
    """Structured response schema for JD evaluation and application field generation."""
    match_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Candidate-to-job match score from 0 to 100 based on technical and experience alignment."
    )
    is_remote: bool = Field(
        ...,
        description="True if the position permits permanent remote work globally or in the candidate's region, False if on-site or restricted."
    )
    reasoning: str = Field(
        ...,
        description="Concise rationale explaining the score, strengths, and any qualification gaps."
    )
    custom_answers: Dict[str, str] = Field(
        default_factory=dict,
        description="Key-value mapping of common or provided application questions to tailored answers written in first-person."
    )


def _build_evaluation_prompt(job_description: str, form_questions: Optional[List[str]] = None) -> str:
    """Builds the comprehensive user prompt for the LLM evaluation."""
    questions_section = ""
    if form_questions:
        questions_section = f"""
Additional Specific Form Questions to Answer:
{json.dumps(form_questions, indent=2)}
Please provide a concise, professional answer for each of the above questions in the 'custom_answers' dictionary.
"""
    else:
        questions_section = """
Also generate suitable answers for common application questions in the 'custom_answers' dictionary, such as:
- 'years_of_php_experience'
- 'experience_with_codeigniter'
- 'salary_expectations'
- 'notice_period'
- 'remote_work_experience'
- 'why_this_role'
"""

    return f"""
You are evaluating a Job Description for a Full-Stack PHP candidate.

Candidate Profile:
{json.dumps(CANDIDATE_PROFILE, indent=2)}

Job Description:
\"\"\"
{job_description}
\"\"\"

{questions_section}

Evaluation Guidelines:
1. 'match_score': Score 0-100 based on technical stack alignment with Saksham Sood's profile (4+ years PHP, CodeIgniter, Laravel, Python, REST APIs, MySQL).
   - High score (80-100): Strong PHP/Laravel/CodeIgniter or Full-Stack web engineering focus, remote/worldwide friendly.
   - Moderate score (60-79): Modern backend/full-stack (Python, REST APIs, SQL, JavaScript) or adjacent web stack.
   - Low score (<60): Non-matching stacks (e.g., pure Java/Rust/Go/Mobile) or heavy staff/director level exceeding 4 years.
2. 'is_remote': Determine if the role supports remote work AND is geographically eligible for candidate Saksham Sood (located in India, seeking 100% Remote Global/Worldwide positions).
   - Set True: If role is open Worldwide, Anywhere, Global, APAC, India, or allows international remote contractors.
   - Set False: If strictly on-site/hybrid, or explicitly restricted to US/North American residents only, US citizenship required, or security clearance required. If ineligible, score must not exceed 25.
3. 'custom_answers': Answers must be written from the first-person perspective of candidate Saksham Sood, maintaining complete consistency with his resume and experience.

Return your evaluation following the required structured format.
"""


def _heuristic_fallback_evaluation(job_description: str, form_questions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Fallback rule-based evaluator used when OpenAI/Gemini API key is unavailable or offline.
    Ensures pipeline continuity and development testability.
    """
    jd_lower = job_description.lower()
    score = 40
    
    # Keyword checks against Saksham Sood's core stack
    if "php" in jd_lower:
        score += 30
    if "codeigniter" in jd_lower:
        score += 20
    elif "laravel" in jd_lower or "mvc" in jd_lower:
        score += 15
        
    if "full stack" in jd_lower or "full-stack" in jd_lower or "backend" in jd_lower:
        score += 10
    if "mysql" in jd_lower or "sql" in jd_lower or "postgresql" in jd_lower:
        score += 5
    if "python" in jd_lower or "rest" in jd_lower or "api" in jd_lower:
        score += 5

    # Location & geographic eligibility check
    # Candidate Saksham Sood is based in India, seeking 100% Worldwide Remote / Global roles
    strict_restrictions = [
        "must reside in the us", "must reside in the united states", "us citizens only",
        "security clearance", "active secret clearance", "must be located in the united states",
        "must be located in the us", "us only", "authorized to work in the us without sponsorship",
        "must be based in north america", "uk only", "eu citizenship required", "germany only"
    ]
    is_restricted = any(r in jd_lower for r in strict_restrictions)

    is_remote = any(kw in jd_lower for kw in ["remote", "work from home", "anywhere", "worldwide", "distributed team", "apac", "global"])
    if is_restricted:
        is_remote = False
        score = 0
        reasoning = "Geographically ineligible: Posting has strict US/regional residency or citizenship restrictions."
    elif not is_remote:
        score = max(0, score - 30)
        reasoning = f"Heuristic evaluation: Role lacks explicit remote designation (PHP={'php' in jd_lower}, CodeIgniter={'codeigniter' in jd_lower})."
    else:
        reasoning = f"Heuristic evaluation: Remote worldwide match (PHP={'php' in jd_lower}, CodeIgniter={'codeigniter' in jd_lower}, Remote={is_remote})."

    score = min(100, max(0, score))
    
    custom_answers = {
        "years_of_php_experience": "4+ years of professional production experience building scalable applications with PHP and CodeIgniter/Laravel.",
        "experience_with_codeigniter": "Extensive experience with CodeIgniter 3 and 4 developing modular MVC architectures, custom libraries, and REST APIs.",
        "salary_expectations": "$70,000 - $95,000 USD / year (flexible depending on compensation structure).",
        "notice_period": "30 days (available for immediate transition).",
        "remote_work_experience": "4+ years of proven track record delivering results asynchronously in distributed, remote-first engineering teams.",
        "authorized_to_work": "Yes, legally authorized to work remotely as an international contractor / employee without visa sponsorship."
    }

    if form_questions:
        for q in form_questions:
            q_clean = q.lower()
            if "salary" in q_clean or "compensation" in q_clean:
                custom_answers[q] = "$70,000 - $95,000 USD / year"
            elif "php" in q_clean:
                custom_answers[q] = "4+ years of production PHP development (7.4, 8.x) with CodeIgniter and Laravel."
            elif "sponsor" in q_clean or "visa" in q_clean:
                custom_answers[q] = "No, I do not require visa sponsorship."
            elif "authorized" in q_clean or "legally" in q_clean:
                custom_answers[q] = "Yes, I am authorized to work remotely."
            elif "start" in q_clean or "notice" in q_clean:
                custom_answers[q] = "30 days"
            else:
                custom_answers[q] = "4+ years full-stack web engineering experience specializing in PHP, Python, and scalable database systems."

    return {
        "match_score": score,
        "is_remote": is_remote,
        "reasoning": reasoning,
        "custom_answers": custom_answers
    }


def evaluate_job(
    job_description: str,
    form_questions: Optional[List[str]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a job description using LangChain and ChatOpenAI against the candidate profile.
    
    Args:
        job_description: Plain text or markdown of the job description.
        form_questions: Optional list of question strings extracted from the application form.
        model_name: Optional OpenAI model override (default: from OPENAI_MODEL_NAME or 'gpt-4o').
        
    Returns:
        Dict[str, Any]:
            - match_score: int (0-100)
            - is_remote: bool
            - reasoning: str
            - custom_answers: Dict[str, str]
    """
    gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    is_gemini_valid = bool(gemini_key) and not any(p in gemini_key.lower() for p in ["your-", "your_", "dummy", "none"])

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    is_openai_valid = bool(openai_key) and not any(p in openai_key.lower() for p in ["your-", "your_", "sk-placeholder", "none", "dummy"])

    if not is_gemini_valid and not is_openai_valid:
        logger.info("No active GEMINI_API_KEY or OPENAI_API_KEY found. Using fast built-in heuristic evaluator.")
        return _heuristic_fallback_evaluation(job_description, form_questions)

    system_content = (
        "You are an expert technical recruiting evaluator and career agent. "
        "You evaluate job postings on behalf of candidate Saksham Sood, a Full-Stack Developer "
        "with 4+ years of experience specializing in PHP (CodeIgniter, Laravel), Python, and modern web systems. "
        "Candidate is based in India seeking 100% Remote Global/Worldwide positions. "
        "Assess technical overlap objectively, verify remote/geographic eligibility, and craft tailored first-person answers."
    )

    truncated_jd = job_description[:12000] if len(job_description) > 12000 else job_description
    user_prompt = _build_evaluation_prompt(truncated_jd, form_questions)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=system_content), HumanMessage(content=user_prompt)]

        if is_gemini_valid:
            from langchain_google_genai import ChatGoogleGenerativeAI
            chosen_model = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
            logger.info(f"Invoking LangChain ChatGoogleGenerativeAI ({chosen_model}) for JD evaluation...")
            llm = ChatGoogleGenerativeAI(
                model=chosen_model,
                temperature=0.2,
                api_key=gemini_key
            )
        else:
            from langchain_openai import ChatOpenAI
            chosen_model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
            logger.info(f"Invoking LangChain ChatOpenAI ({chosen_model}) for JD evaluation...")
            llm = ChatOpenAI(
                model=chosen_model,
                temperature=0.2,
                api_key=openai_key
            )

        structured_llm = llm.with_structured_output(EvaluationResult)
        result: EvaluationResult = structured_llm.invoke(messages)

        eval_dict = result.model_dump()
        logger.info(
            f"Evaluation complete: Score={eval_dict['match_score']}, "
            f"Remote={eval_dict['is_remote']}, Questions answered={len(eval_dict['custom_answers'])}"
        )
        return eval_dict

    except Exception as e:
        logger.error(f"Error during LangChain LLM evaluation ({e}). Falling back to heuristic evaluation.")
        fallback = _heuristic_fallback_evaluation(job_description, form_questions)
        fallback["reasoning"] += f" (LLM invocation error: {str(e)})"
        return fallback


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_jd = """
    Job Title: Remote Full-Stack PHP Developer
    Location: Remote - Worldwide
    About the Role:
    We are looking for a Full-Stack PHP Engineer with 3-5 years of experience to maintain and enhance
    our core SaaS platform. The legacy backend runs on CodeIgniter and is gradually migrating to Laravel,
    backed by MySQL. Frontend uses responsive JavaScript and HTML5.
    Must be comfortable working across global timezones.
    """
    res = evaluate_job(sample_jd, form_questions=["What is your expected salary?", "Are you authorized to work remotely?"])
    print("\n--- Evaluation Result ---")
    print(json.dumps(res, indent=2))
