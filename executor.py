"""
executor.py - Execution Layer for Job Autopilot Pipeline
Uses Playwright with stealth plugin to automate headlessly filling out
Greenhouse application forms, uploading the candidate's resume, and
saving review screenshots without submitting (V1 Safety constraint).
"""

import os
import re
import asyncio
import logging
from twocaptcha import TwoCaptcha
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth

from evaluator import CANDIDATE_PROFILE

logger = logging.getLogger("JobAutopilot.Executor")

SCREENSHOTS_DIR = Path("screenshots")
DEFAULT_TIMEOUT_MS = 40000


def _apply_stealth(page: Page) -> None:
    """Applies stealth evasions to the given Playwright page."""
    try:
        stealth = Stealth()
        if hasattr(stealth, "apply_stealth_async"):
            asyncio.create_task(stealth.apply_stealth_async(page))
        elif hasattr(stealth, "apply_stealth_sync"):
            stealth.apply_stealth_sync(page)
    except Exception as e:
        logger.warning(f"Could not initialize Stealth evasions ({e}). Proceeding with standard browser headers.")


async def detect_anti_bot_challenge(page: Page) -> Optional[str]:
    """
    Detects if the page presents a CAPTCHA, Cloudflare challenge, or bot block.
    Returns challenge description if detected, else None.
    """
    try:
        content = (await page.content()).lower()
        title = (await page.title()).lower()

        indicators = [
            ("cloudflare turnstile", "Cloudflare Turnstile challenge detected"),
            ("cf-turnstile", "Cloudflare Turnstile element detected"),
            ("just a moment...", "Cloudflare waiting room detected"),
            ("attention required! | cloudflare", "Cloudflare IP block detected"),
            ("verify you are human", "Human verification challenge detected"),
            ("checking your browser", "Cloudflare browser inspection detected"),
            ("recaptcha", "Google reCAPTCHA challenge detected"),
            ("hcaptcha", "hCaptcha challenge detected"),
            ("access denied", "Access Denied (WAF block)"),
        ]

        for trigger, desc in indicators:
            if trigger in title or trigger in content:
                # Double check with specific challenge element selectors if visible
                challenge_el = await page.query_selector(
                    "iframe[src*='challenges.cloudflare'], .cf-turnstile, #cf-wrapper, div[class*='g-recaptcha']"
                )
                if challenge_el or trigger in title:
                    return desc
        return None
    except Exception as e:
        logger.debug(f"Error checking bot challenge: {e}")
        return None
async def solve_captcha_if_present(page: Page, url: str) -> bool:
    """Detect & solve reCAPTCHA/hCaptcha using 2Captcha.

    Returns True if solved (or no captcha present), False on timeout/failure.
    """
    try:
        # Look for captcha widget
        captcha_el = await page.query_selector('[data-sitekey], .g-recaptcha, .h-captcha')
        if not captcha_el:
            return True  # No captcha
        # Extract sitekey
        sitekey = await captcha_el.get_attribute('data-sitekey')
        if not sitekey:
            logger.warning('CAPTCHA element found but no data-sitekey attribute')
            return False
        api_key = os.getenv('TWOCAPTCHA_API_KEY')
        if not api_key:
            logger.error('TWOCAPTCHA_API_KEY not set in .env')
            return False
        solver = TwoCaptcha(api_key)
        # Determine method based on element class
        class_name = await captcha_el.evaluate('(el) => el.className')
        method = 'hcaptcha' if 'h-captcha' in class_name else 'userrecaptcha'
        solve_kwargs = {
            'method': method,
            'sitekey': sitekey,
            'url': url,
            'json': 1,
        }
        # Run solver with timeout
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(solver.solve_captcha, **solve_kwargs),
                timeout=120,
            )
        except asyncio.TimeoutError:
            logger.error('CAPTCHA solving timed out after 120 seconds')
            return False
        except Exception as e:
            logger.error(f'2Captcha solver error: {e}')
            return False
        token = response.get('code')
        if not token:
            logger.error('2Captcha did not return a token')
            return False
        # Inject token into page
        await page.evaluate('''
            (token) => {
                const rc = document.getElementById('g-recaptcha-response') || document.getElementById('h-captcha-response');
                if (rc) rc.value = token;
                if (window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients) {
                    for (const clientKey in window.___grecaptcha_cfg.clients) {
                        const client = window.___grecaptcha_cfg.clients[clientKey];
                        if (client && client.callback) client.callback(token);
                    }
                }
            }
        ''', token)
        logger.info('CAPTCHA solved and token injected')
        return True
    except Exception as e:
        logger.error(f'Unexpected error in CAPTCHA solving: {e}')
        return False

async def fill_field_by_selectors(
    page: Page,
    selectors: List[str],
    value: str,
    field_name: str
) -> bool:
    """
    Attempts to fill an input field by a cascade of candidate selectors.
    """
    for sel in selectors:
        try:
            element = await page.query_selector(sel)
            if element and await element.is_visible():
                await element.scroll_into_view_if_needed()
                await element.fill(value)
                logger.info(f"Filled '{field_name}' using selector '{sel}'")
                return True
        except Exception as e:
            logger.debug(f"Selector '{sel}' failed for '{field_name}': {e}")
            continue
    return False


async def upload_resume(page: Page, resume_path: str) -> bool:
    """
    Handles <input type="file"> for the resume upload programmatically
    using page.set_input_files().
    """
    resolved_path = Path(resume_path).resolve()
    if not resolved_path.exists():
        logger.error(f"Resume file not found at path: {resolved_path}")
        return False

    file_selectors = [
        "input[type='file'][id*='resume' i]",
        "input[type='file'][name*='resume' i]",
        "input[type='file'][data-qa='resume-upload']",
        "input[type='file'][aria-label*='resume' i]",
        "#resume",
        "input[type='file']",
    ]

    for sel in file_selectors:
        try:
            file_input = await page.query_selector(sel)
            if file_input:
                logger.info(f"Found resume file input selector: '{sel}'. Uploading: {resolved_path.name}")
                await page.set_input_files(sel, str(resolved_path))
                # Allow DOM to reflect attachment state
                await page.wait_for_timeout(1000)
                return True
        except Exception as e:
            logger.debug(f"File upload attempt with selector '{sel}' failed: {e}")
            continue

    logger.warning("Could not find an <input type='file'> element for resume upload.")
    return False


async def map_custom_answers(
    page: Page,
    custom_answers: Dict[str, str],
    candidate_info: Dict[str, Any]
) -> List[str]:
    """
    Maps LLM-generated custom answers and profile links to Greenhouse application form questions.
    Iterates through form field containers (.field, div[id*='custom'], etc.) and fills inputs.
    """
    filled_questions = []

    # Comprehensive answer dictionary including candidate links and preferences
    combined_answers: Dict[str, str] = {
        **custom_answers,
        "linkedin": candidate_info.get("linkedin", ""),
        "github": candidate_info.get("github", ""),
        "website": candidate_info.get("portfolio", ""),
        "portfolio": candidate_info.get("portfolio", ""),
    }

    try:
        # Greenhouse structures questions in containers like .field or .form-field
        field_containers = await page.query_selector_all(
            ".field, .form-field, div[class*='field--'], div[data-qa*='question']"
        )

        for container in field_containers:
            try:
                # Find the label or question prompt text
                label_el = await container.query_selector("label, legend, .label, [data-qa='label']")
                if not label_el:
                    continue

                label_text = (await label_el.inner_text()).strip()
                label_lower = label_text.lower()

                # Skip basic personal fields already handled
                if any(x in label_lower for x in ["first name", "last name", "email", "phone", "resume"]):
                    continue

                # Locate corresponding input, textarea, or select
                input_el = await container.query_selector("input[type='text'], input[type='url'], input:not([type])")
                textarea_el = await container.query_selector("textarea")
                select_el = await container.query_selector("select")

                # Determine the best matching answer from combined_answers
                matched_answer = None
                for q_key, answer_val in combined_answers.items():
                    key_lower = q_key.lower().replace("_", " ")
                    # Match if key keywords appear in question or label
                    if key_lower in label_lower or any(word in label_lower for word in key_lower.split() if len(word) > 3):
                        matched_answer = answer_val
                        break

                # Specific fallback heuristics for common Greenhouse questions
                if not matched_answer:
                    if "linkedin" in label_lower:
                        matched_answer = candidate_info.get("linkedin")
                    elif "github" in label_lower:
                        matched_answer = candidate_info.get("github")
                    elif "website" in label_lower or "portfolio" in label_lower:
                        matched_answer = candidate_info.get("portfolio")
                    elif "salary" in label_lower or "compensation" in label_lower:
                        matched_answer = custom_answers.get("salary_expectations", "$90,000 - $110,000 USD / year")
                    elif "notice" in label_lower or "start" in label_lower:
                        matched_answer = custom_answers.get("notice_period", "2 weeks")
                    elif "sponsor" in label_lower or "visa" in label_lower:
                        matched_answer = "No"
                    elif "authorized" in label_lower or "legally" in label_lower:
                        matched_answer = "Yes"

                if not matched_answer:
                    continue

                # Fill Text Input
                if input_el and await input_el.is_visible():
                    current_val = await input_el.input_value()
                    if not current_val:
                        await input_el.scroll_into_view_if_needed()
                        await input_el.fill(str(matched_answer))
                        filled_questions.append(label_text)
                        logger.info(f"Filled text field: '{label_text}'")

                # Fill Textarea
                elif textarea_el and await textarea_el.is_visible():
                    current_val = await textarea_el.input_value()
                    if not current_val:
                        await textarea_el.scroll_into_view_if_needed()
                        await textarea_el.fill(str(matched_answer))
                        filled_questions.append(label_text)
                        logger.info(f"Filled textarea: '{label_text}'")

                # Select Dropdown
                elif select_el and await select_el.is_visible():
                    options = await select_el.query_selector_all("option")
                    target_opt_val = None
                    ans_lower = str(matched_answer).lower()

                    for opt in options:
                        opt_text = (await opt.inner_text()).lower().strip()
                        opt_val = await opt.get_attribute("value")
                        if opt_val and (ans_lower == opt_text or opt_text.startswith(ans_lower)):
                            target_opt_val = opt_val
                            break

                    if target_opt_val:
                        await select_el.select_option(value=target_opt_val)
                        filled_questions.append(label_text)
                        logger.info(f"Selected option for '{label_text}': {target_opt_val}")

            except Exception as item_err:
                logger.debug(f"Error mapping form field container: {item_err}")
                continue

    except Exception as e:
        logger.warning(f"Error discovering custom questions: {e}")

    return filled_questions


async def fill_job_application(
    job_url: str,
    custom_answers: Dict[str, str],
    resume_path: Optional[str] = None,
    candidate_profile: Optional[Dict[str, Any]] = None,
    headless: bool = True,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Dict[str, Any]:
    """
    Automates loading the job posting, navigating to the Greenhouse application form,
    populating standard personal details, uploading the resume, and filling custom answers.
    
    SAFETY CONSTRAINT (V1):
    This function will NEVER click the final submit button. It takes and saves a
    local screenshot of the filled form for human review.
    
    Returns:
        Dict[str, Any]:
            - success: bool
            - job_url: str
            - screenshot_path: Optional[str]
            - fields_filled: List[str]
            - error: Optional[str]
    """
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    profile = candidate_profile or CANDIDATE_PROFILE
    personal = profile.get("personal_info", {})
    resolved_resume = resume_path or os.getenv("RESUME_PATH", "assets/Saksham_Sood_Resume(1).pdf")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_url_slug = re.sub(r"[^a-zA-Z0-9]+", "_", job_url.split("/")[-1] or "job")[:30]
    screenshot_file = SCREENSHOTS_DIR / f"application_{safe_url_slug}_{timestamp_str}.png"

    result: Dict[str, Any] = {
        "success": False,
        "job_url": job_url,
        "screenshot_path": None,
        "fields_filled": [],
        "error": None
    }

    async with async_playwright() as p:
        browser = None
        try:
            logger.info(f"Launching Chromium (headless={headless}) with stealth plugin...")
            browser = await p.chromium.launch(
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-infobars",
                    "--window-position=0,0",
                    "--ignore-certifcate-errors",
                    "--ignore-certifcate-errors-spki-list",
                ]
            )

            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
            )

            page = await context.new_page()
            _apply_stealth(page)

            logger.info(f"Navigating to job posting: {job_url}")
            try:
                await page.goto(job_url, wait_until="domcontentloaded", timeout=timeout_ms)
            except PlaywrightTimeoutError:
                logger.error(f"Navigation timed out after {timeout_ms}ms on {job_url}")
                result["error"] = "Navigation timeout"
                return result

            # Check for bot / CAPTCHA challenge
            challenge_msg = await detect_anti_bot_challenge(page)
            if challenge_msg:
                captcha_screenshot = SCREENSHOTS_DIR / f"challenge_{safe_url_slug}_{timestamp_str}.png"
                await page.screenshot(path=str(captcha_screenshot))
                logger.warning(f"Bot challenge encountered: {challenge_msg}. Screenshot saved to {captcha_screenshot}")
                result["error"] = f"Bot Challenge: {challenge_msg}"
                result["screenshot_path"] = str(captcha_screenshot)
                return result

            # Attempt CAPTCHA solving before finalizing form
            try:
                solved = await solve_captcha_if_present(page, job_url)
                if not solved:
                    logger.warning("CAPTCHA solving failed or not present. Marking job as captcha_failed.")
                    result["error"] = "CAPTCHA solving failed"
                    # No direct Supabase update here; main pipeline will handle status logging
                    return result
            except Exception as cap_err:
                logger.error(f"Error during CAPTCHA solving: {cap_err}")
                result["error"] = f"CAPTCHA error: {cap_err}"
                return result
            # If application form is inside an iframe or behind an "Apply Now" button
            apply_btn = await page.query_selector(
                "a[href*='#app'], a[href*='apply'], button:has-text('Apply Now'), a:has-text('Apply Now')"
            )
            if apply_btn and await apply_btn.is_visible():
                try:
                    await apply_btn.click()
                    await page.wait_for_timeout(1000)
                except Exception as click_err:
                    logger.debug(f"Could not click 'Apply Now' button: {click_err}")

            # Wait briefly for form rendering
            await page.wait_for_timeout(1500)

            # Fill Standard Fields
            fields_filled = []

            # 1. First Name
            if await fill_field_by_selectors(
                page,
                ["#first_name", "input[name='first_name']", "input[id*='first_name' i]"],
                personal.get("first_name", "Saksham"),
                "First Name"
            ):
                fields_filled.append("first_name")

            # 2. Last Name
            if await fill_field_by_selectors(
                page,
                ["#last_name", "input[name='last_name']", "input[id*='last_name' i]"],
                personal.get("last_name", "Sood"),
                "Last Name"
            ):
                fields_filled.append("last_name")

            # 3. Email
            if await fill_field_by_selectors(
                page,
                ["#email", "input[name='email']", "input[type='email']", "input[id*='email' i]"],
                personal.get("email", "sakshamsood007@gmail.com"),
                "Email"
            ):
                fields_filled.append("email")

            # 4. Phone
            if await fill_field_by_selectors(
                page,
                ["#phone", "input[name='phone']", "input[type='tel']", "input[id*='phone' i]"],
                personal.get("phone", "+919882315768"),
                "Phone"
            ):
                fields_filled.append("phone")

            # 5. Programmatic Resume Upload via page.set_input_files()
            if await upload_resume(page, resolved_resume):
                fields_filled.append("resume_upload")

            # 6. Map LLM Custom Answers and Profile Links
            custom_mapped = await map_custom_answers(page, custom_answers, personal)
            fields_filled.extend(custom_mapped)

            # 7. Take Full-Page Screenshot of populated form for review
            await page.wait_for_timeout(1000)
            await page.screenshot(path=str(screenshot_file), full_page=True)
            logger.info(f"Saved filled form screenshot to: {screenshot_file}")

            result["success"] = True
            result["screenshot_path"] = str(screenshot_file)
            result["fields_filled"] = fields_filled
            logger.info(f"Application autofill successful! Fields filled: {len(fields_filled)}")
            return result

        except Exception as e:
            logger.error(f"Unexpected error executing Playwright application form fill on {job_url}: {e}")
            result["error"] = str(e)
            try:
                err_screenshot = SCREENSHOTS_DIR / f"error_{safe_url_slug}_{timestamp_str}.png"
                if page:
                    await page.screenshot(path=str(err_screenshot))
                    result["screenshot_path"] = str(err_screenshot)
            except Exception:
                pass
            return result

        finally:
            if browser:
                await browser.close()


def fill_job_application_sync(
    job_url: str,
    custom_answers: Dict[str, str],
    resume_path: Optional[str] = None,
    headless: bool = True
) -> Dict[str, Any]:
    """Synchronous convenience wrapper around fill_job_application."""
    return asyncio.run(
        fill_job_application(
            job_url=job_url,
            custom_answers=custom_answers,
            resume_path=resume_path,
            headless=headless
        )
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://job-boards.greenhouse.io/canonical/jobs/5569916"
    test_answers = {
        "salary_expectations": "$95,000 USD / year",
        "notice_period": "2 weeks",
        "years_of_php_experience": "4 years",
    }
    print(f"\n--- Testing Playwright Form Fill on: {test_url} ---")
    out = fill_job_application_sync(test_url, test_answers, headless=True)
    print("Result:", out)
