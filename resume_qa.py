"""resume_qa.py – Retrieval-augmented QA over a stored resume.

Exports:
    answer_application_question(question: str) -> str
"""

import os
import sys
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# ----------------------------------------------------------------------
# Load environment & set up logger
# ----------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ResumeQA")


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise EnvironmentError(f"{var_name} not set")
    return value


def _get_vector_store() -> PGVector:
    connection_str = _require_env("SUPABASE_CONNECTION_STRING")
    gemini_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=gemini_key,
    )

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="resume_embeddings",
        connection=connection_str,
        use_jsonb=True,
    )
    return vector_store


def _build_retriever():
    store = _get_vector_store()
    return store.as_retriever(search_kwargs={"k": 4})


PROMPT = PromptTemplate.from_template(
    """You are the candidate whose resume has been provided. Answer the following
question **in the first person** and **only using information that appears in the
resume**. If the resume does not contain direct experience with an asked technology,
state that you do not have direct production experience, but briefly mention any
related transferable skill that is present.

Resume excerpts (provided for context):
{context}

Question: {question}

Answer (2-4 concise professional sentences, no fluff):"""
)


def _get_llm():
    gemini_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        temperature=0.2,
        google_api_key=gemini_key,
    )


def _build_chain():
    retriever = _build_retriever()
    llm = _get_llm()
    doc_chain = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever, doc_chain)


def answer_application_question(question: str) -> str:
    """Return a concise first-person answer grounded in the resume.

    Args:
        question: The screening question text.

    Returns:
        A short (2-4 sentence) answer string.
    """
    if not question:
        raise ValueError("Question must be a non-empty string")

    chain = _build_chain()
    result = chain.invoke({"input": question, "question": question})
    answer = result.get("answer") or result.get("output") or ""
    logger.info("Generated answer for screening question.")
    return answer.strip()


if __name__ == "__main__":
    sample_q = (
        "Which backend technologies have you used, and can you describe a project "
        "where you built a REST API?"
    )
    try:
        print("\n--- Sample Resume QA ---")
        print("Question:", sample_q)
        print("Answer :", answer_application_question(sample_q))
    except Exception as exc:
        logger.exception(f"Failed to answer sample question: {exc}")
        sys.exit(1)
