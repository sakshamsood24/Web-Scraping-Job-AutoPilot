"""ingest_resume.py – Ingest a PDF resume into a Supabase pgvector collection.

Usage:
    python ingest_resume.py /path/to/resume.pdf
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# ----------------------------------------------------------------------
# Configuration & Logging
# ----------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ResumeIngestion")


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Missing required environment variable: {var_name}")
        sys.exit(1)
    return value


def _load_resume(pdf_path: Path) -> List[Document]:
    """Load PDF and return Document objects."""
    if not pdf_path.is_file():
        logger.error(f"Resume file not found: {pdf_path}")
        sys.exit(1)

    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from resume.")
    return docs


def _chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "•", "-", " "],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source": "resume",
                "category": "experience",
                "chunk_index": i,
            }
        )
    logger.info(f"Created {len(chunks)} text chunks.")
    return chunks


def _create_vector_store() -> PGVector:
    """Configure a PGVector store pointing to Supabase."""
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


def ingest_resume(pdf_path: str) -> None:
    """Run the complete ingestion pipeline."""
    path = Path(pdf_path).expanduser().resolve()
    docs = _load_resume(path)
    chunks = _chunk_documents(docs)

    store = _create_vector_store()
    logger.info("Upserting chunks into Supabase pgvector collection...")
    store.add_documents(chunks)
    logger.info("Ingestion complete ✅")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python ingest_resume.py /path/to/resume.pdf")
        sys.exit(1)

    resume_file = sys.argv[1]
    ingest_resume(resume_file)
