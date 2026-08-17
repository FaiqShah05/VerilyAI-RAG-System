"""VerilyAI — single-file Streamlit RAG app.
"""

Everything (config, errors, sources, chunking, embeddings, ingestion,
vector store, RAG logic, transcript export, and the UI) lives in this
one file on purpose, so deployment needs zero folder structure — just
this file plus requirements.txt.

Run locally:  streamlit run app.py
Deploy:       push this file + requirements.txt to a GitHub repo,
              point Streamlit Community Cloud at app.py, and add
              GOOGLE_API_KEY under Manage app -> Settings -> Secrets.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
import secrets
import shutil
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

import requests
import streamlit as st
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError


# ===========================================================================
# Errors
# ===========================================================================

class ArzensError(Exception):
    """Base class. Anything the UI catches is one of these."""

    default_message = "Something went wrong."

    def __init__(
        self,
        user_message: str | None = None,
        *,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.user_message = user_message or self.default_message
        self.hint = hint
        self.cause = cause
        super().__init__(self.user_message)

    def display(self) -> str:
        return f"{self.user_message}\n\n{self.hint}" if self.hint else self.user_message

    def technical_detail(self) -> str:
        if self.cause is None:
            return "No additional detail."
        return f"{type(self.cause).__name__}: {self.cause}"


class ConfigError(ArzensError):
    default_message = "The app is not configured correctly."


class IngestionError(ArzensError):
    default_message = "That source could not be ingested."


class WebFetchError(IngestionError):
    default_message = "That website could not be read."


class PDFParseError(IngestionError):
    default_message = "That PDF could not be read."


class EmptyContentError(IngestionError):
    default_message = "That source contained no readable text."


class EmbeddingError(ArzensError):
    default_message = "The documents could not be embedded."


class VectorStoreError(ArzensError):
    default_message = "The knowledge base could not be opened."


class GenerationError(ArzensError):
    default_message = "The assistant could not produce an answer."


# ===========================================================================
# Config
# ===========================================================================

CHAT_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

RETRIEVAL_K = 6
RELEVANCE_FLOOR = 0.30
HISTORY_TURNS = 6

MAX_PDF_MB = 25
MAX_PDFS_PER_UPLOAD = 10
MAX_URLS_PER_INGEST = 10
MAX_CHARS_PER_SOURCE = 600_000
WEB_TIMEOUT_SECONDS = 45

import os  # noqa: E402  (kept near the paths it configures)

DATA_DIR = Path(os.environ.get("ARZENS_DATA_DIR", ".arzens_data"))
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "embedding_cache"
MANIFEST_DIR = DATA_DIR / "manifests"

WORKSPACE_TTL_HOURS = 48

APP_TITLE = "VerilyAI"
APP_ICON = "📜"

# Brand palette — an "ink and seal" theme: deep verified-navy, a gold
# seal accent for citations/trust, and a muted teal for success states.
BRAND = {
    "navy": "#132A3E",
    "navy_light": "#1E3E58",
    "ink": "#1A1A1A",
    "paper": "#F7F5F0",
    "paper_dim": "#EDEAE1",
    "gold": "#C6952C",
    "gold_light": "#E0B84B",
    "teal": "#2F6F62",
    "rose": "#B0463A",
}

STARTER_QUESTIONS = [
    "What is this knowledge base about?",
    "Summarise the key points in three bullets.",
    "What does the source say about pricing or cost?",
    "List any dates, deadlines, or timelines mentioned.",
]


@dataclass(frozen=True)
class GroundingPolicy:
    key: str
    label: str
    description: str
    instruction: str


STRICT = GroundingPolicy(
    key="strict",
    label="Strict — documents only",
    description=(
        "Answers come only from the ingested sources. If the passages do not "
        "contain the answer, the assistant says so instead of guessing."
    ),
    instruction=(
        "Answer ONLY from the CONTEXT passages above. You have no other "
        "knowledge available for this task.\n"
        "If the CONTEXT does not contain enough information to answer, reply "
        "exactly in this shape: state plainly that the knowledge base does not "
        "cover it, then name in one short line what the sources *do* cover that "
        "is closest to the question. Do not answer from general knowledge, and "
        "do not speculate about what the documents probably say."
    ),
)

HYBRID = GroundingPolicy(
    key="hybrid",
    label="Hybrid — documents first, general knowledge labelled",
    description=(
        "Prefers the ingested sources. May add general knowledge, but flags it "
        "clearly so sourced facts are never confused with model knowledge."
    ),
    instruction=(
        "Answer from the CONTEXT passages above wherever they cover the "
        "question. If the CONTEXT is insufficient, you may add general "
        "knowledge, but you MUST place it under a final line that reads exactly:\n"
        "> **Outside the knowledge base:**\n"
        "Never blend unsourced claims into the sourced part of the answer."
    ),
)

GROUNDING_POLICIES: dict[str, GroundingPolicy] = {
    STRICT.key: STRICT,
    HYBRID.key: HYBRID,
}
DEFAULT_GROUNDING = STRICT.key


@dataclass(frozen=True)
class Settings:
    api_key: str
    chat_model: str = CHAT_MODEL
    embedding_model: str = EMBEDDING_MODEL
    data_dir: Path = field(default=DATA_DIR)


def _from_streamlit_secrets(name: str) -> str | None:
    try:
        value = st.secrets.get(name)  # type: ignore[union-attr]
    except Exception:
        return None
    return str(value) if value else None


def resolve_api_key() -> str:
    for getter in (
        lambda: _from_streamlit_secrets("GOOGLE_API_KEY"),
        lambda: _from_streamlit_secrets("GEMINI_API_KEY"),
        lambda: os.environ.get("GOOGLE_API_KEY"),
        lambda: os.environ.get("GEMINI_API_KEY"),
    ):
        value = getter()
        if value and value.strip():
            return value.strip()

    raise ConfigError(
        "No Gemini API key found, so the assistant cannot embed or answer.",
        hint=(
            "Local: create a `.env` file with `GOOGLE_API_KEY=...`, or export it "
            "in your shell.\n"
            "Streamlit Cloud: open **Manage app → Settings → Secrets** and add:\n"
            '```toml\nGOOGLE_API_KEY = "your-key-here"\n```\n'
            "Get a key at https://aistudio.google.com/apikey"
        ),
    )


def get_settings() -> Settings:
    return Settings(api_key=resolve_api_key())


def ensure_directories() -> None:
    for path in (DATA_DIR, CHROMA_DIR, CACHE_DIR, MANIFEST_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Logging
# ===========================================================================

_NOISY = (
    "pypdf",
    "pypdf._reader",
    "pypdf.generic",
    "chromadb",
    "chromadb.telemetry",
    "chromadb.segment",
    "httpx",
    "httpcore",
    "google_genai",
    "google.auth",
    "urllib3",
)

_configured = False


def configure_logging(level: int = logging.WARNING) -> None:
    global _configured
    if _configured:
        return

    logging.getLogger("arzens").setLevel(logging.INFO)
    for name in _NOISY:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    logging.getLogger().setLevel(level)
    _configured = True


# ===========================================================================
# Sources (data model)
# ===========================================================================

@dataclass(frozen=True)
class Section:
    label: str
    text: str
    ordinal: int = 0


@dataclass
class Source:
    kind: str  # "web" | "pdf"
    title: str
    locator: str
    sections: list[Section] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)

    @property
    def display_name(self) -> str:
        return self.title or self.locator


def content_fingerprint(payload: bytes | str) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()[:32]


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def split_markdown_sections(markdown: str, fallback_label: str) -> list[Section]:
    text = normalise_text(markdown)
    if not text:
        return []

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [Section(label=fallback_label, text=text, ordinal=0)]

    sections: list[Section] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(label=fallback_label, text=preamble, ordinal=0))

    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        sections.append(
            Section(
                label=heading[:120] or fallback_label,
                text=f"{heading}\n\n{body}",
                ordinal=len(sections),
            )
        )

    return sections or [Section(label=fallback_label, text=text, ordinal=0)]


def truncate_sections(sections: list[Section], max_chars: int) -> tuple[list[Section], bool]:
    kept: list[Section] = []
    running = 0
    for section in sections:
        if running + len(section.text) > max_chars:
            return kept or sections[:1], True
        kept.append(section)
        running += len(section.text)
    return kept, False


# ===========================================================================
# Chunking
# ===========================================================================

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def build_splitter(
    chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


def sanitise_metadata(metadata: dict) -> dict:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def format_citation(title: str, section_label: str) -> str:
    title = (title or "source").strip()
    section_label = (section_label or "").strip()
    if not section_label or section_label == title:
        return title
    return f"{title} — {section_label}"


def chunk_source(source: Source, splitter: RecursiveCharacterTextSplitter | None = None) -> list[Document]:
    splitter = splitter or build_splitter()
    documents: list[Document] = []

    for section in source.sections:
        pieces = [p for p in splitter.split_text(section.text) if p.strip()]
        for position, piece in enumerate(pieces):
            metadata = sanitise_metadata(
                {
                    "source_id": source.fingerprint,
                    "kind": source.kind,
                    "title": source.display_name,
                    "locator": source.locator,
                    "section": section.label,
                    "section_ordinal": section.ordinal,
                    "chunk_index": position,
                    "chunk_total": len(pieces),
                    "chunk_id": content_fingerprint(
                        f"{source.fingerprint}|{section.ordinal}|{position}|{piece}"
                    ),
                    "citation": format_citation(source.display_name, section.label),
                }
            )
            documents.append(Document(page_content=piece, metadata=metadata))

    return documents


def chunk_ids(documents: list[Document]) -> list[str]:
    return [str(doc.metadata["chunk_id"]) for doc in documents]


# ===========================================================================
# Embeddings
# ===========================================================================

_BATCH_SIZE = 32
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 1.5

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


class VectorDiskCache:
    def __init__(self, root: Path = CACHE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(text: str, *, model: str, dimensions: int, task_type: str) -> str:
        digest = hashlib.sha256(
            f"{model}|{dimensions}|{task_type}|{text}".encode("utf-8")
        ).hexdigest()
        return digest

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.vec"

    def get(self, key: str) -> list[float] | None:
        path = self._path(key)
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, OSError):
            self.misses += 1
            return None
        if not raw or len(raw) % 4 != 0:
            self.misses += 1
            return None
        self.hits += 1
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))

    def put(self, key: str, vector: list[float]) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(struct.pack(f"<{len(vector)}f", *vector))
            tmp.replace(path)
        except OSError:
            pass

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class CachedGeminiEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        *,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        cache: VectorDiskCache | None = None,
    ) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.model = model
        self.dimensions = dimensions
        self.cache = cache or VectorDiskCache()

        common = {
            "model": model,
            "api_key": api_key,
            "output_dimensionality": dimensions,
        }
        self._doc_client = GoogleGenerativeAIEmbeddings(task_type=DOCUMENT_TASK, **common)
        self._query_client = GoogleGenerativeAIEmbeddings(task_type=QUERY_TASK, **common)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts, task_type=DOCUMENT_TASK)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_many([text], task_type=QUERY_TASK)[0]

    def _client_for(self, task_type: str):
        return self._doc_client if task_type == DOCUMENT_TASK else self._query_client

    def _embed_many(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not texts:
            return []

        keys = [
            VectorDiskCache.make_key(
                text, model=self.model, dimensions=self.dimensions, task_type=task_type
            )
            for text in texts
        ]

        results: list[list[float] | None] = [self.cache.get(key) for key in keys]
        pending = [i for i, vector in enumerate(results) if vector is None]

        for start in range(0, len(pending), _BATCH_SIZE):
            batch_indices = pending[start : start + _BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_indices]
            vectors = self._call_api(batch_texts, task_type=task_type)
            for index, vector in zip(batch_indices, vectors):
                normalised = _normalise(vector)
                results[index] = normalised
                self.cache.put(keys[index], normalised)

        missing = [i for i, vector in enumerate(results) if vector is None]
        if missing:
            raise EmbeddingError(
                f"{len(missing)} passage(s) could not be embedded.",
                hint="Try ingesting again — partial progress was cached, so the "
                "retry will only redo what failed.",
            )
        return [vector for vector in results if vector is not None]

    def _call_api(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        client = self._client_for(task_type)
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                if task_type == QUERY_TASK and len(texts) == 1:
                    return [client.embed_query(texts[0])]
                return client.embed_documents(texts)
            except Exception as exc:
                last_error = exc
                if not _is_transient(exc) or attempt == _MAX_RETRIES - 1:
                    break
                time.sleep(_RETRY_BASE_SECONDS * (2**attempt))

        raise EmbeddingError(
            _embedding_error_message(last_error),
            hint=_embedding_error_hint(last_error),
            cause=last_error,
        )


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429", "rate limit", "resource_exhausted", "quota",
            "500", "502", "503", "504", "unavailable", "internal",
            "deadline", "timeout", "connection",
        )
    )


def _embedding_error_message(exc: Exception | None) -> str:
    text = str(exc or "").lower()
    if "api key" in text or "api_key" in text or "401" in text or "permission" in text:
        return "The Gemini API key was rejected while embedding."
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return "The Gemini embedding quota was exhausted."
    if "not found" in text or "404" in text:
        return "The embedding model is unavailable for this API key."
    return "The documents could not be embedded."


def _embedding_error_hint(exc: Exception | None) -> str:
    text = str(exc or "").lower()
    if "api key" in text or "401" in text or "permission" in text:
        return "Check GOOGLE_API_KEY in your Streamlit secrets or environment."
    if "429" in text or "quota" in text:
        return (
            "Wait a minute and try again, or ingest fewer documents at once. "
            "Everything embedded so far is cached and will not be re-charged."
        )
    return "Try again in a moment. Progress so far has been cached."


# ===========================================================================
# PDF ingestion
# ===========================================================================

def _open_pdf_reader(data: bytes, filename: str) -> PdfReader:
    configure_logging()
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise PDFParseError(
            f"`{filename}` is not a readable PDF.",
            hint="The file may be corrupt or only partially downloaded. "
            "Try re-exporting or re-downloading it.",
            cause=exc,
        ) from exc
    except Exception as exc:
        raise PDFParseError(
            f"`{filename}` could not be opened.", cause=exc
        ) from exc

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise PDFParseError(
                    f"`{filename}` is password-protected.",
                    hint="Remove the password and upload it again.",
                )
        except PDFParseError:
            raise
        except Exception as exc:
            raise PDFParseError(
                f"`{filename}` is encrypted and could not be decrypted.",
                hint="Remove the password and upload it again.",
                cause=exc,
            ) from exc

    return reader


def ingest_pdf(data: bytes, filename: str) -> Source:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        raise PDFParseError(
            f"`{filename}` is {size_mb:.1f} MB, over the {MAX_PDF_MB} MB limit.",
            hint="Split the document, or upload only the chapters you need.",
        )
    if not data:
        raise PDFParseError(f"`{filename}` is empty.")

    reader = _open_pdf_reader(data, filename)

    if len(reader.pages) == 0:
        raise EmptyContentError(f"`{filename}` has no pages.")

    sections: list[Section] = []
    empty_pages = 0

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            empty_pages += 1
            continue

        text = normalise_text(raw)
        if not text:
            empty_pages += 1
            continue

        sections.append(Section(label=f"page {index}", text=text, ordinal=index))

    if not sections:
        raise EmptyContentError(
            f"`{filename}` contains no extractable text.",
            hint=(
                "This is almost certainly a scanned document — the pages are "
                "images, not text. Run it through an OCR tool first, then "
                "upload the searchable version."
            ),
        )

    sections, truncated = truncate_sections(sections, MAX_CHARS_PER_SOURCE)
    if truncated:
        sections.append(
            Section(
                label="truncated",
                text=f"[This document exceeded the {MAX_CHARS_PER_SOURCE:,}-character "
                f"per-source limit and was truncated.]",
                ordinal=len(sections) + 1,
            )
        )

    return Source(
        kind="pdf",
        title=filename,
        locator=filename,
        sections=sections,
        fingerprint=content_fingerprint(data),
    )


# ===========================================================================
# Web ingestion
# ===========================================================================

JINA_READER_PREFIX = "https://r.jina.ai/"
_USER_AGENT = "Mozilla/5.0 (compatible; ArzensKnowledgeAssistant/1.0)"

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg|template)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_CHROME_RE = re.compile(
    r"<(nav|header|footer|aside|form)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_BLOCK_END_RE = re.compile(
    r"</(p|div|section|article|li|tr|h[1-6]|blockquote|pre)\s*>", re.IGNORECASE
)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)


def normalise_url(raw: str) -> str:
    url = raw.strip().strip("<>\"'")
    if not url:
        raise WebFetchError("An empty URL was provided.")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    if not parsed.netloc or "." not in parsed.netloc:
        raise WebFetchError(
            f"`{raw}` does not look like a valid web address.",
            hint="Use a full address, for example `https://example.com/about`.",
        )
    return url


def _unescape_entities(text: str) -> str:
    import html as html_module

    return html_module.unescape(text)


def strip_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(
            ["script", "style", "noscript", "svg", "nav", "header", "footer",
             "aside", "form", "template"]
        ):
            tag.decompose()
        return normalise_text(soup.get_text(separator="\n"))
    except ImportError:
        pass

    text = _COMMENT_RE.sub(" ", html)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _CHROME_RE.sub(" ", text)
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_END_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = _unescape_entities(text)
    return normalise_text(text)


def extract_html_title(html: str, fallback: str) -> str:
    for pattern in (_TITLE_RE, _H1_RE):
        match = pattern.search(html)
        if match:
            title = normalise_text(_unescape_entities(_TAG_RE.sub(" ", match.group(1))))
            if title:
                return title[:160]
    return fallback


def _fetch(url: str, *, headers: dict[str, str], label: str) -> requests.Response:
    try:
        response = requests.get(url, headers=headers, timeout=WEB_TIMEOUT_SECONDS)
    except requests.Timeout as exc:
        raise WebFetchError(
            f"{label} timed out after {WEB_TIMEOUT_SECONDS}s.",
            hint="The site may be slow or blocking automated readers. Try again, "
            "or upload the content as a PDF instead.",
            cause=exc,
        ) from exc
    except requests.RequestException as exc:
        raise WebFetchError(
            f"{label} could not be reached.",
            hint="Check the address and your connection. Some sites block "
            "automated access entirely.",
            cause=exc,
        ) from exc

    if response.status_code >= 400:
        raise WebFetchError(
            f"{label} returned HTTP {response.status_code}.",
            hint="The page may be private, removed, or protected against scraping.",
        )
    return response


def fetch_via_jina(url: str) -> str:
    response = _fetch(
        JINA_READER_PREFIX + url,
        headers={"Accept": "text/plain", "User-Agent": _USER_AGENT},
        label="The reader service",
    )
    text = normalise_text(response.text)
    if len(text) < 40:
        raise WebFetchError("The reader service returned an empty page.")
    return text


def fetch_direct(url: str) -> tuple[str, str]:
    response = _fetch(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
        },
        label="The website",
    )
    content_type = response.headers.get("content-type", "").lower()
    raw = response.text

    if "html" in content_type or raw.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        return strip_html(raw), extract_html_title(raw, url)
    return normalise_text(raw), url


def ingest_url(raw_url: str) -> Source:
    url = normalise_url(raw_url)
    host = urlparse(url).netloc

    title = url
    reader_failure: WebFetchError | None = None

    try:
        text = fetch_via_jina(url)
        first_line, _, remainder = text.partition("\n")
        if first_line.lower().startswith("title:"):
            title = first_line.split(":", 1)[1].strip() or url
            text = remainder.strip()
        else:
            title = host
    except WebFetchError as exc:
        reader_failure = exc
        text, title = fetch_direct(url)

    if not text or len(text) < 40:
        raise EmptyContentError(
            f"`{url}` contained no readable text.",
            hint=(
                "The page is probably rendered entirely by JavaScript, or it is "
                "a login wall. Save it as a PDF and upload that instead."
            ),
            cause=reader_failure,
        )

    sections = split_markdown_sections(text, fallback_label=host or "page")
    sections, truncated = truncate_sections(sections, MAX_CHARS_PER_SOURCE)
    if truncated:
        sections.append(
            Section(
                label="truncated",
                text=f"[This page exceeded the {MAX_CHARS_PER_SOURCE:,}-character "
                f"per-source limit and was truncated.]",
                ordinal=len(sections),
            )
        )

    return Source(
        kind="web",
        title=(title or host)[:160],
        locator=url,
        sections=sections,
        fingerprint=content_fingerprint(url + "\n" + text),
    )


# ===========================================================================
# Vector store & workspace management
# ===========================================================================

_WORKSPACE_RE = re.compile(r"^[a-f0-9]{16}$")
_client = None


def new_workspace_id() -> str:
    return secrets.token_hex(8)


def is_valid_workspace_id(value: str | None) -> bool:
    return bool(value) and bool(_WORKSPACE_RE.match(str(value)))


def collection_name(workspace_id: str) -> str:
    if not is_valid_workspace_id(workspace_id):
        raise VectorStoreError("Invalid workspace id.")
    return f"ws_{workspace_id}"


def manifest_path(workspace_id: str) -> Path:
    if not is_valid_workspace_id(workspace_id):
        raise VectorStoreError("Invalid workspace id.")
    return MANIFEST_DIR / f"{workspace_id}.json"


@dataclass
class SourceRecord:
    source_id: str
    kind: str
    title: str
    locator: str
    chunk_count: int
    char_count: int
    ingested_at: float

    @property
    def ingested_label(self) -> str:
        return time.strftime("%d %b %H:%M", time.localtime(self.ingested_at))


class Manifest:
    """What has been ingested into one workspace."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.path = manifest_path(workspace_id)
        self.records: dict[str, SourceRecord] = {}
        self.load()

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.records = {}
            return
        self.records = {
            key: SourceRecord(**value)
            for key, value in payload.get("sources", {}).items()
            if isinstance(value, dict)
        }

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "workspace_id": self.workspace_id,
                "updated_at": time.time(),
                "sources": {key: asdict(rec) for key, rec in self.records.items()},
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            raise VectorStoreError(
                "The knowledge base index could not be saved to disk.",
                hint="Check that the app has write access to its data directory.",
                cause=exc,
            ) from exc

    def has(self, source_id: str) -> bool:
        return source_id in self.records

    def add(self, record: SourceRecord) -> None:
        self.records[record.source_id] = record
        self.save()

    def remove(self, source_id: str) -> None:
        self.records.pop(source_id, None)
        self.save()

    def clear(self) -> None:
        self.records = {}
        self.save()

    def touch(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.save()
            else:
                self.path.touch()
        except OSError:
            pass

    @property
    def is_empty(self) -> bool:
        return not self.records

    def summary(self) -> tuple[int, int]:
        return len(self.records), sum(r.chunk_count for r in self.records.values())


def get_client():
    global _client
    if _client is not None:
        return _client

    ensure_directories()
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
    except Exception as exc:
        raise VectorStoreError(
            "The local vector database could not be opened.",
            hint="If this persists, delete the app's data directory and re-ingest "
            "your sources — the store may have been left in a bad state.",
            cause=exc,
        ) from exc
    return _client


def get_vectorstore(workspace_id: str, embeddings):
    from langchain_chroma import Chroma

    try:
        return Chroma(
            client=get_client(),
            collection_name=collection_name(workspace_id),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            create_collection_if_not_exists=True,
        )
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            "This session's knowledge base could not be opened.", cause=exc
        ) from exc


def add_documents(vectorstore, documents: list[Document]) -> int:
    if not documents:
        return 0
    try:
        vectorstore.add_documents(documents=documents, ids=chunk_ids(documents))
    except VectorStoreError:
        raise
    except Exception as exc:
        if isinstance(exc, EmbeddingError):
            raise
        raise VectorStoreError(
            "The passages could not be written to the knowledge base.", cause=exc
        ) from exc
    return len(documents)


def delete_source(vectorstore, source_id: str) -> None:
    try:
        vectorstore.delete(where={"source_id": source_id})
    except Exception as exc:
        raise VectorStoreError("That source could not be removed.", cause=exc) from exc


def drop_workspace(workspace_id: str) -> None:
    try:
        get_client().delete_collection(collection_name(workspace_id))
    except Exception:
        pass
    try:
        manifest_path(workspace_id).unlink(missing_ok=True)
    except OSError:
        pass


def retrieve(vectorstore, query: str, *, k: int = RETRIEVAL_K,
             floor: float = RELEVANCE_FLOOR) -> list[tuple[Document, float]]:
    if not query.strip():
        return []
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:
        raise VectorStoreError(
            "The knowledge base could not be searched.",
            hint="Try re-ingesting your sources.",
            cause=exc,
        ) from exc

    return [(doc, score) for doc, score in scored if score is None or score >= floor]


def sweep_stale_workspaces(ttl_hours: int = WORKSPACE_TTL_HOURS) -> int:
    ensure_directories()
    cutoff = time.time() - ttl_hours * 3600
    removed = 0

    for path in MANIFEST_DIR.glob("*.json"):
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            workspace_id = path.stem
            if not is_valid_workspace_id(workspace_id):
                continue
            drop_workspace(workspace_id)
            removed += 1
        except OSError:
            continue

    return removed


# ===========================================================================
# RAG: condense -> retrieve -> generate
# ===========================================================================

_MAX_CONDENSE_CHARS = 4000


@dataclass
class Turn:
    question: str
    answer: str
    citations: list[str] | None = None


@dataclass
class Passage:
    number: int
    document: Document
    score: float

    @property
    def citation(self) -> str:
        return str(self.document.metadata.get("citation", "source"))

    @property
    def locator(self) -> str:
        return str(self.document.metadata.get("locator", ""))

    @property
    def kind(self) -> str:
        return str(self.document.metadata.get("kind", ""))

    @property
    def text(self) -> str:
        return self.document.page_content


def build_llm(api_key: str, *, model: str = CHAT_MODEL, temperature: float = 0.2):
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=2048,
        )
    except Exception as exc:
        raise GenerationError(
            "The language model could not be initialised.",
            hint="Check that GOOGLE_API_KEY is valid and that the configured "
            f"model (`{model}`) is available to your account.",
            cause=exc,
        ) from exc


_CONDENSE_SYSTEM = (
    "Rewrite the user's latest message into a single standalone search query.\n"
    "Resolve every pronoun and implicit reference using the conversation.\n"
    "Preserve all specific nouns, names, and numbers — they carry the search signal.\n"
    "Do not answer the question. Do not explain. Output the rewritten query only.\n"
    "If the message is already standalone, output it unchanged."
)


def condense_question(llm, question: str, history: Sequence[Turn]) -> str:
    if not history or not question.strip():
        return question.strip()

    transcript_lines: list[str] = []
    for turn in list(history)[-HISTORY_TURNS:]:
        transcript_lines.append(f"User: {turn.question}")
        transcript_lines.append(f"Assistant: {turn.answer[:400]}")
    transcript = "\n".join(transcript_lines)[-_MAX_CONDENSE_CHARS:]

    messages = [
        SystemMessage(content=_CONDENSE_SYSTEM),
        HumanMessage(
            content=f"Conversation so far:\n{transcript}\n\n"
            f"Latest message: {question}\n\nStandalone query:"
        ),
    ]

    try:
        response = llm.invoke(messages)
        rewritten = (getattr(response, "content", "") or "").strip()
    except Exception:
        return question.strip()

    if not rewritten or len(rewritten) > 500:
        return question.strip()
    return rewritten


def to_passages(scored: Sequence[tuple[Document, float]]) -> list[Passage]:
    return [
        Passage(number=i, document=doc, score=float(score or 0.0))
        for i, (doc, score) in enumerate(scored, start=1)
    ]


def format_context(passages: Sequence[Passage]) -> str:
    if not passages:
        return "(no relevant passages were found in the knowledge base)"
    return "\n\n".join(
        f"[{p.number}] Source: {p.citation}\n{p.text}" for p in passages
    )


def describe_library(source_titles: Sequence[str], limit: int = 8) -> str:
    titles = [t for t in source_titles if t]
    if not titles:
        return "(the knowledge base is empty)"
    shown = ", ".join(titles[:limit])
    if len(titles) > limit:
        shown += f", and {len(titles) - limit} more"
    return shown


_BASE_SYSTEM = """You are {app_name}, a retrieval-grounded knowledge assistant.

You are given numbered CONTEXT passages retrieved from documents the user \
ingested, plus the conversation so far.

GROUNDING RULES
{grounding_instruction}

CITATION RULES
- Cite with the bracketed passage number immediately after the claim it \
supports, like this: "The deadline is 14 March [2]."
- Every factual sentence drawn from CONTEXT carries at least one citation.
- Cite only numbers that appear in CONTEXT. Never invent a citation number.
- Do not append a bibliography or "Sources:" list — the interface renders \
the sources itself.

SCOPE RULES — answer what was asked, and only what was asked
- Answer the precise question. Do not volunteer adjacent facts, background, \
or related topics the user did not ask about.
- No preamble ("Great question", "Based on the context provided"). Open with \
the answer itself.
- No unsolicited summaries, caveats, next steps, or follow-up suggestions.
- Match the length to the question: a yes/no question gets a sentence, not a \
section. Use bullets only for genuinely enumerable answers.
- If the question has several distinct parts, answer each part, and nothing more.

The knowledge base currently contains: {library}"""


def build_system_prompt(
    *, app_name: str, grounding_key: str, library: str
) -> str:
    policy = GROUNDING_POLICIES.get(grounding_key) or GROUNDING_POLICIES[DEFAULT_GROUNDING]
    return _BASE_SYSTEM.format(
        app_name=app_name,
        grounding_instruction=policy.instruction,
        library=library,
    )


def build_messages(
    *,
    system_prompt: str,
    history: Sequence[Turn],
    question: str,
    passages: Sequence[Passage],
) -> list:
    messages: list = [SystemMessage(content=system_prompt)]

    for turn in list(history)[-HISTORY_TURNS:]:
        messages.append(HumanMessage(content=turn.question))
        messages.append(AIMessage(content=turn.answer))

    messages.append(
        HumanMessage(
            content=f"CONTEXT:\n{format_context(passages)}\n\n"
            f"QUESTION: {question}"
        )
    )
    return messages


def stream_answer(llm, messages: list) -> Iterator[str]:
    started = False
    try:
        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", None)
            if text is None:
                continue
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in text
                )
            if text:
                started = True
                yield text
    except Exception as exc:
        if started:
            yield f"\n\n---\n*The answer was cut short: {_generation_reason(exc)}*"
            return
        raise GenerationError(
            _generation_message(exc),
            hint=_generation_hint(exc),
            cause=exc,
        ) from exc

    if not started:
        yield (
            "The model returned an empty response. This usually means a safety "
            "filter blocked the output. Try rephrasing the question."
        )


def _generation_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "quota" in text:
        return "the API rate limit was hit"
    if "timeout" in text or "deadline" in text:
        return "the request timed out"
    return "the connection to the model dropped"


def _generation_message(exc: Exception) -> str:
    text = str(exc).lower()
    if "api key" in text or "401" in text or "permission" in text:
        return "The Gemini API key was rejected."
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return "The Gemini rate limit or quota was reached."
    if "404" in text or "not found" in text:
        return "The configured model is not available to this API key."
    if "safety" in text or "blocked" in text:
        return "The response was blocked by a safety filter."
    return "The assistant could not generate an answer."


def _generation_hint(exc: Exception) -> str:
    text = str(exc).lower()
    if "api key" in text or "401" in text:
        return "Check GOOGLE_API_KEY in your Streamlit secrets or environment."
    if "429" in text or "quota" in text:
        return "Wait a moment and ask again — free-tier quotas reset quickly."
    if "404" in text:
        return f"Confirm `{CHAT_MODEL}` is enabled for your project in Google AI Studio."
    return "Try asking again. If it keeps failing, check your network connection."


# ===========================================================================
# Transcript export
# ===========================================================================

def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def to_markdown(
    turns: Sequence[Turn],
    *,
    app_name: str,
    sources: Sequence[str] = (),
    grounding_label: str = "",
) -> str:
    lines = [
        f"# {app_name} — conversation transcript",
        "",
        f"*Exported {_timestamp()}*",
    ]
    if grounding_label:
        lines.append(f"*Grounding mode: {grounding_label}*")
    lines.append("")

    if sources:
        lines.append("## Knowledge base")
        lines.append("")
        lines.extend(f"- {name}" for name in sources)
        lines.append("")

    lines.append("## Conversation")
    lines.append("")

    if not turns:
        lines.append("*No messages yet.*")

    for index, turn in enumerate(turns, start=1):
        lines.append(f"### {index}. {turn.question}")
        lines.append("")
        lines.append(turn.answer or "*(no answer)*")
        lines.append("")
        if turn.citations:
            lines.append("**Sources cited**")
            lines.append("")
            lines.extend(f"- {c}" for c in turn.citations)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_json(
    turns: Sequence[Turn],
    *,
    app_name: str,
    sources: Sequence[str] = (),
    grounding_label: str = "",
) -> str:
    payload = {
        "app": app_name,
        "exported_at": _timestamp(),
        "grounding_mode": grounding_label,
        "knowledge_base": list(sources),
        "turns": [
            {
                "index": i,
                "question": t.question,
                "answer": t.answer,
                "citations": t.citations or [],
            }
            for i, t in enumerate(turns, start=1)
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def transcript_filename(extension: str) -> str:
    return f"arzens-transcript-{time.strftime('%Y%m%d-%H%M%S')}.{extension}"


# ===========================================================================
# Streamlit UI
# ===========================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_brand_css() -> None:
    """Ink-and-seal theme: deep navy + gold accent, serif display type.

    Targets Streamlit's data-testid hooks (stable across releases) rather
    than generated class names, so this survives Streamlit version bumps.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --navy: {BRAND['navy']};
            --navy-light: {BRAND['navy_light']};
            --ink: {BRAND['ink']};
            --paper: {BRAND['paper']};
            --paper-dim: {BRAND['paper_dim']};
            --gold: {BRAND['gold']};
            --gold-light: {BRAND['gold_light']};
            --teal: {BRAND['teal']};
            --rose: {BRAND['rose']};
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            background: var(--paper);
            color: var(--ink);
            font-family: 'Inter', sans-serif;
            color-scheme: light;
        }}

        [data-testid="stAppViewContainer"] > .main {{
            background: var(--paper);
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Fraunces', serif;
            font-weight: 600;
            color: var(--navy);
            letter-spacing: -0.01em;
        }}

        h1 {{
            border-bottom: 2px solid var(--gold);
            padding-bottom: 0.5rem;
            display: inline-block;
        }}

        p, li, span, label {{
            font-family: 'Inter', sans-serif;
        }}

        code, pre, .stCode {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* Sidebar: deep navy, reads as the "ledger spine" of the app */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--navy) 0%, var(--navy-light) 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #EDEAE1 !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: var(--gold-light) !important;
            font-family: 'Fraunces', serif;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(230, 220, 190, 0.18);
        }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: rgba(237, 234, 225, 0.65) !important;
        }}

        /* Sidebar form fields (URL box, file uploader text, etc.) — these
           sit on the dark navy sidebar but must render as a light "paper"
           field with dark ink text, or the text disappears entirely. */
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] input[type="text"],
        [data-testid="stSidebar"] input[type="password"],
        [data-testid="stSidebar"] input[type="number"] {{
            background: var(--paper) !important;
            color: var(--ink) !important;
            caret-color: var(--ink) !important;
            border: 1px solid rgba(230, 190, 100, 0.45) !important;
            border-radius: 6px !important;
        }}
        [data-testid="stSidebar"] textarea::placeholder,
        [data-testid="stSidebar"] input::placeholder {{
            color: rgba(26, 26, 26, 0.45) !important;
        }}
        [data-testid="stSidebar"] textarea:focus,
        [data-testid="stSidebar"] input:focus {{
            border-color: var(--gold) !important;
            box-shadow: 0 0 0 1px var(--gold) !important;
        }}
        /* File uploader dropzone: keep it light with dark ink text too */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: var(--paper) !important;
            border: 1px dashed rgba(230, 190, 100, 0.45) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
            color: var(--ink) !important;
        }}

        /* Primary buttons: gold seal */
        .stButton > button[kind="primary"],
        button[data-testid="stFormSubmitButton"] {{
            background: var(--gold) !important;
            color: var(--navy) !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            transition: background 0.15s ease;
        }}
        .stButton > button[kind="primary"]:hover,
        button[data-testid="stFormSubmitButton"]:hover {{
            background: var(--gold-light) !important;
        }}

        /* Secondary / sidebar buttons: outlined gold on navy */
        [data-testid="stSidebar"] .stButton > button {{
            background: transparent !important;
            border: 1px solid rgba(230, 190, 100, 0.45) !important;
            color: #EDEAE1 !important;
            border-radius: 6px !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            border-color: var(--gold-light) !important;
            color: var(--gold-light) !important;
        }}

        /* Main-area buttons (starter questions) */
        .main .stButton > button {{
            background: white !important;
            border: 1px solid var(--paper-dim) !important;
            color: var(--navy) !important;
            border-radius: 8px !important;
            text-align: left !important;
        }}
        .main .stButton > button:hover {{
            border-color: var(--gold) !important;
            color: var(--navy) !important;
        }}

        /* Chat bubbles — force ink text on white regardless of the
           browser/OS theme, so answers never render white-on-white. */
        [data-testid="stChatMessage"] {{
            background: white !important;
            border-radius: 10px;
            border: 1px solid var(--paper-dim);
            padding: 0.6rem 0.9rem;
        }}
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] strong,
        [data-testid="stChatMessage"] em,
        [data-testid="stChatMessage"] h1,
        [data-testid="stChatMessage"] h2,
        [data-testid="stChatMessage"] h3,
        [data-testid="stChatMessage"] h4 {{
            color: var(--ink) !important;
        }}
        [data-testid="stChatMessage"] code {{
            color: var(--navy) !important;
            background: var(--paper-dim) !important;
        }}
        [data-testid="stChatMessage"] a {{
            color: var(--teal) !important;
        }}

        /* Chat input — the main search/ask bar. Explicit light field with
           dark text so it never inherits an invisible dark-on-dark or
           light-on-light combo from the ambient theme. */
        [data-testid="stChatInput"] {{
            border-color: var(--navy) !important;
            background: white !important;
        }}
        [data-testid="stChatInput"] textarea {{
            background: white !important;
            color: var(--ink) !important;
            caret-color: var(--ink) !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: rgba(26, 26, 26, 0.45) !important;
        }}

        /* Radio (grounding mode) */
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            color: #EDEAE1 !important;
        }}

        /* Alerts: recolor to match brand instead of default streamlit red/blue */
        [data-testid="stAlertContentInfo"] {{
            background: rgba(198, 149, 44, 0.10) !important;
            color: var(--ink) !important;
        }}
        [data-testid="stAlertContentSuccess"] {{
            background: rgba(47, 111, 98, 0.10) !important;
            color: var(--ink) !important;
        }}
        [data-testid="stAlertContentError"] {{
            background: rgba(176, 70, 58, 0.08) !important;
            color: var(--ink) !important;
        }}
        [data-testid="stAlertContentInfo"] *,
        [data-testid="stAlertContentSuccess"] *,
        [data-testid="stAlertContentError"] * {{
            color: var(--ink) !important;
        }}

        /* Expanders (citations) get a gold left border — the "seal" motif */
        [data-testid="stExpander"] {{
            border-left: 3px solid var(--gold) !important;
            border-radius: 6px;
        }}

        /* Progress bar */
        [data-testid="stProgressBar"] > div > div {{
            background-color: var(--gold) !important;
        }}

        /* Download buttons */
        [data-testid="stDownloadButton"] > button {{
            border-color: var(--navy) !important;
            color: var(--navy) !important;
            border-radius: 6px !important;
        }}
        [data-testid="stDownloadButton"] > button:hover {{
            background: var(--navy) !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str) -> CachedGeminiEmbeddings:
    return CachedGeminiEmbeddings(api_key=api_key, cache=VectorDiskCache())


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return build_llm(api_key)


@st.cache_resource(show_spinner=False)
def run_startup_housekeeping() -> int:
    configure_logging()
    ensure_directories()
    try:
        return sweep_stale_workspaces()
    except Exception:
        return 0


def resolve_workspace_id() -> str:
    existing = st.query_params.get("ws")
    if is_valid_workspace_id(existing):
        return str(existing)

    workspace_id = new_workspace_id()
    st.query_params["ws"] = workspace_id
    return workspace_id


def init_state() -> None:
    defaults = {
        "turns": [],
        "pending_question": None,
        "grounding": DEFAULT_GROUNDING,
        "last_ingest_report": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def ingest_sources(
    *,
    urls: list[str],
    pdfs: list,
    vectorstore,
    manifest: Manifest,
    force: bool,
) -> dict:
    report = {"added": [], "skipped": [], "failed": []}
    splitter = build_splitter()

    total = len(urls) + len(pdfs)
    if total == 0:
        return report

    progress = st.progress(0.0, text="Starting…")
    done = 0

    def advance(label: str) -> None:
        nonlocal done
        done += 1
        progress.progress(min(done / total, 1.0), text=label)

    for raw_url in urls:
        try:
            progress.progress(done / total, text=f"Reading {raw_url}…")
            source = ingest_url(raw_url)
            _store_source(source, vectorstore, manifest, splitter, force, report)
        except ArzensError as exc:
            report["failed"].append((raw_url, exc))
        except Exception as exc:
            report["failed"].append(
                (raw_url, ArzensError(f"`{raw_url}` failed unexpectedly.", cause=exc))
            )
        advance(f"Processed {raw_url}")

    for upload in pdfs:
        name = getattr(upload, "name", "document.pdf")
        try:
            progress.progress(done / total, text=f"Reading {name}…")
            source = ingest_pdf(upload.getvalue(), name)
            _store_source(source, vectorstore, manifest, splitter, force, report)
        except ArzensError as exc:
            report["failed"].append((name, exc))
        except Exception as exc:
            report["failed"].append(
                (name, ArzensError(f"`{name}` failed unexpectedly.", cause=exc))
            )
        advance(f"Processed {name}")

    progress.empty()
    return report


def _store_source(
    source: Source,
    vectorstore,
    manifest: Manifest,
    splitter,
    force: bool,
    report: dict,
) -> None:
    if not force and manifest.has(source.fingerprint):
        report["skipped"].append(source.display_name)
        return

    documents = chunk_source(source, splitter)
    if not documents:
        report["failed"].append(
            (source.display_name, ArzensError("No text could be extracted."))
        )
        return

    add_documents(vectorstore, documents)
    manifest.add(
        SourceRecord(
            source_id=source.fingerprint,
            kind=source.kind,
            title=source.display_name,
            locator=source.locator,
            chunk_count=len(documents),
            char_count=source.char_count,
            ingested_at=time.time(),
        )
    )
    report["added"].append((source.display_name, len(documents)))


def render_ingest_report(report: dict) -> None:
    for name, count in report["added"]:
        st.success(f"**{name}** — {count} passages indexed.")
    if report["skipped"]:
        st.info(
            "Already indexed, skipped (no re-embedding): "
            + ", ".join(f"**{n}**" for n in report["skipped"])
        )
    for name, error in report["failed"]:
        message = error.display() if isinstance(error, ArzensError) else str(error)
        st.error(f"**{name}** — {message}")
        if isinstance(error, ArzensError) and error.cause is not None:
            with st.expander(f"Technical detail for {name}"):
                st.code(error.technical_detail())


def render_sidebar(vectorstore, manifest: Manifest, workspace_id: str) -> None:
    with st.sidebar:
        st.header("Knowledge base")

        source_count, chunk_count = manifest.summary()
        if source_count:
            st.caption(f"{source_count} source(s) · {chunk_count} passages indexed")
        else:
            st.caption("Empty — add a website or a PDF to begin.")

        with st.form("ingest_form", clear_on_submit=False):
            urls_raw = st.text_area(
                "Website URLs",
                placeholder="https://example.com/docs\nhttps://example.com/pricing",
                help=f"One per line. Up to {MAX_URLS_PER_INGEST} at a time.",
                height=90,
            )
            pdfs = st.file_uploader(
                "PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                help=f"Up to {MAX_PDFS_PER_UPLOAD} files, "
                f"{MAX_PDF_MB} MB each. Text-based PDFs only — "
                "scans need OCR first.",
            )
            force = st.checkbox(
                "Force re-index",
                value=False,
                help="Re-embed even if the content is unchanged. Normally "
                "unnecessary — unchanged sources are skipped automatically.",
            )
            submitted = st.form_submit_button(
                "Build knowledge base", type="primary", use_container_width=True
            )

        if submitted:
            urls = [u.strip() for u in (urls_raw or "").splitlines() if u.strip()]
            pdfs = pdfs or []

            if not urls and not pdfs:
                st.warning("Add at least one URL or PDF first.")
            elif len(urls) > MAX_URLS_PER_INGEST:
                st.warning(
                    f"That is {len(urls)} URLs. The limit is "
                    f"{MAX_URLS_PER_INGEST} per batch."
                )
            elif len(pdfs) > MAX_PDFS_PER_UPLOAD:
                st.warning(
                    f"That is {len(pdfs)} files. The limit is "
                    f"{MAX_PDFS_PER_UPLOAD} per batch."
                )
            else:
                report = ingest_sources(
                    urls=urls,
                    pdfs=pdfs,
                    vectorstore=vectorstore,
                    manifest=manifest,
                    force=force,
                )
                st.session_state.last_ingest_report = report
                render_ingest_report(report)

        if manifest.records:
            st.divider()
            st.subheader("Indexed sources")
            for record in sorted(
                manifest.records.values(), key=lambda r: r.ingested_at, reverse=True
            ):
                icon = "🌐" if record.kind == "web" else "📄"
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(f"{icon} **{record.title}**")
                    st.caption(
                        f"{record.chunk_count} passages · added {record.ingested_label}"
                    )
                with right:
                    if st.button(
                        "✕",
                        key=f"del_{record.source_id}",
                        help="Remove this source",
                        use_container_width=True,
                    ):
                        try:
                            delete_source(vectorstore, record.source_id)
                            manifest.remove(record.source_id)
                            st.rerun()
                        except ArzensError as exc:
                            st.error(exc.display())

        st.divider()
        st.subheader("Answering")

        keys = list(GROUNDING_POLICIES)
        current = st.session_state.grounding
        chosen = st.radio(
            "Grounding",
            options=keys,
            index=keys.index(current) if current in keys else 0,
            format_func=lambda k: GROUNDING_POLICIES[k].label,
            label_visibility="collapsed",
        )
        st.session_state.grounding = chosen
        st.caption(GROUNDING_POLICIES[chosen].description)

        st.divider()
        st.subheader("Transcript")
        turns: list[Turn] = st.session_state.turns
        source_names = [r.title for r in manifest.records.values()]
        label = GROUNDING_POLICIES[st.session_state.grounding].label

        md_col, json_col = st.columns(2)
        with md_col:
            st.download_button(
                "Markdown",
                data=to_markdown(
                    turns,
                    app_name=APP_TITLE,
                    sources=source_names,
                    grounding_label=label,
                ),
                file_name=transcript_filename("md"),
                mime="text/markdown",
                disabled=not turns,
                use_container_width=True,
            )
        with json_col:
            st.download_button(
                "JSON",
                data=to_json(
                    turns,
                    app_name=APP_TITLE,
                    sources=source_names,
                    grounding_label=label,
                ),
                file_name=transcript_filename("json"),
                mime="application/json",
                disabled=not turns,
                use_container_width=True,
            )

        st.divider()
        with st.expander("Session"):
            st.caption(
                "Your documents live in a private workspace tied to this URL. "
                "Bookmark the page to come back to the same knowledge base; "
                "unused workspaces are cleared after "
                f"{WORKSPACE_TTL_HOURS} hours."
            )
            st.code(workspace_id, language=None)
            if st.button("Clear chat", use_container_width=True):
                st.session_state.turns = []
                st.rerun()
            if st.button(
                "Delete everything in this workspace",
                use_container_width=True,
                type="secondary",
            ):
                drop_workspace(workspace_id)
                st.session_state.turns = []
                st.query_params.clear()
                st.rerun()


def render_history() -> None:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            st.markdown(turn.answer)
            if turn.citations:
                with st.expander(f"Sources ({len(turn.citations)})"):
                    for citation in turn.citations:
                        st.markdown(f"- {citation}")


def render_welcome(has_sources: bool) -> None:
    if has_sources:
        st.markdown("#### Try one of these")
        columns = st.columns(2)
        for index, question in enumerate(STARTER_QUESTIONS):
            with columns[index % 2]:
                if st.button(question, key=f"starter_{index}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()
        return

    st.info(
        "**Start by building a knowledge base.** Paste one or more website "
        "URLs in the sidebar, upload PDFs, or both — then ask questions and "
        "get answers with the exact passages they came from.",
        icon="👈",
    )


def collect_citations(answer: str, passages) -> list[str]:
    cited_numbers = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}
    seen: list[str] = []
    for passage in passages:
        if passage.number in cited_numbers and passage.citation not in seen:
            seen.append(passage.citation)
    return seen


def answer_question(question: str, vectorstore, manifest: Manifest, llm) -> None:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        passages = []
        try:
            with st.spinner("Searching your sources…"):
                search_query = condense_question(llm, question, st.session_state.turns)
                scored = retrieve(vectorstore, search_query)
                passages = to_passages(scored)
        except ArzensError as exc:
            st.error(exc.display())
            return

        system_prompt = build_system_prompt(
            app_name=APP_TITLE,
            grounding_key=st.session_state.grounding,
            library=describe_library([r.title for r in manifest.records.values()]),
        )
        messages = build_messages(
            system_prompt=system_prompt,
            history=st.session_state.turns,
            question=question,
            passages=passages,
        )

        try:
            answer = st.write_stream(stream_answer(llm, messages))
        except ArzensError as exc:
            st.error(exc.display())
            if exc.cause is not None:
                with st.expander("Technical detail"):
                    st.code(exc.technical_detail())
            return
        except Exception as exc:
            st.error(
                "The assistant hit an unexpected problem while answering. "
                "Please try again."
            )
            with st.expander("Technical detail"):
                st.code(f"{type(exc).__name__}: {exc}")
            return

        answer = answer if isinstance(answer, str) else str(answer)
        citations = collect_citations(answer, passages)

        if citations:
            with st.expander(f"Sources ({len(citations)})", expanded=False):
                for passage in passages:
                    if passage.citation not in citations:
                        continue
                    st.markdown(f"**[{passage.number}] {passage.citation}**")
                    if passage.kind == "web" and passage.locator.startswith("http"):
                        st.caption(passage.locator)
                    st.caption(f"relevance {passage.score:.2f}")
                    st.markdown(
                        f"> {passage.text[:700]}"
                        + ("…" if len(passage.text) > 700 else "")
                    )
                    st.divider()

        st.session_state.turns.append(
            Turn(question=question, answer=answer, citations=citations)
        )


def main() -> None:
    inject_brand_css()
    init_state()
    run_startup_housekeeping()

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(
        "Ask questions against your own websites and PDFs. Every answer is "
        "traced back to the passages it came from."
    )

    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(exc.display())
        st.stop()
        return

    workspace_id = resolve_workspace_id()
    manifest = Manifest(workspace_id)
    manifest.touch()

    try:
        embeddings = get_embeddings(settings.api_key)
        vectorstore = get_vectorstore(workspace_id, embeddings)
        llm = get_llm(settings.api_key)
    except ArzensError as exc:
        st.error(exc.display())
        if exc.cause is not None:
            with st.expander("Technical detail"):
                st.code(exc.technical_detail())
        st.stop()
        return

    render_sidebar(vectorstore, manifest, workspace_id)

    render_history()

    if not st.session_state.turns:
        render_welcome(has_sources=bool(manifest.records))

    question = st.session_state.pending_question or st.chat_input(
        "Ask a question about your sources…"
        if manifest.records
        else "Add a source in the sidebar to get started"
    )
    st.session_state.pending_question = None

    if question:
        if not manifest.records:
            with st.chat_message("assistant"):
                st.warning(
                    "The knowledge base is empty, so there is nothing to search. "
                    "Add a website URL or a PDF in the sidebar first."
                )
            return
        answer_question(question, vectorstore, manifest, llm)


if __name__ == "__main__":
    main()
