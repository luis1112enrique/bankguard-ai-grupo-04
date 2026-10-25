"""
9_agentic_rag.py

Build a persistent Mexican banking-regulation knowledge base in Pinecone.

Corpus:
- Ley de Instituciones de Crédito (LIC)
- Disposiciones del artículo 115 de la LIC
- Guía de evaluación de riesgos PLD/FT de la CNBV
- Guía del enfoque basado en riesgo del GAFI

Pipeline:
text files -> page-aware Documents -> legal chunks -> embeddings -> Pinecone
"""

import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DOCS_DIR = Path(__file__).resolve().parent / "rag_documents"
INDEX_NAME = os.environ["INDEX_NAME"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-large")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
CORPUS_ID = os.getenv("RAG_CORPUS_ID", "mx-pld-ft")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", CORPUS_ID)
DELETE_STALE_VECTORS = os.getenv("DELETE_STALE_VECTORS", "true").lower() in {
    "1",
    "true",
    "yes",
}

TEXT_SOURCE_METADATA = {
    "LIC.txt": {
        "title": "Ley de Instituciones de Crédito",
        "document_type": "law",
        "authority": "Cámara de Diputados del H. Congreso de la Unión",
        "jurisdiction": "México",
        "language": "es",
        "last_reform": "2025-11-14",
    },
    "Disposiciones_Articulo_115_LIC.txt": {
        "title": (
            "Disposiciones de carácter general a que se refiere el artículo "
            "115 de la Ley de Instituciones de Crédito"
        ),
        "document_type": "regulation",
        "authority": "Secretaría de Hacienda y Crédito Público",
        "jurisdiction": "México",
        "language": "es",
        "legal_basis": "Artículo 115 de la Ley de Instituciones de Crédito",
    },
    "CNBV_Guia_Metodologia_Evaluacion_Riesgos_PLD_FT_2019.txt": {
        "title": "Guía para la metodología de evaluación de riesgos PLD/FT",
        "document_type": "methodology_guide",
        "authority": "Comisión Nacional Bancaria y de Valores",
        "jurisdiction": "México",
        "language": "es",
        "publication_date": "2019-09-06",
        "subject": "Prevención de lavado de dinero y financiamiento al terrorismo",
    },
    "GAFI_Enfoque_Basado_en_Riesgo_Sector_Bancario_2014.txt": {
        "title": "Guía del enfoque basado en riesgo para el sector bancario",
        "document_type": "international_guidance",
        "authority": "Grupo de Acción Financiera Internacional",
        "jurisdiction": "Internacional",
        "language": "es",
        "publication_date": "2014-10",
        "subject": "Enfoque basado en riesgo para ALA/CFT",
    },
}

PAGE_MARKER = re.compile(
    r"^=+\s*PÁGINA\s+(\d+)\s+DE\s+\d+\s*=+\s*$",
    flags=re.MULTILINE,
)

ARTICLE_PATTERN = re.compile(
    r"(?im)^\s*ART[IÍ]CULO\s+([0-9]+(?:\s+(?:BIS|TER|QUÁTER))?)\b"
)
PROVISION_PATTERN = re.compile(r"(?m)^\s*(\d+[ªºa]?)\.-")
CHAPTER_PATTERN = re.compile(r"(?im)^\s*CAP[IÍ]TULO\s+([^\n]+)")
PAGE_COUNTER_PATTERN = re.compile(r"^(\d+)(?:\s+de\s+\d+)?$", flags=re.IGNORECASE)


def load_pdf(path: Path) -> list[Document]:
    """Load one LangChain Document per PDF page."""
    reader = PdfReader(path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "corpus_id": CORPUS_ID,
                        "source": path.name,
                        "source_path": path.relative_to(PROJECT_DIR).as_posix(),
                        "format": "pdf",
                        "document_type": "pdf",
                        "page": page_number,
                    },
                )
            )
    return documents


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving legal paragraphs and headings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_text_pages(text: str):
    """Yield (page, text) from form feeds or explicit page markers."""
    markers = list(PAGE_MARKER.finditer(text))
    if markers:
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            content = normalize_text(text[marker.end() : end])
            if content:
                yield int(marker.group(1)), content
        return

    if "\f" in text:
        for page_number, page_text in enumerate(text.split("\f"), start=1):
            content = normalize_text(page_text)
            if content:
                yield page_number, content
        return

    content = normalize_text(text)
    if content:
        yield None, content


def load_text(path: Path) -> list[Document]:
    """Load UTF-8 regulatory text and preserve page-level citations."""
    raw_text = path.read_text(encoding="utf-8-sig")
    file_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    source_metadata = TEXT_SOURCE_METADATA.get(
        path.name,
        {
            "title": path.stem.replace("_", " "),
            "document_type": "reference_text",
            "language": "es",
        },
    )

    documents = []
    for page_number, page_content in iter_text_pages(raw_text):
        metadata = {
            "corpus_id": CORPUS_ID,
            "source": path.name,
            "source_path": path.relative_to(PROJECT_DIR).as_posix(),
            "format": "text",
            "file_sha256": file_sha256,
            **source_metadata,
        }
        if page_number is not None:
            metadata["page"] = page_number

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def load_csv(path: Path) -> list[Document]:
    """Turn each catalog row into a readable Document."""
    documents = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_number, row in enumerate(reader, start=2):
            content = "\n".join(f"{key}: {value}" for key, value in row.items())

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "corpus_id": CORPUS_ID,
                        "source": path.name,
                        "source_path": path.relative_to(PROJECT_DIR).as_posix(),
                        "format": "csv",
                        "document_type": "data_catalog",
                        "row": row_number,
                        "field_name": row.get("field_name", ""),
                    },
                )
            )

    return documents


def load_corpus(folder: Path) -> list[Document]:
    """Load all supported files from the governance corpus."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {folder}")

    documents = []

    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            documents.extend(load_pdf(path))
        elif suffix == ".txt":
            documents.extend(load_text(path))
        elif suffix == ".csv":
            documents.extend(load_csv(path))

    return documents


def merge_small_chunks(
    chunks: list[Document], minimum_size: int = 80
) -> list[Document]:
    """Merge short headings with nearby text and discard verified page counters."""
    pending = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        page_counter = PAGE_COUNTER_PATTERN.fullmatch(content)
        if (
            page_counter
            and "page" in chunk.metadata
            and int(page_counter.group(1)) == chunk.metadata["page"]
        ):
            continue
        chunk.page_content = content
        pending.append(chunk)

    merged = []
    index = 0
    while index < len(pending):
        chunk = pending[index]
        unit = (
            chunk.metadata.get("source"),
            chunk.metadata.get("page"),
            chunk.metadata.get("row"),
        )

        if len(chunk.page_content) < minimum_size and index + 1 < len(pending):
            next_chunk = pending[index + 1]
            next_unit = (
                next_chunk.metadata.get("source"),
                next_chunk.metadata.get("page"),
                next_chunk.metadata.get("row"),
            )
            if unit == next_unit:
                next_chunk.page_content = (
                    f"{chunk.page_content}\n\n{next_chunk.page_content.strip()}"
                )
                next_chunk.metadata["start_index"] = min(
                    chunk.metadata.get("start_index", 0),
                    next_chunk.metadata.get("start_index", 0),
                )
                index += 1
                continue

        if len(chunk.page_content) < minimum_size and merged:
            previous = merged[-1]
            previous_unit = (
                previous.metadata.get("source"),
                previous.metadata.get("page"),
                previous.metadata.get("row"),
            )
            if unit == previous_unit:
                previous.page_content = (
                    f"{previous.page_content.strip()}\n\n{chunk.page_content}"
                )
                index += 1
                continue

        merged.append(chunk)
        index += 1

    return merged


def split_corpus(documents: list[Document]) -> list[Document]:
    """Split documents on legal structure while retaining citation metadata."""
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        keep_separator="start",
        is_separator_regex=True,
        separators=[
            r"\n(?=T[IÍ]TULO\b)",
            r"\n(?=CAP[IÍ]TULO\b)",
            r"\n(?=SECCI[ÓO]N\b)",
            r"\n(?=ART[IÍ]CULO\b)",
            r"\n(?=\d+[ªºa](?:\s+BIS)?\.-)",
            r"\n\n+",
            r"\n",
            r"(?<=\.)\s+",
            r"\s+",
            "",
        ],
    )
    chunks = merge_small_chunks(
        [
            chunk
            for chunk in splitter.split_documents(documents)
            if chunk.page_content.strip()
        ]
    )

    source_indexes = defaultdict(int)
    source_counts = Counter(chunk.metadata["source"] for chunk in chunks)

    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_index = source_indexes[source]
        source_indexes[source] += 1

        content = chunk.page_content.strip()
        chunk.page_content = content
        chunk.metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_count": source_counts[source],
                "chunk_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "embedding_model": EMBEDDING_MODEL,
                "schema_version": 2,
            }
        )

        article = ARTICLE_PATTERN.search(content)
        provision = PROVISION_PATTERN.search(content)
        chapter = CHAPTER_PATTERN.search(content)
        if article:
            chunk.metadata["article"] = article.group(1)
        if provision:
            chunk.metadata["provision"] = provision.group(1)
        if chapter:
            chunk.metadata["chapter"] = chapter.group(1).strip()

        location = [source]
        if "page" in chunk.metadata:
            location.append(f"página {chunk.metadata['page']}")
        if article:
            location.append(f"artículo {article.group(1)}")
        elif provision:
            location.append(f"disposición {provision.group(1)}")
        chunk.metadata["citation"] = ", ".join(location)

    return chunks


def stable_chunk_id(doc: Document) -> str:
    """Create a deterministic ID so rerunning ingestion does not duplicate chunks."""
    identity = (
        f"{CORPUS_ID}|"
        f"{doc.metadata.get('source_path', doc.metadata.get('source', ''))}|"
        f"{doc.metadata.get('page', '')}|"
        f"{doc.metadata.get('row', '')}|"
        f"{doc.metadata.get('start_index', '')}|"
        f"{doc.metadata.get('chunk_index', '')}|"
        f"{doc.metadata.get('chunk_sha256', '')}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{CORPUS_ID}:{digest}"


def list_managed_vector_ids(index) -> set[str]:
    """List vectors owned by this corpus in its dedicated namespace."""
    return {
        vector_id
        for page in index.list(
            prefix=f"{CORPUS_ID}:",
            namespace=PINECONE_NAMESPACE,
        )
        for vector_id in page
    }


def batched(values: list[str], size: int):
    """Yield fixed-size slices from a list."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def sync_chunks(vector_store, index, chunks: list[Document]) -> tuple[int, int]:
    """Upsert the complete corpus, then remove obsolete managed vectors."""
    ids = [stable_chunk_id(chunk) for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk ID collision detected; ingestion stopped")

    existing_ids = list_managed_vector_ids(index)
    total = len(chunks)
    ingestion_batch_size = 64

    for start in range(0, total, ingestion_batch_size):
        end = min(start + ingestion_batch_size, total)
        vector_store.add_documents(
            documents=chunks[start:end],
            ids=ids[start:end],
            batch_size=32,
            embedding_chunk_size=ingestion_batch_size,
        )
        print(f"Embedded and upserted {end}/{total} chunks")

    stale_ids = sorted(existing_ids - set(ids))
    if DELETE_STALE_VECTORS:
        for stale_batch in batched(stale_ids, 500):
            index.delete(ids=stale_batch, namespace=PINECONE_NAMESPACE)

    return len(ids), len(stale_ids) if DELETE_STALE_VECTORS else 0


def ensure_vector_index(client: Pinecone, dimension: int) -> None:
    """Create a Pinecone vector index when it does not already exist."""
    if not client.has_index(INDEX_NAME):
        print(
            f"Creating Pinecone index {INDEX_NAME!r} "
            f"(dimension={dimension}, metric=cosine)"
        )
        client.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            vector_type="dense",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        return

    index = client.describe_index(INDEX_NAME)
    if index.dimension != dimension:
        raise ValueError(
            f"Pinecone index {INDEX_NAME!r} has dimension {index.dimension}, "
            f"but {EMBEDDING_MODEL!r} produces dimension {dimension}."
        )
    if index.metric != "cosine":
        raise ValueError(
            f"Pinecone index {INDEX_NAME!r} uses metric {index.metric!r}; "
            "this example expects 'cosine'."
        )

    print(f"Using existing Pinecone index: {INDEX_NAME}")


if __name__ == "__main__":
    print(f"Loading regulatory documents from: {DOCS_DIR}")

    documents = load_corpus(DOCS_DIR)
    if not documents:
        raise RuntimeError(f"No supported documents found in {DOCS_DIR}")
    print(f"Loaded {len(documents)} source units/pages/rows")

    chunks = split_corpus(documents)
    if not chunks:
        raise RuntimeError("The corpus produced no chunks; ingestion stopped")
    print(f"Created {len(chunks)} chunks")
    for source, count in sorted(
        Counter(chunk.metadata["source"] for chunk in chunks).items()
    ):
        print(f"  {source}: {count} chunks")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    embedding_dimension = len(embeddings.embed_query("embedding dimension probe"))

    pinecone = Pinecone()
    ensure_vector_index(pinecone, embedding_dimension)
    index = pinecone.Index(INDEX_NAME)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    ingested_count, deleted_count = sync_chunks(vector_store, index, chunks)

    print(
        f"Ingested {ingested_count} chunks into Pinecone index "
        f"{INDEX_NAME}, namespace {PINECONE_NAMESPACE!r}"
    )
    if deleted_count:
        print(f"Deleted {deleted_count} stale chunks from the managed corpus")
