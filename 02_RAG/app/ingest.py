from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHROMA_DIR, EMBEDDING_MODEL, RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME

def load_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            loaded = loader.load()

        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
            loaded = loader.load()

        elif suffix == ".doc":
            print(f"Skipping legacy DOC file: {path.name}")
            continue

        elif suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding = True)
            loaded = loader.load()

        else:
            continue

        for doc in loaded:
            doc.metadata["source"] = path.name
            doc.metadata["source_path"] = str(
                path.relative_to(folder)
            )
            documents.append(doc)

    return documents

def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

def build_index() -> int:
    docs = load_documents(RAW_DOCS_DIR)

    if not docs:
        raise RuntimeError(
            f"No readable PDF, DOCX, or TXT documents found in "
            f"{RAW_DOCS_DIR}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        add_start_index = True,
    )
    chunks = splitter.split_documents(docs)
    vector_store = get_vector_store()

    # Group chunks by source file.
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_path", "unknown")
        chunks_by_source.setdefault(source, []).append(chunk)

    for source, source_chunks in chunks_by_source.items():
        # Find and remove the existing chunks for this source.
        existing = vector_store.get(
            where={"source_path": source},
            include=[],
        )

    old_ids = existing.get("ids", [])

    if old_ids:
        vector_store.delete(ids=old_ids)

    # Deterministic IDs make repeat ingestion idempotent
    # for unchanged content and metadata.
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source_path", "unknown")
        page = chunk.metadata.get("page", -1)
        start = chunk.metadata.get("start_index", 0)
        ids.append(
            f"{source}:{page}:{start}"
        )

    if source_chunks:
        vector_store.add_documents(
            documents=source_chunks,
            ids=ids,
        )

    # Re-ingest of the same IDs updates existing chunks.
    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(f"Loaded {len(docs)} document pages/files.")
    print(f"Created {len(chunks)} chunks.")
    print(f"Index stored in {CHROMA_DIR}")

    return len(chunks)


if __name__ == "__main__":
    build_index()