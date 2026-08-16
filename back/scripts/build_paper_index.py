"""Utility script for building a FAISS index from wafer-related papers."""
import argparse
from pathlib import Path
from typing import Iterable, List, Set, Union

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from config.paths import PAPER_DIR, INDEX_DIR, TRACKING_FILE_PATH
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

# Load environment variables before accessing embedding providers.
load_dotenv()


# Normalizes incoming path-like values to absolute Path instances.
def _ensure_path(path_like: Union[Path, str]) -> Path:
    return Path(path_like).resolve()


def load_papers(paper_dir: Path, processed_files: Set[str]) -> Iterable[Document]:
    """Yield LangChain Document instances for new papers in the directory."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for path in paper_dir.rglob('*'):
        resolved = str(path.resolve())
        if resolved in processed_files:
            continue

        if path.suffix.lower() == '.pdf':
            loader = PyMuPDFLoader(str(path))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                yield chunk
            continue

        if path.suffix.lower() in {'.txt', '.md'}:
            text = path.read_text(encoding='utf-8', errors='ignore')
            docs = [Document(page_content=text, metadata={"source": resolved})]
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                yield chunk


def create_or_update_index(documents: List[Document], index_dir: Path) -> None:
    """Create a new FAISS index or update an existing one."""
    if not documents:
        print("No new documents found. Skipping index update.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    index_file = index_dir / "index.faiss"
    if index_file.exists():
        print(f"Loading existing FAISS index from {index_dir}.")
        vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"Appending {len(documents)} new document chunks to the index.")
        vector_store.add_documents(documents)
    else:
        print("Creating a fresh FAISS index.")
        index_dir.mkdir(parents=True, exist_ok=True)
        vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(str(index_dir))
    print(f"Index persisted to {index_dir}.")


# Loads the tracking file that records already processed sources.
def _load_processed_files(tracking_file: Path) -> Set[str]:
    if not tracking_file.exists():
        return set()
    return {
        line.strip()
        for line in tracking_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    }


# Appends newly processed sources to the tracking file.
def _append_processed_files(tracking_file: Path, paths: Set[str]) -> None:
    if not paths:
        return
    tracking_file.parent.mkdir(parents=True, exist_ok=True)
    with tracking_file.open('a', encoding='utf-8') as handle:
        for item in sorted(paths):
            handle.write(f"{item}\n")

# Rebuilds the FAISS index and returns True when new content was indexed.
def rebuild_index(
    paper_dir: Union[Path, str] = PAPER_DIR,
    index_dir: Union[Path, str] = INDEX_DIR,
    tracking_file: Union[Path, str] = TRACKING_FILE_PATH,
) -> bool:
    paper_dir_path = _ensure_path(paper_dir)
    index_dir_path = _ensure_path(index_dir)
    tracking_file_path = _ensure_path(tracking_file)

    processed_files = _load_processed_files(tracking_file_path)
    print(f"Tracking {len(processed_files)} previously processed sources.")

    new_documents = list(load_papers(paper_dir_path, processed_files))
    if not new_documents:
        print("No new documents detected. Nothing to index.")
        return False

    create_or_update_index(new_documents, index_dir_path)

    newly_processed_paths = {
        doc.metadata.get('source', '')
        for doc in new_documents
        if doc.metadata.get('source')
    }
    _append_processed_files(tracking_file_path, newly_processed_paths)
    print(f"Logged {len(newly_processed_paths)} processed sources.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Build or update FAISS index for wafer papers.')
    parser.add_argument('--paper-dir', type=Path, default=_ensure_path(PAPER_DIR))
    parser.add_argument('--index-dir', type=Path, default=_ensure_path(INDEX_DIR))
    parser.add_argument('--tracking-file', type=Path, default=_ensure_path(TRACKING_FILE_PATH))
    args = parser.parse_args()

    updated = rebuild_index(
        paper_dir=args.paper_dir,
        index_dir=args.index_dir,
        tracking_file=args.tracking_file,
    )
    if not updated:
        raise SystemExit()


if __name__ == '__main__':
    main()
