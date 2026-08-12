import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def load_documents(docs_path="docs"):
    """Load all .txt files from the docs directory."""

    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}."
        )

    print(f"Loaded {len(documents)} documents.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


def split_documents(
    documents,
    chunk_size=512,
    chunk_overlap=50,
):
    """Split documents into smaller chunks."""

    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print("Content:")
        print(chunk.page_content)
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(
    chunks,
    persist_directory="db/chroma_db",
    batch_size=10,
):
    """Create and persist ChromaDB using batched embeddings."""

    print("\nCreating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large:335m"
    )

    # IMPORTANT:
    # We create the Chroma collection first.
    # We DO NOT use Chroma.from_documents().
    print("--- Creating vector store ---")

    vectorstore = Chroma(
        collection_name="company_documents",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    total_chunks = len(chunks)

    print(f"Total chunks: {total_chunks}")
    print(f"Batch size: {batch_size}")

    for start in range(0, total_chunks, batch_size):

        end = min(start + batch_size, total_chunks)

        batch = chunks[start:end]

        batch_number = (start // batch_size) + 1
        total_batches = (
            total_chunks + batch_size - 1
        ) // batch_size

        print(
            f"\nEmbedding batch "
            f"{batch_number}/{total_batches} "
            f"({start + 1}-{end} of {total_chunks})"
        )

        vectorstore.add_documents(batch)

        print(f"Batch {batch_number} completed.")

    print("\n--- Finished creating vector store ---")

    print(
        f"Vector store created and saved to "
        f"{persist_directory}"
    )

    return vectorstore



def main():
    """Main ingestion pipeline"""

    print("=== RAG Document Ingestion Pipeline ===\n")

    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print(
            "✅ Vector store already exists. "
            "No need to re-process documents."
        )

        # IMPORTANT:
        # Use the SAME embedding model that was used
        # when the vector store was created.
        embedding_model = OllamaEmbeddings(
            model="mxbai-embed-large:335m"
        )

        vectorstore = Chroma(
            collection_name="company_documents",
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print(
            f"Loaded existing vector store with "
            f"{vectorstore._collection.count()} documents"
        )

        return vectorstore

    print(
        "Persistent directory does not exist. "
        "Initializing vector store...\n"
    )

    # 1. Load documents
    documents = load_documents(
        docs_path=docs_path
    )

    # 2. Split documents
    chunks = split_documents(
        documents,
        chunk_size=512,
        chunk_overlap=50,
    )

    # 3. Create vector store
    vectorstore = create_vector_store(
        chunks,
        persist_directory=persistent_directory,
        batch_size=10,
    )


    print("INGESTION PIPELINE COMPLETED SUCCESSFULLY")

    return vectorstore


if __name__ == "__main__":
    main()