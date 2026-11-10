# phase2_embeddings.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

def test_embeddings():
    # 1. LOAD: Read the text file from your folder
    print("Loading document...")
    loader = TextLoader("book.txt", encoding="utf-8")
    document = loader.load()

    # 2. CHUNK: Split into smaller pieces
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Max 500 characters per chunk
        chunk_overlap=50 # Share 50 characters between chunks to keep context
    )
    chunks = text_splitter.split_documents(document)
    print(f"Success! Book split into {len(chunks)} chunks.")

    # Show the very first chunk so you can see it
    print("\n--- First Chunk of Text ---")
    print(chunks[0].page_content)

    # 3. EMBED: Convert that text into numbers using the local model
    print("\nConnecting to local embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Translate the first chunk into a mathematical vector
    print("Translating chunk into a vector...")
    vector = embeddings.embed_query(chunks[0].page_content)

    # Show the results!
    print("\n--- Vector Data (The Math) ---")
    print(f"Total numbers in this single vector: {len(vector)}")
    print(f"First 5 numbers: {vector[:5]}")

if __name__ == "__main__":
    test_embeddings()