import chromadb
import pandas as pd
import ast
import os
import shutil

# 1. Config
DB_PATH = "./my_books_db"
INPUT_CSV = "books_with_embeddings.csv"
COLLECTION_NAME = "african_literature"
STREAM_BATCH_SIZE = 2000  # rows read from CSV and uploaded per chunk


def clean_chunk(chunk):
    """Drop rows missing required fields. Chroma rejects NaN documents, and
    a few rows can end up incomplete if main.py was interrupted/resumed."""
    return chunk.dropna(subset=['title', 'author', 'desc', 'embedding'])


def build_metadata(chunk):
    """Build the list of metadata dicts for a chunk. Only includes columns
    that actually exist, so nothing is fabricated (e.g. no fake genre)."""
    metadata_cols = ['title', 'author']
    if 'genre' in chunk.columns:
        metadata_cols.append('genre')
    return chunk[metadata_cols].to_dict(orient='records')


def parse_embeddings(chunk):
    """Convert the embedding column from its string representation back
    into actual lists of floats."""
    return [ast.literal_eval(x) for x in chunk['embedding'].tolist()]


def generate_ids(start, count):
    """Generate sequential string IDs for a batch of rows, continuing a
    running counter across chunks so IDs stay unique."""
    return [str(i) for i in range(start, start + count)]


def build_database():
    """Streams books_with_embeddings.csv into a fresh ChromaDB collection,
    chunked to avoid loading the whole file into memory at once."""
    if os.path.exists(DB_PATH):
        print("Removing old database folder...")
        shutil.rmtree(DB_PATH)

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.create_collection(name=COLLECTION_NAME)

    if not os.path.exists(INPUT_CSV):
        raise SystemExit(
            f"ERROR: '{INPUT_CSV}' not found. Run main.py first to generate embeddings."
        )

    print("Loading and uploading books from CSV in streams...")
    row_counter = 0

    for chunk in pd.read_csv(INPUT_CSV, chunksize=STREAM_BATCH_SIZE):
        before = len(chunk)
        chunk = clean_chunk(chunk)
        skipped = before - len(chunk)
        if skipped:
            print(f"  Skipped {skipped} row(s) with missing data in this chunk.")
        if len(chunk) == 0:
            continue

        current_embeddings = parse_embeddings(chunk)
        current_ids = generate_ids(row_counter, len(chunk))
        row_counter += len(chunk)
        current_documents = chunk['desc'].tolist()
        current_metadatas = build_metadata(chunk)

        collection.add(
            ids=current_ids,
            embeddings=current_embeddings,
            metadatas=current_metadatas,
            documents=current_documents,
        )

        print(f"  Inserted a batch of {len(chunk)} books ({row_counter} total so far).")

    print(f"\nSuccess! Total books now in database: {collection.count()}")


if __name__ == "__main__":
    build_database()