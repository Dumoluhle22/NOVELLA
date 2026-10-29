# 📖 Novella

A semantic book recommendation app built for a Cyber & Data Engineering elective. Instead of searching by title or keyword, you describe the kind of story you want — mood, themes, setting — and Novella finds books that actually match.

WETHINKCODE vdrification code: WTC-JZWNKUTZ

## How it works

Novella is a three-stage pipeline:

**1. Embed** (`main.py`)
Cleans the raw book dataset (title, author, description) and generates vector embeddings for each book using OpenAI's `text-embedding-3-small` model. Books are processed in batches of 100 with checkpointing, so the script can be safely stopped and resumed without re-embedding anything already done.

**2. Index** (`vector_store.py`)
Loads the embeddings into a local ChromaDB vector database. The CSV is streamed in chunks rather than loaded all at once, since the full dataset (~90k+ books) is too large to comfortably fit in memory in one go.

**3. Search** (`app.py`)
A Streamlit app that:
- Takes a free-text query (e.g. *"a story set in Ghana exploring slavery, colonialism, and love"*)
- Expands it into a richer, theme-specific description using GPT-4o
- Runs a semantic similarity search against the book embeddings in ChromaDB
- Reranks the top candidates with an LLM to surface the 3 best actual fits, with a reason for each
- Looks up each recommended book on Open Library, and falls back to a Google search link if it isn't found there

## Tech stack

- **Python** / **pandas** — data cleaning
- **OpenAI API** — embeddings (`text-embedding-3-small`) + query expansion & reranking (`gpt-4o`)
- **ChromaDB** — vector database
- **Streamlit** — UI
- **Open Library API** — book lookup links

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Add your OpenAI API key in a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-key-here
   ```

4. Add your book dataset as `books.csv` in the project root, with at least `title`, `author`, and `desc` columns. (Not included in this repo due to file size — see note below.)

## Running it

Run these in order:

```
python main.py            # generates books_with_embeddings.csv
python vector_store.py    # builds the ChromaDB index
streamlit run app.py      # launches the search UI
```

`main.py` prints an estimated embedding cost before it starts, so you can check it looks reasonable first. If it's interrupted partway through, just rerun it — it resumes automatically from where it left off.

## Note on the dataset

`books.csv` isn't included in this repository because it exceeds GitHub's file size limit. It's a dataset of book titles, authors, and synopses — any CSV with those three columns will work with the pipeline as-is.

## Project background

This project was built for a Data Engineering elective, as the data engineering component. The focus is on the retrieval pipeline: cleaning and embedding a large dataset, indexing it for fast semantic search, and using a two-stage LLM process (query expansion + reranking) to improve result quality beyond simple vector similarity.
