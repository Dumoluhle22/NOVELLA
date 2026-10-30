"""
Unit tests for vector_store.py — the ChromaDB loading logic.
Run with: pytest tests/test_vector_store.py -v
"""
import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vector_store


class TestVectorStore:
    """Tests for the chunk-cleaning, metadata-building, and ID-generation
    logic used when streaming embeddings into ChromaDB."""

    def test_clean_chunk_drops_rows_with_missing_fields(self):
        chunk = pd.DataFrame({
            "title": ["Book A", "Book B", "Book C"],
            "author": ["Author A", "Author B", "Author C"],
            "desc": ["Desc A", None, "Desc C"],
            "embedding": ["[0.1, 0.2]", "[0.3, 0.4]", "[0.5, 0.6]"],
        })
        cleaned = vector_store.clean_chunk(chunk)
        assert len(cleaned) == 2
        assert "Book B" not in cleaned["title"].values

    def test_clean_chunk_keeps_fully_populated_rows(self):
        chunk = pd.DataFrame({
            "title": ["Book A"],
            "author": ["Author A"],
            "desc": ["Desc A"],
            "embedding": ["[0.1, 0.2]"],
        })
        cleaned = vector_store.clean_chunk(chunk)
        assert len(cleaned) == 1

    def test_build_metadata_includes_title_and_author(self):
        chunk = pd.DataFrame({
            "title": ["Book A"],
            "author": ["Author A"],
            "desc": ["Desc A"],
        })
        metadata = vector_store.build_metadata(chunk)
        assert metadata == [{"title": "Book A", "author": "Author A"}]

    def test_build_metadata_includes_genre_when_present(self):
        chunk = pd.DataFrame({
            "title": ["Book A"],
            "author": ["Author A"],
            "genre": ["Romance"],
        })
        metadata = vector_store.build_metadata(chunk)
        assert metadata[0]["genre"] == "Romance"

    def test_build_metadata_does_not_fabricate_genre(self):
        # Regression test: an earlier version hardcoded a fake genre/trope
        # for every book. This confirms that no longer happens.
        chunk = pd.DataFrame({
            "title": ["Book A"],
            "author": ["Author A"],
        })
        metadata = vector_store.build_metadata(chunk)
        assert "genre" not in metadata[0]
        assert "trope" not in metadata[0]

    def test_parse_embeddings_converts_string_to_list(self):
        chunk = pd.DataFrame({"embedding": ["[0.1, 0.2, 0.3]"]})
        result = vector_store.parse_embeddings(chunk)
        assert result == [[0.1, 0.2, 0.3]]

    def test_parse_embeddings_handles_multiple_rows(self):
        chunk = pd.DataFrame({"embedding": ["[1.0, 2.0]", "[3.0, 4.0]"]})
        result = vector_store.parse_embeddings(chunk)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_generate_ids_returns_correct_count(self):
        ids = vector_store.generate_ids(start=0, count=5)
        assert len(ids) == 5

    def test_generate_ids_continues_from_start_value(self):
        ids = vector_store.generate_ids(start=2000, count=3)
        assert ids == ["2000", "2001", "2002"]

    def test_generate_ids_are_strings(self):
        ids = vector_store.generate_ids(start=0, count=2)
        assert all(isinstance(i, str) for i in ids)