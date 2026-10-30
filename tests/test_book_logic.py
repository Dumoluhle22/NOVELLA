"""
Unit tests for book_logic.py — query expansion, reranking, and book-link lookup.
Run with: pytest tests/test_book_logic.py -v
"""
import sys
import os
import json
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import book_logic


class TestSearchAndRecommendation:
    """Tests for the search app's logic layer: query expansion, LLM
    reranking, and finding a real link for each recommended book."""

    # --- expand_query ---

    def test_expand_query_returns_llm_text(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "A rich, descriptive rewritten query."
        )
        result = book_logic.expand_query(mock_client, "sad book")
        assert result == "A rich, descriptive rewritten query."

    def test_expand_query_includes_user_input_in_prompt(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"
        book_logic.expand_query(mock_client, "a book about dragons")

        sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert "a book about dragons" in sent_messages[0]["content"]

    # --- get_reranked_results ---

    def test_get_reranked_results_parses_valid_json(self):
        mock_client = MagicMock()
        fake_json = json.dumps({
            "recommendations": [
                {"title": "Book A", "reason": "Great match"},
                {"title": "Book B", "reason": "Also fits"},
            ]
        })
        mock_client.chat.completions.create.return_value.choices[0].message.content = fake_json

        fake_results = {
            "metadatas": [[{"title": "Book A"}, {"title": "Book B"}]],
            "documents": [["desc A", "desc B"]],
        }

        recs = book_logic.get_reranked_results(mock_client, "a query", fake_results)
        assert len(recs) == 2
        assert recs[0]["title"] == "Book A"

    def test_get_reranked_results_returns_empty_list_on_bad_json(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "this is not valid json"
        )
        fake_results = {
            "metadatas": [[{"title": "Book A"}]],
            "documents": [["desc A"]],
        }
        recs = book_logic.get_reranked_results(mock_client, "a query", fake_results)
        assert recs == []

    def test_get_reranked_results_requests_json_response_format(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            json.dumps({"recommendations": []})
        )
        fake_results = {"metadatas": [[{"title": "Book A"}]], "documents": [["desc A"]]}
        book_logic.get_reranked_results(mock_client, "a query", fake_results)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    # --- find_readable_book ---

    @patch("book_logic.requests.get")
    def test_find_readable_book_returns_public_read_link(self, mock_get):
        mock_get.return_value.json.return_value = {
            "docs": [{
                "title": "Dune",
                "ebook_access": "public",
                "availability": {"itemURL": "https://openlibrary.org/read/dune"},
            }]
        }
        mock_get.return_value.raise_for_status = MagicMock()

        result = book_logic.find_readable_book("Dune")
        assert result["url"] == "https://openlibrary.org/read/dune"
        assert "Read online" in result["label"]

    @patch("book_logic.requests.get")
    def test_find_readable_book_returns_borrow_link(self, mock_get):
        mock_get.return_value.json.return_value = {
            "docs": [{
                "title": "Dune",
                "ebook_access": "borrowable",
                "availability": {"itemURL": "https://openlibrary.org/borrow/dune"},
            }]
        }
        mock_get.return_value.raise_for_status = MagicMock()

        result = book_logic.find_readable_book("Dune")
        assert "Borrow" in result["label"]

    @patch("book_logic.requests.get")
    def test_find_readable_book_ignores_mismatched_titles(self, mock_get):
        mock_get.return_value.json.return_value = {
            "docs": [{
                "title": "A Completely Different Book",
                "ebook_access": "public",
                "availability": {"itemURL": "https://openlibrary.org/read/wrong"},
            }]
        }
        mock_get.return_value.raise_for_status = MagicMock()

        result = book_logic.find_readable_book("Dune")
        assert result is None

    @patch("book_logic.requests.get")
    def test_find_readable_book_returns_none_when_no_docs(self, mock_get):
        mock_get.return_value.json.return_value = {"docs": []}
        mock_get.return_value.raise_for_status = MagicMock()

        result = book_logic.find_readable_book("A Book That Doesn't Exist")
        assert result is None

    @patch("book_logic.requests.get")
    def test_find_readable_book_handles_network_errors_gracefully(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("network down")

        result = book_logic.find_readable_book("Dune")
        assert result is None

    # --- find_book_link ---

    @patch("book_logic.find_readable_book")
    def test_find_book_link_uses_open_library_result_when_available(self, mock_find_readable):
        mock_find_readable.return_value = {"url": "https://openlibrary.org/x", "label": "📖 Read online →"}
        result = book_logic.find_book_link("Dune")
        assert result["url"] == "https://openlibrary.org/x"

    @patch("book_logic.find_readable_book")
    def test_find_book_link_falls_back_to_google_search(self, mock_find_readable):
        mock_find_readable.return_value = None
        result = book_logic.find_book_link("Some Obscure Book", "Some Author")
        assert "google.com/search" in result["url"]
        assert "Search" in result["label"]