"""
Unit tests for main.py — the embedding/cleaning pipeline.
Run with: pytest tests/test_main.py -v
"""
import sys
import os
from unittest.mock import MagicMock
import pandas as pd
import pytest

# Make the project root importable when running pytest from the project folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main


class TestEmbeddingPipeline:
    """Tests for the data cleaning and embedding logic in main.py."""

    def test_clean_text_strips_html_tags(self):
        result = main.clean_text("A <b>great</b> story about <i>love</i>")
        assert "<" not in result
        assert ">" not in result
        assert "great" in result
        assert "love" in result

    def test_clean_text_removes_special_characters(self):
        result = main.clean_text("Amazing!! @#$% story... totally worth it?")
        # Allowed punctuation (. , ! ? ' -) should survive, @#$% should not
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result

    def test_clean_text_collapses_whitespace(self):
        result = main.clean_text("Too   many    spaces\n\nand lines")
        assert "  " not in result

    def test_clean_text_handles_non_string_input(self):
        # NaN/float values should return an empty string, not crash
        assert main.clean_text(float("nan")) == ""
        assert main.clean_text(None) == ""
        assert main.clean_text(123) == ""

    def test_clean_dataframe_drops_incomplete_rows(self):
        df = pd.DataFrame({
            "title": ["Book A", None, "Book C"],
            "author": ["Author A", "Author B", "Author C"],
            "desc": ["A story.", "Another story.", None],
        })
        cleaned = main.clean_dataframe(df)
        # Only "Book A" has all three fields present
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["title"] == "Book A"

    def test_clean_dataframe_builds_combined_text(self):
        df = pd.DataFrame({
            "title": ["Dune"],
            "author": ["Frank Herbert"],
            "desc": ["A story about a desert planet."],
        })
        cleaned = main.clean_dataframe(df)
        combined = cleaned.iloc[0]["combined_text"]
        assert "Dune" in combined
        assert "Frank Herbert" in combined
        assert "desert planet" in combined

    def test_estimate_cost_returns_positive_numbers(self):
        df = pd.DataFrame({"combined_text": ["a" * 1000, "b" * 2000]})
        tokens, cost = main.estimate_cost(df)
        assert tokens > 0
        assert cost > 0

    def test_estimate_cost_scales_with_text_length(self):
        small_df = pd.DataFrame({"combined_text": ["short text"]})
        big_df = pd.DataFrame({"combined_text": ["much longer text " * 100]})
        _, small_cost = main.estimate_cost(small_df)
        _, big_cost = main.estimate_cost(big_df)
        assert big_cost > small_cost

    def test_embed_batch_calls_openai_with_correct_args(self):
        mock_client = MagicMock()
        fake_embedding_item = MagicMock()
        fake_embedding_item.embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value.data = [fake_embedding_item]

        result = main.embed_batch(mock_client, ["some book description"])

        mock_client.embeddings.create.assert_called_once_with(
            input=["some book description"], model=main.EMBED_MODEL
        )
        assert result == [[0.1, 0.2, 0.3]]

    def test_embed_batch_preserves_order_for_multiple_texts(self):
        mock_client = MagicMock()
        item1, item2 = MagicMock(), MagicMock()
        item1.embedding = [1, 1, 1]
        item2.embedding = [2, 2, 2]
        mock_client.embeddings.create.return_value.data = [item1, item2]

        result = main.embed_batch(mock_client, ["book one", "book two"])

        assert result == [[1, 1, 1], [2, 2, 2]]