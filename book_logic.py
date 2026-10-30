import requests
import json
from urllib.parse import quote

RERANK_MODEL = "gpt-4o"


def expand_query(client, user_input, model=RERANK_MODEL):
    """Rewrite a user's search query into a richer, theme-specific description."""
    prompt = (
        f"Rewrite this book search query to be descriptive and identify "
        f"specific tropes or themes: '{user_input}'"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_reranked_results(client, user_query, results, model=RERANK_MODEL):
    """Ask the LLM to pick and justify the top 3 books from a set of
    Chroma search results. Returns a list of {"title", "reason"} dicts,
    or an empty list if the response couldn't be parsed."""
    books_text = ""
    for i in range(len(results['metadatas'][0])):
        title = results['metadatas'][0][i]['title']
        desc = results['documents'][0][i]
        books_text += f"{i+1}. {title}: {desc}\n"

    prompt = f"""
    The user is looking for books matching this request: '{user_query}'

    Here are the retrieved books from our database:
    {books_text}

    Pick the top 3 books that best fit the user's request.

    Respond with ONLY valid JSON, no other text, in exactly this format:
    {{
      "recommendations": [
        {{"title": "Book Title", "reason": "brief reason why it fits"}},
        {{"title": "Book Title", "reason": "brief reason why it fits"}},
        {{"title": "Book Title", "reason": "brief reason why it fits"}}
      ]
    }}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        return parsed.get("recommendations", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def find_readable_book(title, author=""):
    """
    Try to find a legitimate online reading/borrowing option via Open Library.
    Returns {"url": ..., "label": ...} or None if nothing usable is found.
    """
    try:
        params = {
            "title": title,
            "limit": 10,
            "fields": "key,title,author_name,ebook_access,has_fulltext,edition_key,ia,availability",
        }
        if author:
            params["author"] = author

        response = requests.get(
            "https://openlibrary.org/search.json",
            params=params,
            timeout=8,
        )
        response.raise_for_status()
        docs = response.json().get("docs", [])

        for doc in docs:
            book_title = doc.get("title", "")
            if book_title.lower().strip() != title.lower().strip():
                continue

            ebook_access = doc.get("ebook_access")
            if ebook_access in ["public", "borrowable"]:
                availability = doc.get("availability", {})
                item_url = availability.get("itemURL")

                if item_url:
                    label = "📖 Read online →" if ebook_access == "public" else "📚 Borrow & read →"
                    return {"url": item_url, "label": label}

                key = doc.get("key")
                if key:
                    return {"url": f"https://openlibrary.org{key}", "label": "📖 Open in Open Library →"}

    except requests.RequestException:
        pass

    return None


def find_book_link(title, author=""):
    """
    Try Open Library for the exact book. If it's not there, fall back to a
    plain Google search so the user can look it up themselves.
    Returns {"url": ..., "label": ...}
    """
    readable = find_readable_book(title, author)
    if readable:
        return readable

    search_url = f"https://www.google.com/search?q={quote(title + ' ' + author + ' book')}"
    return {"url": search_url, "label": "🔍 Search on Google →"}