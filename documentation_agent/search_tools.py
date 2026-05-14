from typing import Any, Dict, List

from gitsource import GithubRepositoryDataReader
from minsearch import AppendableIndex, Highlighter, Tokenizer
from minsearch.tokenizer import DEFAULT_ENGLISH_STOP_WORDS


def build_index(
    repo_owner: str, repo_name: str
) -> tuple[AppendableIndex, Highlighter, Dict[str, str]]:
    """Read a GitHub docs repo, index it, and return the index + highlighter + file map."""
    reader = GithubRepositoryDataReader(
        repo_owner=repo_owner,
        repo_name=repo_name,
        allowed_extensions={"md", "mdx"},
    )
    files = reader.read()
    parsed_docs = [doc.parse() for doc in files]

    index = AppendableIndex(
        text_fields=["title", "description", "content"],
        keyword_fields=["filename"],
    )
    index.fit(parsed_docs)

    stopwords = DEFAULT_ENGLISH_STOP_WORDS | {"evidently"}
    highlighter = Highlighter(
        highlight_fields=["content"],
        max_matches=3,
        snippet_size=50,
        tokenizer=Tokenizer(stemmer="snowball", stop_words=stopwords),
    )

    file_index = {doc["filename"]: doc["content"] for doc in parsed_docs}

    return index, highlighter, file_index


class SearchTools:
    """Search and file retrieval utilities over an indexed documentation store."""

    def __init__(
        self, index: Any, highlighter: Any, file_index: Dict[str, str]
    ) -> None:
        self.index = index
        self.highlighter = highlighter
        self.file_index = file_index

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search the index for results matching a query and highlight them."""
        search_results = self.index.search(query=query, num_results=5)
        return self.highlighter.highlight(query, search_results)

    def get_file(self, filename: str) -> str:
        """Retrieve a file's contents by filename."""
        if filename in self.file_index:
            return self.file_index[filename]
        return f"file {filename} does not exist"
