from typing import Any, Dict, List


class SearchTools:
    """
    Provides search and file retrieval utilities over an indexed data store.
    """

    def __init__(
        self, index: Any, highlighter: Any, file_index: Dict[str, str]
    ) -> None:
        """
        Initialize the SearchTools instance.
        Args:
            index: Searchable index providing a `search` method.
            highlighter: Highlighter used to annotate search results.
            file_index (Dict[str, str]): Mapping of filenames to file contents.
        """
        self.index = index
        self.highlighter = highlighter
        self.file_index = file_index

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the index for results matching a query and highlight them.
        Args:
            query (str): The search query to look up in the index.
        Returns:
            List[Dict[str, Any]]: A list of highlighted search result objects.
        """
        search_results = self.index.search(query=query, num_results=5)
        return self.highlighter.highlight(query, search_results)

    def get_file(self, filename: str) -> str:
        """
        Retrieve a file's contents by filename.
        Args:
            filename (str): The filename of the file to retrieve.
        Returns:
            str: The file contents if found, otherwise an error message.
        """
        if filename in self.file_index:
            return self.file_index[filename]
        return f"file {filename} does not exist"
