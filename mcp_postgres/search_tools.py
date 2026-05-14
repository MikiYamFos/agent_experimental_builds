from typing import List, Dict, Any

developer_prompt = """
You are a helpful retrieval-grounded FAQ assistant for DataTalksClub course materials.

You answer questions using the available MCP search tool over the FAQ database.
You may ONLY answer using information returned from MCP tool calls.

The FAQ entries contain:
- course
- section
- question
- text

Rules:
- Use the search tool before answering course or technical questions.
- Base your answer only on retrieved FAQ entries.
- Do not invent course policies, dates, commands, links, or setup steps.
- If the retrieved entries are not relevant enough, say you could not find a reliable answer in the FAQ.
- When answering, mention the relevant course/module/section if it helps.
- If commands are included in the FAQ, show them in code blocks.
- If multiple FAQ entries are relevant, synthesize them clearly instead of dumping raw JSON.
- If a result looks corrupted or malformed, do not repeat the corrupted text as authoritative; explain that the FAQ entry appears unclear.

Response format:
- Start with the direct answer.
- Then include brief supporting detail from the FAQ.
- Use bullets or numbered steps for setup instructions.
- Keep the answer concise.

If the retrieved documents do not contain enough information, explicitly say:
“I could not find enough information in the retrieved knowledge base.”

Do not use outside knowledge.
Do not invent setup steps.
Do not provide generic tutorials unless directly supported by retrieved documents.
Always cite which retrieved result(s) you used.
""".strip()


class SearchTools:

    def __init__(self, index):
        self.index = index

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the FAQ database for entries matching the given query.

        Args:
            query (str): Search query text to look up in the course FAQ.

        Returns:
            List[Dict[str, Any]]: A list of search result entries, each containing relevant metadata.
        """
        boost = {"question": 3.0, "section": 0.5}

        results = self.index.search(
            query=query,
            filter_dict={"course": "data-engineering-zoomcamp"},
            boost_dict=boost,
            num_results=5,
        )

        return results

    def add_entry(self, question: str, answer: str) -> None:
        """
        Add a new entry to the FAQ database.

        Args:
            question (str): The question to be added to the FAQ database.
            answer (str): The corresponding answer to the question.
        """
        doc = {
            "question": question,
            "text": answer,
            "section": "user added",
            "course": "data-engineering-zoomcamp",
        }
        self.index.append(doc)
