def parse_markdown_file(markdown_path: str | Path) -> dict[str, str]:
    markdown_path = Path(markdown_path)
    text = markdown_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {"filename": markdown_path.name, "content": "\n".join(lines)}
parsed_docs = [parse_markdown_file(markdown_path)]
chunked_docs = chunk_documents(parsed_docs, size=3000, step=1500)
print(f"Chunked {len(chunked_docs)} chunks from {len(parsed_docs)} document")