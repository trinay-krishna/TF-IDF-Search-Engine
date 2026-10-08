import re

from document import Document
from query import Query


def parse_file(path, kind="document"):
    """Parses a CISI-format file into Document objects (kind="document")
    or Query objects (kind="query")."""
    with open(file=path, mode="r", encoding="utf-8") as file:
        text = file.read()

    items = []

    #Converts the raw dataset text to structured records.
    records = text.split(".I ")[1:]

    for record in records:
        item_id = int(record.split("\n", 1)[0].strip())

        if kind == "query":
            #Queries have no .X section, so content runs to the next field or end of record.
            content_match = re.search(r"\.W[ \t]*\n(.*?)(?=\n\.[A-Z]|\Z)", record, re.DOTALL)
            content = content_match.group(1).strip() if content_match else ""
            items.append(Query(id=item_id, content=content))
            continue

        title_match = re.search(r"\.T[ \t]*\n(.*?)\n\.A", record, re.DOTALL)
        author_match = re.search(r"\.A[ \t]*\n(.*?)\n\.W", record, re.DOTALL)
        content_match = re.search(r"\.W[ \t]*\n(.*?)\n\.X", record, re.DOTALL)

        title = title_match.group(1).strip() if title_match else ""
        author = author_match.group(1).strip() if author_match else ""
        content = content_match.group(1).strip() if content_match else ""

        items.append(Document(id=item_id, title=title, author=author, content=content))

    return items
