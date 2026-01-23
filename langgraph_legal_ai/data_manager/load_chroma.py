import re
from pathlib import Path
from typing import Literal

from langchain_chroma import Chroma
from langchain_classic.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

SECTION_PATTERN = re.compile(r"\n\s*(\d+[A-Z]?)\.\s+([A-Z][^\n]+)", re.MULTILINE)

SUBSECTION_PATTERN = re.compile(r"\(\s*(\d+[A-Z]?)\s*\)")

SCHEDULE_PATTERN = re.compile(
    r"\n\s*THE\s+(SECTION|CLAUSE|FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+SCHEDULE",
    re.IGNORECASE,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Vector Store (Main Knowledge Base)
db = Chroma(
    persist_directory="../chroma",
    embedding_function=embeddings,
    collection_name="legal",
)
print(f"Chroma DB initialized with {db._collection.count()} documents.")


def extract_act_name(file_path: str) -> str:
    name = Path(file_path).stem  # Sale_of_Goods_Act_1930
    name = name.replace("_", " ")  # Sale of Goods Act 1930
    return name


def split_pdf(text, metadata):
    documents = []

    section_matches = list(SECTION_PATTERN.finditer(text))
    schedule_matches = list(SCHEDULE_PATTERN.finditer(text))

    # --- SECTIONS ---
    for i, match in enumerate(section_matches):
        section_no = match.group(1)
        section_title = match.group(2).strip()

        start = match.end()
        end = (
            section_matches[i + 1].start()
            if i + 1 < len(section_matches)
            else len(text)
        )

        section_body = text[start:end].strip()

        # --- SUBSECTIONS ---
        sub_matches = list(SUBSECTION_PATTERN.finditer(section_body))

        if sub_matches:
            for j, sm in enumerate(sub_matches):
                sub_no = sm.group(1)

                ss_start = sm.start()
                ss_end = (
                    sub_matches[j + 1].start()
                    if j + 1 < len(sub_matches)
                    else len(section_body)
                )

                subsection_text = section_body[ss_start:ss_end].strip()

                documents.append(
                    Document(
                        page_content=(
                            f"Section {section_no}({sub_no}). {section_title}\n"
                            f"{subsection_text}"
                        ),
                        metadata={
                            "act": metadata["act"],
                            "section": section_no,
                            "subsection": sub_no,
                            "title": section_title,
                            "type": "subsection",
                            "source": metadata["source"],
                        },
                    )
                )
        else:
            documents.append(
                Document(
                    page_content=(
                        f"Section {section_no}. {section_title}\n" f"{section_body}"
                    ),
                    metadata={
                        "act": metadata["act"],
                        "section": section_no,
                        "title": section_title,
                        "type": "section",
                        "source": "IndiaCode",
                    },
                )
            )

    # --- SCHEDULES ---
    for i, match in enumerate(schedule_matches):
        schedule_name = match.group(1).title()

        start = match.end()
        end = (
            schedule_matches[i + 1].start()
            if i + 1 < len(schedule_matches)
            else len(text)
        )

        schedule_text = text[start:end].strip()

        documents.append(
            Document(
                page_content=f"{schedule_name} Schedule\n{schedule_text}",
                metadata={
                    "act": metadata["act"],
                    "schedule": schedule_name,
                    "type": "schedule",
                    "source": "IndiaCode",
                },
            )
        )

    return documents


def load_to_chroma(db, docs):
    ids = []
    for i, d in enumerate(docs):
        if d.metadata["type"] == "subsection":
            ids.append(
                f"{d.metadata['act']}_s{d.metadata['section']}_{d.metadata['subsection']}_{i}"
            )
        elif d.metadata["type"] == "section":
            ids.append(f"{d.metadata['act']}_s{d.metadata['section']}_{i}")
        else:
            ids.append(
                f"{d.metadata['act']}_schedule_{d.metadata['schedule'].lower()}_{i}"
            )

    db.add_documents(docs, ids=ids)


def load_legal_document(
    relative_path: str, path_type: Literal["File", "Folder"] = "File"
):
    print(f"Loading Started with {relative_path} ... {path_type}")
    if path_type == "Folder":
        path = Path(relative_path)
        file_names = path.glob("**/*.pdf")
    else:
        file_names = [relative_path]

    for file_name in file_names:
        print(f"Processing {file_name} ... ", end=" ")

        loader = PyMuPDFLoader(file_name)
        loaded_pdf_data = loader.load()

        print("Loaded ... ", end=" ")
        metadata = loaded_pdf_data[0].metadata
        full_text = "\n".join(e.page_content for e in loaded_pdf_data if e.page_content)
        docs = split_pdf(
            full_text,
            {"source": metadata["source"], "act": extract_act_name(metadata["source"])},
        )
        print("Chunked ... ", end=" ")
        load_to_chroma(db, docs)
        print("Loaded to Chroma DB Successfully.")

    print(f"Chroma DB has {db._collection.count()} documents.")


load_legal_document("../legal_corpus", "Folder")
