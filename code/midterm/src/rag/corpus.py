from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    filename: str
    text: str
    doc_type: str


def infer_doc_type(filename: str) -> str:
    stem = Path(filename).stem
    if stem.startswith("credit-card-bill"):
        return "credit_card_bill"
    if stem.startswith("invoice"):
        return "invoice"
    if stem.startswith("receipt"):
        return "receipt"
    if stem.startswith("bank-statement"):
        return "bank_statement"
    if stem.startswith("utility-bill"):
        return "utility_bill"
    if stem.startswith("pharmacy-receipt"):
        return "pharmacy_receipt"
    if stem.startswith("policy-"):
        return "policy"
    return "unknown"


def load_documents(doc_dir: Path) -> list[Document]:
    paths = sorted(Path(doc_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown documents found in {doc_dir}")
    docs = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                doc_id=path.stem,
                filename=path.name,
                text=text,
                doc_type=infer_doc_type(path.name),
            )
        )
    return docs
