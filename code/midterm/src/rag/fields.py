from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from datetime import datetime

from .corpus import Document

DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
MONEY_RE = re.compile(r"-?\$?\s*-?\d[\d,]*(?:\.\d+)?")
INVOICE_RE = re.compile(r"\bINV[-\s]?\d{4}[-\s]?\d+\b", re.I)
ACCOUNT_RE = re.compile(r"\b\d{2}-\d{4}-\d{4}\b")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_identifier(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def normalize_account(value: str) -> str:
    return re.sub(r"\s+", "", value)


def normalize_date(value: str) -> str:
    match = DATE_RE.search(value)
    if not match:
        return value
    month, day, year = (int(match.group(i)) for i in (1, 2, 3))
    if month > 12 and day <= 12:
        month, day = day, month
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return f"{year:04d}-{month:02d}-{day:02d}"


def extract_dates(text: str) -> list[str]:
    seen = []
    for match in DATE_RE.finditer(text):
        value = normalize_date(match.group(0))
        if value not in seen:
            seen.append(value)
    return seen


def parse_money(value: str | None) -> float | None:
    if not value:
        return None
    match = MONEY_RE.search(value)
    if not match:
        return None
    raw = match.group(0).replace("$", "").replace(",", "").replace(" ", "")
    if raw.startswith("$-"):
        raw = "-" + raw[2:]
    try:
        return float(raw)
    except ValueError:
        return None


def _first(pattern: str, text: str, flags: int = re.I | re.M) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _money(pattern: str, text: str, flags: int = re.I | re.M) -> float | None:
    return parse_money(_first(pattern, text, flags))


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_payment(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    if "CASH" in upper:
        return "CASH"
    if "DEBIT" in upper:
        return "DEBIT"
    if any(word in upper for word in ("VISA", "MASTERCARD", "AMEX", "DISCOVER", "CREDIT")):
        return "CREDIT"
    return upper.split()[0]


def _parse_bank_transactions(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4 or not DATE_RE.search(parts[0]):
            continue
        amount = parse_money(parts[2])
        balance = parse_money(parts[3])
        if amount is None:
            continue
        rows.append(
            {
                "date": normalize_date(parts[0]),
                "description": parts[1],
                "amount": amount,
                "balance": balance,
            }
        )
    return rows


@dataclass
class DocFields:
    filename: str
    doc_type: str
    title: str
    text: str
    text_norm: str
    name: str | None = None
    company: str | None = None
    invoice_number: str | None = None
    account_number: str | None = None
    dates: dict[str, str] = dc_field(default_factory=dict)
    primary_dates: list[str] = dc_field(default_factory=list)
    all_dates: list[str] = dc_field(default_factory=list)
    amounts: dict[str, float] = dc_field(default_factory=dict)
    payment_method: str | None = None
    transactions: list[dict] = dc_field(default_factory=list)
    extra: dict = dc_field(default_factory=dict)


def extract_fields(document: Document) -> DocFields:
    text = document.text
    lines = _nonempty_lines(text)
    title = lines[0] if lines else document.filename
    fields = DocFields(
        filename=document.filename,
        doc_type=document.doc_type,
        title=title,
        text=text,
        text_norm=normalize_text(text),
        all_dates=extract_dates(text),
    )

    if document.doc_type == "credit_card_bill":
        fields.name = _first(r"^Name:\s*(.+?)\s+Phone:", text)
        fields.dates["bill_received"] = normalize_date(_first(r"Bill Received:\s*([^\n]+)", text) or "")
        fields.dates["due_date"] = normalize_date(_first(r"Payment Due Date:\s*([^\n]+)", text) or "")
        fields.amounts["fine"] = _money(r"a\s+\$?([\d,]+\.\d{2})\s+fine", text)
        fields.amounts["total"] = _money(r"^\|?\s*Total:\s*([^|\n]+)", text)
        fields.amounts["previous_balance"] = _money(r"Previous Balance:\s*([^\n]+)", text)
        fields.amounts["new_balance"] = _money(r"New Balance:\s*([^\n]+)", text)
        fields.amounts["minimum_payment"] = _money(r"Minimum Payment Due:\s*([^\n]+)", text)

    elif document.doc_type == "invoice":
        fields.company = lines[1] if len(lines) > 1 else None
        fields.invoice_number = normalize_identifier(
            _first(r"Invoice Number:\s*([^\n]+)", text) or ""
        )
        fields.name = _first(r"Bill To:\s*\n\s*(?:Mr\./Ms\.\s*)?([^\n]+)", text)
        fields.dates["invoice_date"] = normalize_date(_first(r"Invoice Date:\s*([^\n]+)", text) or "")
        fields.dates["due_date"] = normalize_date(_first(r"Due Date:\s*([^\n]+)", text) or "")
        fields.amounts["subtotal"] = _money(r"Subtotal:\s*([^\n]+)", text)
        fields.amounts["sales_tax"] = _money(r"Sales Tax[^:]*:\s*([^\n]+)", text)
        fields.amounts["total_amount_due"] = _money(r"Total Amount Due:\s*([^\n]+)", text)

    elif document.doc_type == "receipt":
        fields.company = lines[0] if lines else None
        fields.extra["receipt_number"] = _first(r"Receipt #\s*([^\n]+?)(?:\s|$)", text)
        fields.dates["receipt_date"] = normalize_date(_first(r"\bDate:\s*([^\n]+)", text) or "")
        fields.amounts["subtotal"] = _money(r"Subtotal:\s*([^\n]+)", text)
        fields.amounts["discount"] = _money(r"Discount:\s*([^\n]+)", text)
        fields.amounts["tax"] = _money(r"^Tax:\s*([^\n]+)", text)
        fields.amounts["total"] = _money(r"^TOTAL:\s*([^\n]+)", text)
        fields.payment_method = normalize_payment(_first(r"^Payment:\s*([^\n]+)", text))

    elif document.doc_type == "bank_statement":
        fields.company = lines[1] if len(lines) > 1 else None
        fields.name = _first(r"Account Holder:\s*([^\n]+)", text)
        fields.account_number = normalize_account(_first(r"Account Number:\s*([^\n]+)", text) or "")
        period = _first(r"Statement Period:\s*([^\n]+)", text) or ""
        period_dates = extract_dates(period)
        if len(period_dates) >= 2:
            fields.dates["period_start"] = period_dates[0]
            fields.dates["period_end"] = period_dates[1]
        elif period_dates:
            fields.dates["period_start"] = period_dates[0]
        fields.amounts["opening_balance"] = _money(r"Opening Balance:\s*([^\n]+)", text)
        fields.amounts["ending_balance"] = _money(r"Ending Balance:\s*([^\n]+)", text)
        fields.transactions = _parse_bank_transactions(text)

    elif document.doc_type == "utility_bill":
        fields.company = lines[1] if len(lines) > 1 else None
        fields.account_number = normalize_account(_first(r"Account Number:\s*([^\n]+)", text) or "")
        fields.dates["statement_date"] = normalize_date(
            _first(r"Statement Date:\s*([^\n]+)", text) or ""
        )
        fields.dates["due_date"] = normalize_date(_first(r"Payment Due:\s*([^\n]+)", text) or "")
        fields.amounts["total_current_charges"] = _money(
            r"\|\s*Total Current Charges\s*\|\s*([^|]+)\|", text
        )
        fields.amounts["previous_balance"] = _money(r"Previous Balance:\s*([^\n]+)", text)
        fields.amounts["amount_due_now"] = _money(r"Amount Due Now:\s*([^\n]+)", text)
        for usage in ("Water", "Electricity", "Natural gas", "Internet"):
            match = re.search(rf"\|\s*{usage} usage\s*\(([\d,.]+)\s*([^)]+)\)", text, re.I)
            if match:
                fields.extra["usage_type"] = usage
                fields.extra["usage_quantity"] = float(match.group(1).replace(",", ""))
                fields.extra["usage_unit"] = match.group(2).strip()

    elif document.doc_type == "pharmacy_receipt":
        fields.company = lines[0] if lines else None
        fields.extra["receipt_number"] = _first(r"Receipt #\s*([^\n]+?)(?:\s|$)", text)
        fields.dates["receipt_date"] = normalize_date(_first(r"\bDate:\s*([^\n]+)", text) or "")
        fields.amounts["subtotal"] = _money(r"Subtotal:\s*([^\n]+)", text)
        fields.amounts["tax"] = _money(r"^Tax:\s*([^\n]+)", text)
        fields.amounts["total"] = _money(r"^TOTAL:\s*([^\n]+)", text)
        fields.amounts["copay"] = _money(r"Copay:\s*([^\n]+)", text)
        fields.payment_method = normalize_payment(
            _first(r"^Paid by:\s*([^\n$]+)", text)
        )
        fields.extra["insurance"] = _first(r"Insurance billed:\s*([^\n]+)", text)

    fields.primary_dates = []
    for value in fields.dates.values():
        if value and value not in fields.primary_dates:
            fields.primary_dates.append(value)
    return fields


def extract_query_signals(question: str, all_fields: list[DocFields]) -> dict:
    normalized = normalize_text(question)
    invoice_numbers = {
        normalize_identifier(match.group(0)) for match in INVOICE_RE.finditer(question)
    }
    account_numbers = {normalize_account(match.group(0)) for match in ACCOUNT_RE.finditer(question)}
    query_dates = set(extract_dates(question))
    names = set()
    companies = set()
    for fields in all_fields:
        if fields.name and normalize_text(fields.name) in normalized:
            names.add(fields.name)
        if fields.company and normalize_text(fields.company) in normalized:
            companies.add(fields.company)
    return {
        "normalized": normalized,
        "invoice_numbers": invoice_numbers,
        "account_numbers": account_numbers,
        "dates": query_dates,
        "names": names,
        "companies": companies,
    }


def field_match_score(question: str, fields: DocFields, signals: dict) -> float:
    score = 0.0
    if fields.invoice_number and fields.invoice_number in signals["invoice_numbers"]:
        score += 2.0
    if fields.account_number and fields.account_number in signals["account_numbers"]:
        score += 2.0

    name_norm = normalize_text(fields.name) if fields.name else None
    if name_norm and name_norm in signals["normalized"]:
        score += 0.8

    company_norm = normalize_text(fields.company) if fields.company else None
    if company_norm and company_norm in signals["normalized"]:
        score += 0.6

    matched_dates = set(fields.primary_dates) & signals["dates"]
    if matched_dates:
        score += min(0.8, 0.45 * len(matched_dates))

    hints = document_type_hints(question)
    if fields.doc_type in hints:
        score += 0.15
    return score


def document_type_hints(question: str) -> set[str]:
    q = question.casefold()
    hints = set()
    if "credit card" in q or "card bill" in q:
        hints.add("credit_card_bill")
    if "invoice" in q:
        hints.add("invoice")
    if "bank statement" in q or "statement period" in q:
        hints.add("bank_statement")
    if "utility" in q:
        hints.add("utility_bill")
    if "pharmacy" in q:
        hints.add("pharmacy_receipt")
    elif "receipt" in q:
        hints.add("receipt")
    return hints
