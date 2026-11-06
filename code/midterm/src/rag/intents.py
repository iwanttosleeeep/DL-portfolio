from __future__ import annotations

import re

from .fields import DocFields

POLICY_INTENTS = {
    "late_payment": {
        "query": [
            (r"\blate fee\b", 4),
            (r"\blate payment", 4),
            (r"\boverdue\b", 4),
            (r"\bpast due\b", 4),
            (r"\bpenalt", 3),
            (r"\bgrace period\b", 2),
            (r"\bleeway\b", 3),
            (r"after (?:the )?due date", 2),
            (r"payment (?:arrives|received).*(?:after|late)", 2),
            (r"more than \d+ days", 3),
        ],
        "doc": [
            (r"\blate fee\b", 4),
            (r"\bthirty days\b", 2),
            (r"\bgrace period\b", 3),
            (r"\blate payments\b", 3),
            (r"payment is received more than", 2),
        ],
    },
    "refunds": {
        "query": [
            (r"\brefund", 4),
            (r"\breturn", 3),
            (r"send (?:it|them) back", 4),
            (r"\bunopened\b", 3),
            (r"\bsealed\b", 2),
            (r"\bdamaged\b", 3),
            (r"\bdefective\b", 3),
            (r"\bcracked\b", 4),
            (r"\bbroken\b", 3),
            (r"\bstore credit\b", 3),
            (r"\bexchange\b", 2),
            (r"\bprescription", 2),
            (r"\breturn window\b", 4),
        ],
        "doc": [
            (r"\brefund", 4),
            (r"\breturn", 3),
            (r"\bunopened\b", 3),
            (r"\bdamaged\b", 3),
            (r"\bdefective\b", 3),
            (r"\bstore credit\b", 3),
            (r"\bprescription medications\b", 3),
            (r"\bninety days\b", 3),
            (r"\bthirty days\b", 2),
        ],
    },
    "rewards": {
        "query": [
            (r"\brewards?\b", 4),
            (r"\bpoints?\b", 4),
            (r"\bearn\b", 2),
            (r"\bredeem\b", 3),
            (r"\bstatement credit\b", 3),
            (r"\bgas station", 2),
            (r"\brestaurant", 2),
            (r"\bswipe\b", 2),
            (r"\bexpire\b", 3),
            (r"lose their value", 3),
        ],
        "doc": [
            (r"\brewards\b", 4),
            (r"\bpoints\b", 4),
            (r"\bearn\b", 2),
            (r"\bredeem", 3),
            (r"\bgas stations\b", 3),
            (r"\brestaurants\b", 3),
            (r"\bstatement credit", 3),
            (r"\bexpire\b", 3),
        ],
    },
    "security": {
        "query": [
            (r"\bunauthori[sz]ed\b", 4),
            (r"\bfraud", 4),
            (r"\bstolen\b", 4),
            (r"someone used my card", 4),
            (r"without my permission", 4),
            (r"\bzero liability\b", 4),
            (r"\bliability\b", 2),
            (r"\bverification code", 4),
            (r"\blogin code\b", 4),
            (r"\bone-time code\b", 4),
            (r"\btext message\b", 2),
            (r"read out", 3),
            (r"two[- ]factor", 3),
            (r"responsible for those charges", 3),
            (r"reported within", 3),
        ],
        "doc": [
            (r"\bunauthorized\b", 4),
            (r"\bfraud\b", 4),
            (r"\bzero liability\b", 4),
            (r"\bverification codes\b", 4),
            (r"\btwo-factor\b", 3),
            (r"\breported within\b", 3),
            (r"\blogin\b", 2),
        ],
    },
    "payment": {
        "query": [
            (r"\bautopay\b", 5),
            (r"automatic payment", 4),
            (r"remembering to pay", 4),
            (r"worrying about remembering", 4),
            (r"\bcheque\b", 5),
            (r"\bcheck in the mail\b", 4),
            (r"\bpostmark", 4),
            (r"\bmail\b", 2),
            (r"\bonline portal\b", 3),
            (r"\bwebsite\b", 3),
            (r"ways? (?:to )?(?:pay|settle)", 4),
            (r"settle a bill", 4),
            (r"pay (?:my )?bill every month", 4),
            (r"missed deadline", 3),
            (r"by phone", 2),
            (r"by mail", 2),
        ],
        "doc": [
            (r"\bautopay\b", 5),
            (r"\bcheques?\b", 4),
            (r"\bpostmarked\b", 4),
            (r"pay your bill online", 3),
            (r"by phone", 2),
            (r"by mail", 2),
            (r"missed deadlines", 3),
        ],
    },
    "overdraft": {
        "query": [
            (r"\boverdraft\b", 5),
            (r"\bbelow zero\b", 5),
            (r"\bnot enough money\b", 5),
            (r"\binsufficient\b", 4),
            (r"\bshortfall\b", 4),
            (r"\bchecking balance\b", 4),
            (r"dip below", 4),
            (r"\bdeclined\b", 3),
            (r"\bbackup\b", 2),
            (r"\bsavings account\b", 2),
            (r"avoid being charged", 3),
            (r"\btransfer fee\b", 3),
        ],
        "doc": [
            (r"\boverdraft\b", 5),
            (r"\bbelow zero\b", 5),
            (r"\bshortfall\b", 4),
            (r"\bsavings account\b", 3),
            (r"\bdeclined\b", 3),
        ],
    },
    "privacy": {
        "query": [
            (r"\bprivacy\b", 5),
            (r"\bpersonal (?:data|information)\b", 5),
            (r"\bbanking details\b", 5),
            (r"\bshare.*(?:data|information|details)", 4),
            (r"hand over", 4),
            (r"\bsell\b", 4),
            (r"\bthird[- ]party\b", 4),
            (r"\bbusiness partners\b", 4),
            (r"\baffiliates\b", 3),
            (r"\bmarketing\b", 3),
            (r"\bconsent\b", 3),
            (r"without asking", 3),
            (r"\bopt out\b", 2),
            (r"for money", 2),
        ],
        "doc": [
            (r"\bpersonal information\b", 5),
            (r"\baffiliates\b", 4),
            (r"\bconsent\b", 4),
            (r"\bnever sold\b", 5),
            (r"\bthird[- ]party\b", 4),
            (r"\bmarketing\b", 3),
            (r"\bopt out\b", 3),
        ],
    },
    "billing_cycle": {
        "query": [
            (r"\bbilling cycle\b", 5),
            (r"\bbilling period\b", 5),
            (r"\bstatement closes?\b", 5),
            (r"\bclosing date\b", 4),
            (r"\binterest\b", 4),
            (r"cash advance", 5),
            (r"withdraw against", 5),
            (r"\badvance\b", 3),
            (r"statement produced", 4),
            (r"how often.*statement", 4),
            (r"interest starts?", 4),
            (r"pay.*in full", 2),
        ],
        "doc": [
            (r"\bbilling cycle\b", 5),
            (r"\bstatement closes\b", 4),
            (r"\bclosing date\b", 4),
            (r"\binterest\b", 4),
            (r"\bgrace period\b", 3),
            (r"\bcash advances\b", 5),
        ],
    },
    "support": {
        "query": [
            (r"\bsupport\b", 4),
            (r"\bcontact\b", 4),
            (r"\breal person\b", 5),
            (r"\brepresentative\b", 4),
            (r"\bcustomer service\b", 4),
            (r"\blive chat\b", 5),
            (r"\bchat\b", 4),
            (r"two in the morning", 5),
            (r"\b2 ?am\b", 4),
            (r"\burgent\b", 3),
            (r"\breach\b", 2),
            (r"\bphone\b", 2),
            (r"\bcall\b", 2),
            (r"twenty-four", 3),
            (r"24 hours", 3),
            (r"\bbranch", 3),
            (r"\bappointment", 3),
            (r"secure messaging", 3),
        ],
        "doc": [
            (r"\bphone support\b", 5),
            (r"\btwenty-four hours\b", 4),
            (r"\blive chat\b", 4),
            (r"\bsecure messaging\b", 3),
            (r"\bbranches\b", 3),
        ],
    },
    "fees": {
        "query": [
            (r"maintenance fee", 5),
            (r"\bwire transfer", 5),
            (r"\binternational", 4),
            (r"another country", 4),
            (r"move money", 3),
            (r"cashier'?s? check", 5),
            (r"out-of-network", 5),
            (r"\batm\b", 4),
            (r"cash machine", 4),
            (r"machine.*network", 4),
            (r"\bwithdrawal", 3),
            (r"\bprice\b", 3),
            (r"\bcost\b", 2),
            (r"avoid.*fee", 3),
            (r"\brelative", 2),
            (r"\bdomestic", 2),
        ],
        "doc": [
            (r"monthly maintenance fee", 5),
            (r"\bwire transfers\b", 5),
            (r"\bATM withdrawals\b", 5),
            (r"cashier'?s? check", 5),
            (r"\bfee schedule\b", 4),
            (r"out-of-network", 4),
        ],
    },
}

SUPERLATIVE_RE = re.compile(
    r"\b(largest|biggest|highest|smallest|lowest|least|cheapest|most|greatest)\b", re.I
)
GLOBAL_RE = re.compile(
    r"\b(in the dataset|across the dataset|in this dataset|dataset[- ]wide)\b", re.I
)
SUM_PATTERNS = [
    r"\badd up\b",
    r"\bcombined total\b",
    r"\ball together\b",
    r"\bsum of\b",
    r"\btotal of all\b",
    r"\bwhat do all\b",
    r"\bwhat would they owe\b",
    r"\bevery .* in the dataset\b",
    r"\ball (?:the )?(?:invoices|receipts|utility bills|bank statements)\b",
]
FILTER_PATTERNS = [
    r"\bwhich (?:purchases|receipts|invoices|bank statements|credit card bills|transactions)\b",
    r"\bhow many (?:bank statements|receipts|invoices|purchases|credit card bills)\b",
    r"\bpaid for with\b",
    r"\bsettled with\b",
    r"\brecord a\b",
]


def _pattern_score(patterns: list[tuple[str, int]], text: str) -> int:
    return sum(weight for pattern, weight in patterns if re.search(pattern, text, re.I))


def policy_query_scores(question: str) -> dict[str, int]:
    return {
        concept: _pattern_score(definition["query"], question)
        for concept, definition in POLICY_INTENTS.items()
    }


def best_policy_concept(question: str) -> tuple[str | None, int]:
    scores = policy_query_scores(question)
    concept = max(scores, key=scores.get)
    return concept, scores[concept]


def detect_category(question: str) -> str | None:
    q = question.casefold()
    if "utility" in q:
        return "utility_bill"
    if "invoice" in q:
        return "invoice"
    if "account holder" in q or "statement period" in q or "bank statement" in q:
        return "bank_statement"
    if "credit card" in q or "card bill" in q:
        return "credit_card_bill"
    if "pharmacy" in q:
        return "pharmacy_receipt"
    if "receipt" in q:
        return "receipt"
    if "purchase" in q:
        return "purchases"
    return None


def _has_structured_anchor(question: str, signals: dict) -> bool:
    if signals["invoice_numbers"] or signals["account_numbers"]:
        return True
    if signals["names"] or signals["companies"] or signals["dates"]:
        return True
    return False


def _is_global_extreme(question: str, signals: dict) -> bool:
    if not SUPERLATIVE_RE.search(question):
        return False
    category = detect_category(question)
    if GLOBAL_RE.search(question):
        return category is not None
    if question.strip().casefold().startswith("which") and category and not _has_structured_anchor(
        question, signals
    ):
        return True
    return False


def _is_aggregate_sum(question: str) -> bool:
    return any(re.search(pattern, question, re.I) for pattern in SUM_PATTERNS)


def _is_filtered_set(question: str) -> bool:
    return any(re.search(pattern, question, re.I) for pattern in FILTER_PATTERNS)


def _global_field(question: str, category: str) -> str | None:
    q = question.casefold()
    if category == "bank_statement":
        if "opening balance" in q:
            return "opening_balance"
        if "deposit" in q:
            return "largest_deposit"
        if "charge" in q:
            return "largest_charge"
        return "ending_balance"
    if category == "credit_card_bill":
        if "minimum payment" in q:
            return "minimum_payment"
        if "fine" in q:
            return "fine"
        if "total amount" in q:
            return "total"
        return "new_balance"
    if category == "invoice":
        if "subtotal" in q:
            return "subtotal"
        if "tax" in q:
            return "sales_tax"
        return "total_amount_due"
    if category == "utility_bill":
        if "amount due" in q:
            return "amount_due_now"
        return "total_current_charges"
    if category in {"receipt", "pharmacy_receipt", "purchases"}:
        return "total"
    return None


def _aggregate_field(category: str) -> str | None:
    return {
        "invoice": "total_amount_due",
        "utility_bill": "total_current_charges",
        "receipt": "total",
        "pharmacy_receipt": "total",
        "credit_card_bill": "new_balance",
        "bank_statement": "ending_balance",
    }.get(category)


def _filter_condition(question: str) -> tuple[str, str] | None:
    q = question.casefold()
    if re.search(r"\bcash\b", q):
        return "payment_method", "CASH"
    if re.search(r"\bdebit\b", q):
        return "payment_method", "DEBIT"
    if re.search(r"\bcredit card", q) and re.search(r"paid|settled|purchase", q):
        return "payment_method", "CREDIT"
    if re.search(r"\bservice fee\b", q) or (
        "service" in q and "fee" in q
    ):
        return "transaction_description", "service fee"
    return None


def _filter_category(question: str) -> str | None:
    category = detect_category(question)
    if category == "purchases":
        return "purchases"
    if category == "receipt":
        return "receipt" if "pharmacy" not in question.casefold() else "pharmacy_receipt"
    return category


def classify_intent(question: str, signals: dict) -> dict:
    category = detect_category(question)
    if _is_aggregate_sum(question) and category:
        return {
            "kind": "aggregate_sum",
            "category": category,
            "field": _aggregate_field(category),
        }
    if _is_global_extreme(question, signals):
        direction = "min" if re.search(r"\b(least|smallest|lowest|cheapest)\b", question, re.I) else "max"
        return {
            "kind": "global_extreme",
            "category": category,
            "field": _global_field(question, category or ""),
            "direction": direction,
        }
    if _is_filtered_set(question):
        condition = _filter_condition(question)
        filter_category = _filter_category(question)
        if condition and filter_category:
            return {
                "kind": "filtered_set",
                "category": filter_category,
                "condition": condition[0],
                "value": condition[1],
            }
    concept, score = best_policy_concept(question)
    if score > 0 and not _has_structured_anchor(question, signals):
        return {"kind": "policy", "concept": concept, "score": score}
    return {"kind": "single"}


def select_policy_document(
    question: str, concept: str | None, fields: list[DocFields], rank_hint: list[str]
) -> str | None:
    policy_fields = [item for item in fields if item.doc_type == "policy"]
    if not policy_fields:
        return None
    rank_position = {filename: index for index, filename in enumerate(rank_hint)}
    if not concept:
        return min(policy_fields, key=lambda item: rank_position.get(item.filename, 10_000)).filename

    evidence = {item.filename: _pattern_score(POLICY_INTENTS[concept]["doc"], item.text) for item in policy_fields}
    best_evidence = max(evidence.values())
    if best_evidence == 0:
        return min(policy_fields, key=lambda item: rank_position.get(item.filename, 10_000)).filename
    candidates = [item.filename for item in policy_fields if evidence[item.filename] == best_evidence]
    return min(candidates, key=lambda filename: rank_position.get(filename, 10_000))


def filter_documents(
    fields: list[DocFields], category: str, condition: str, value: str
) -> list[str]:
    if category == "purchases":
        allowed = {"receipt", "pharmacy_receipt"}
    else:
        allowed = {category}
    matches = []
    for item in fields:
        if item.doc_type not in allowed:
            continue
        if condition == "payment_method" and item.payment_method == value:
            matches.append(item.filename)
        elif condition == "transaction_description":
            if any(value.casefold() in row["description"].casefold() for row in item.transactions):
                matches.append(item.filename)
    return sorted(matches)


def category_documents(fields: list[DocFields], category: str) -> list[str]:
    if category == "purchases":
        allowed = {"receipt", "pharmacy_receipt"}
    else:
        allowed = {category}
    return sorted(item.filename for item in fields if item.doc_type in allowed)
