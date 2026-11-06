from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests


DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
MAX_DOC_CHARS = 2200
MAX_QUERY_CHARS = 600
MAX_JUSTIFICATION_CHARS = 240


class JudgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class JudgeResult:
    filename: str
    acceptable: str
    justification: str
    raw: str


def build_prompt(query: str, filename: str, document_text: str) -> str:
    query = query[:MAX_QUERY_CHARS]
    document_text = document_text[:MAX_DOC_CHARS]
    return f"""You are a strict retrieval relevance judge. Your task is NOT to answer the question.

Judge only these two points:
1. Is the document relevant to the question?
2. Does the document contain sufficient information to answer the question?

Do not quote, restate, calculate, or reveal any answer, amount, date, name, or other answer value. Refer to evidence only in general terms. Do not add outside facts.

Question:
{query}

Document filename: {filename}
Document text:
\"\"\"
{document_text}
\"\"\"

Return exactly one JSON object with these keys:
- "acceptable": one of "Yes", "No", "Uncertain"
- "justification": one short sentence, at most 25 words, about relevance and sufficiency only.

Use "Yes" only if the document is directly relevant and sufficient. Use "No" if unrelated. Use "Uncertain" if partially relevant or insufficient."""


def _extract_json(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise JudgeError("The local model returned a response that was not valid JSON.")


def _normalise_acceptable(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "acceptable"}:
        return "Yes"
    if text in {"no", "n", "false", "not acceptable"}:
        return "No"
    if text in {"uncertain", "maybe", "partial"}:
        return "Uncertain"
    raise JudgeError("The local model returned an invalid acceptable value.")


def check_ollama(
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    timeout: float = 5.0,
) -> tuple[bool, str]:
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
    except requests.ConnectionError:
        return False, f"Could not connect to Ollama at {host}."
    except requests.Timeout:
        return False, f"Ollama did not respond within {timeout:g} seconds."
    except requests.RequestException as exc:
        return False, f"Ollama request failed: {exc}"
    if response.status_code != 200:
        return False, f"Ollama returned HTTP {response.status_code}."
    try:
        names = [item.get("name", "") for item in response.json().get("models", [])]
    except (ValueError, AttributeError):
        return False, "Ollama returned an invalid model list."
    if model not in names:
        return False, f"Ollama is running, but model {model} is not available."
    return True, f"Ollama is available with {model}."


def judge_document(
    query: str,
    filename: str,
    document_text: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 180.0,
) -> JudgeResult:
    payload = {
        "model": model,
        "prompt": build_prompt(query, filename, document_text),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 120,
        },
    }
    try:
        response = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json=payload,
            timeout=(5.0, timeout),
        )
    except requests.ConnectionError as exc:
        raise JudgeError(f"Could not connect to Ollama at {host}.") from exc
    except requests.Timeout as exc:
        raise JudgeError(f"Ollama judgment timed out after {timeout:g} seconds.") from exc
    except requests.RequestException as exc:
        raise JudgeError(f"Ollama request failed: {exc}") from exc

    if response.status_code == 404:
        raise JudgeError(
            f"Model {model} was not found. Start Ollama and make sure the model is installed."
        )
    if response.status_code != 200:
        detail = response.text.strip().replace("\n", " ")[:300]
        raise JudgeError(f"Ollama returned HTTP {response.status_code}: {detail}")

    try:
        body = response.json()
    except ValueError as exc:
        raise JudgeError("Ollama returned a non-JSON API response.") from exc
    if body.get("error"):
        raise JudgeError(f"Ollama error: {body['error']}")

    raw = body.get("response", "")
    parsed = _extract_json(raw)
    acceptable = _normalise_acceptable(parsed.get("acceptable", ""))
    justification = " ".join(str(parsed.get("justification", "")).split())
    if not justification:
        raise JudgeError("The local model did not provide a justification.")
    if len(justification) > MAX_JUSTIFICATION_CHARS:
        justification = justification[: MAX_JUSTIFICATION_CHARS - 3].rstrip() + "..."
    return JudgeResult(
        filename=filename,
        acceptable=acceptable,
        justification=justification,
        raw=raw,
    )


def judge_documents(
    query: str,
    documents: list[dict],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 180.0,
) -> list[JudgeResult]:
    results = []
    for document in documents:
        results.append(
            judge_document(
                query=query,
                filename=document["filename"],
                document_text=document["text"],
                model=model,
                host=host,
                timeout=timeout,
            )
        )
    return results
