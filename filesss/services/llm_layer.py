import os

import requests


LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
}


def _llm_chat_completion(messages, temperature=0.3):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def translate_advisory(advisory_text, language_code):
    if language_code == "en":
        return advisory_text

    language_name = LANGUAGE_MAP.get(language_code, "English")
    prompt = (
        f"Translate the following agricultural advisory into simple {language_name} suitable for a farmer. "
        "Use clear and practical language.\n\n"
        f"{advisory_text}"
    )

    translated = _llm_chat_completion(
        [
            {"role": "system", "content": "You are an agricultural language assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    if translated:
        return translated

    return (
        f"[{language_name} translation unavailable: configure OPENAI_API_KEY.]\n\n"
        f"{advisory_text}"
    )


def advisory_chat_reply(context_payload, user_question, language_code):
    language_name = LANGUAGE_MAP.get(language_code, "English")
    context_text = (
        f"Disease: {context_payload.get('condition')}\n"
        f"Severity: {context_payload.get('severity_level')}\n"
        f"Environmental Risk: {context_payload.get('risk_level')} ({context_payload.get('risk_score')})\n"
        f"Chemicals: {', '.join(context_payload.get('input_chemicals', [])) or 'Not provided'}\n"
        f"Advisory: {context_payload.get('summary_text', '')}"
    )

    prompt = (
        "You are an agricultural advisor. Based on the following analysis, answer the farmer's question clearly and practically.\n\n"
        f"Analysis:\n{context_text}\n\n"
        f"Language: {language_name}\n"
        f"Question: {user_question}"
    )

    reply = _llm_chat_completion(
        [
            {"role": "system", "content": "You are a practical farm advisory assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    if reply:
        return reply

    return (
        "AI chat is currently unavailable because OPENAI_API_KEY is not configured. "
        "Please set the API key to enable multilingual follow-up guidance."
    )
