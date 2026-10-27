from __future__ import annotations

import html
import os

import httpx


LANGUAGES = {"zh-CN": "简体中文", "zh-TW": "繁體中文", "en": "English", "ja": "日本語", "ko": "한국어", "fr": "Français", "de": "Deutsch", "es": "Español", "ru": "Русский", "pt": "Português", "ar": "العربية"}
GOOGLE_TRANSLATION_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
GOOGLE_WEB_TRANSLATION_ENDPOINT = "https://translate.googleapis.com/translate_a/single"


def translation_status() -> dict:
    return {"provider": "Google", "mode": "api" if os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip() else "web", "languages": LANGUAGES}


async def _translate_with_client(
    client: httpx.AsyncClient, text: str, source: str, target: str
) -> dict:
    key = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip()
    if source != "auto" and source not in LANGUAGES or target not in LANGUAGES:
        raise ValueError("不支持的翻译语言")
    try:
        if key:
            payload = {"q": text, "target": target, "format": "text"}
            if source != "auto":
                payload["source"] = source
            response = await client.post(GOOGLE_TRANSLATION_ENDPOINT, headers={"X-Goog-Api-Key": key}, json=payload)
        else:
            response = await client.get(GOOGLE_WEB_TRANSLATION_ENDPOINT, params={
                "client": "gtx", "sl": source, "tl": target, "dt": "t", "q": text,
            })
        if response.status_code >= 400:
            # Never echo provider errors: credentials or input text may occur in them.
            raise RuntimeError(f"Google 翻译请求失败（HTTP {response.status_code}），请检查密钥、服务启用状态及配额")
        data = response.json()
        if key:
            item = data["data"]["translations"][0]
            return {"text": html.unescape(item["translatedText"]), "detected_source": item.get("detectedSourceLanguage")}
        translated = "".join(str(item[0]) for item in data[0] if item and item[0] is not None)
        return {"text": html.unescape(translated), "detected_source": data[2] if len(data) > 2 else None}
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("无法获取 Google 译文，请检查服务器网络或稍后重试") from exc


async def translate_text(
    text: str,
    source: str,
    target: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Translate text, optionally reusing a caller-owned connection pool."""
    if client is not None:
        return await _translate_with_client(client, text, source, target)
    async with httpx.AsyncClient(timeout=30) as temporary_client:
        return await _translate_with_client(temporary_client, text, source, target)
