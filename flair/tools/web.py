"""Ricerca web per l'agente generico.

Backend, in ordine di preferenza (il primo che restituisce risultati vince):
1. Tavily — se è impostata TAVILY_API_KEY. API JSON robusta, consigliata.
2. ddgs — se la libreria è installata (`pip install ddgs`). Metaricerca SENZA
   chiave che gestisce l'anti-bot di DuckDuckGo/altri motori: è il modo no-key
   più affidabile.
3. Scraping stdlib (DuckDuckGo lite/html) — best-effort senza dipendenze. Spesso
   bloccato dalle protezioni anti-bot (Cloudflare): può non restituire nulla.
4. DuckDuckGo Instant Answer API (JSON, senza chiave) — solo abstract e voci
   correlate: utile per definizioni, non per le notizie.

Non solleva mai eccezioni verso l'agente: se nessun backend funziona, restituisce
un messaggio chiaro con la causa probabile e come rendere la ricerca affidabile.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request

from ..core.tool import ToolContext, tool

# ddgs (ex duckduckgo-search): import opzionale, due nomi possibili.
try:
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception:
        DDGS = None  # type: ignore

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
_REGION = "it-it"

_A_TAG = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_HREF = re.compile(r"""href=['"]([^'"]+)['"]""", re.IGNORECASE)
_SNIP_HTML = re.compile(
    r"""<a\b[^>]*class=['"][^'"]*result__snippet[^'"]*['"][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL)
_SNIP_LITE = re.compile(
    r"""<td\b[^>]*class=['"][^'"]*result-snippet[^'"]*['"][^>]*>(.*?)</td>""",
    re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return html.unescape(_TAGS.sub("", text)).strip()


def _unwrap(url: str) -> str:
    # I link DuckDuckGo sono spesso redirect: /l/?uddg=<encoded>
    if "uddg=" in url:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "uddg" in params:
            return params["uddg"][0]
    if url.startswith("//"):
        return "https:" + url
    return url


def _http(url: str, data: dict | None, timeout: int) -> str:
    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    }
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _links(page: str, link_class: str) -> list[tuple[str, str]]:
    out = []
    for attrs, inner in _A_TAG.findall(page):
        if link_class in attrs:
            m = _HREF.search(attrs)
            if m:
                out.append((_unwrap(m.group(1)), _clean(inner)))
    return out


# ── backend ───────────────────────────────────────────────────────────────────

def _search_tavily(api_key: str, query: str, k: int, timeout: int) -> list[dict]:
    payload = json.dumps({"api_key": api_key, "query": query,
                          "max_results": k, "search_depth": "basic"}).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search", data=payload,
        headers={"Content-Type": "application/json", "User-Agent": _UA}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return [{"title": i.get("title", ""), "url": i.get("url", ""),
             "snippet": (i.get("content") or "")[:300]} for i in data.get("results", [])[:k]]


def _search_ddgs(query: str, k: int) -> list[dict]:
    with DDGS() as ddgs:  # type: ignore
        rows = list(ddgs.text(query, region=_REGION, safesearch="moderate", max_results=k))
    return [{"title": r.get("title", ""), "url": r.get("href", ""),
             "snippet": (r.get("body") or "")[:300]} for r in rows[:k]]


def _search_scrape(query: str, k: int, timeout: int) -> list[dict]:
    attempts = [
        ("https://lite.duckduckgo.com/lite/", "result-link", _SNIP_LITE),
        ("https://html.duckduckgo.com/html/", "result__a", _SNIP_HTML),
    ]
    for url, link_class, snip_rx in attempts:
        try:
            page = _http(url, {"q": query, "kl": _REGION}, timeout)
        except Exception:
            continue
        links = _links(page, link_class)
        if not links:
            continue
        snippets = [_clean(s) for s in snip_rx.findall(page)]
        out = []
        for i, (link, title) in enumerate(links[:k]):
            out.append({"title": title, "url": link,
                        "snippet": snippets[i] if i < len(snippets) else ""})
        if out:
            return out
    return []


def _search_instant(query: str, k: int, timeout: int) -> list[dict]:
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
    page = _http(url, None, timeout)
    data = json.loads(page)
    out: list[dict] = []
    if data.get("AbstractText"):
        out.append({"title": data.get("Heading") or query,
                    "url": data.get("AbstractURL", ""), "snippet": data["AbstractText"]})
    for topic in data.get("RelatedTopics", []):
        if len(out) >= k:
            break
        if topic.get("Text") and topic.get("FirstURL"):
            out.append({"title": topic["Text"][:80], "url": topic["FirstURL"],
                        "snippet": topic["Text"]})
    return out[:k]


def _format(query: str, backend: str, results: list[dict]) -> str:
    lines = [f"Risultati per '{query}' ({backend}):"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}"
                     + (f"\n   {r['snippet']}" if r["snippet"] else ""))
    return "\n".join(lines)


@tool(
    "web_search",
    ("Cerca informazioni sul web e restituisce i risultati principali (titolo, URL, "
     "estratto). Usalo per domande su fatti attuali, notizie, riferimenti. Dopo, puoi "
     "aprire un risultato con open_url."),
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "La query di ricerca."},
            "max_results": {"type": "integer", "description": "Quanti risultati (default da configurazione)."},
        },
        "required": ["query"],
    },
)
def web_search(ctx: ToolContext, query: str, max_results: int | None = None) -> str:
    k = max_results or ctx.cfg.web_max_results
    timeout = min(ctx.cfg.request_timeout, 25)
    errors: list[str] = []

    backends = []
    if ctx.cfg.tavily_api_key:
        backends.append(("Tavily", lambda: _search_tavily(ctx.cfg.tavily_api_key, query, k, timeout)))
    if DDGS is not None:
        backends.append(("DuckDuckGo (ddgs)", lambda: _search_ddgs(query, k)))
    backends.append(("DuckDuckGo (scraping)", lambda: _search_scrape(query, k, timeout)))
    backends.append(("DuckDuckGo Instant Answer", lambda: _search_instant(query, k, timeout)))

    for name, fn in backends:
        try:
            results = fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}")
            continue
        if results:
            return _format(query, name, results)

    # Nessun backend ha prodotto risultati: spiega causa e rimedio.
    hints = []
    if DDGS is None:
        hints.append("installa la ricerca senza chiave con `pip install ddgs`")
    if not ctx.cfg.tavily_api_key:
        hints.append("imposta TAVILY_API_KEY per una ricerca affidabile")
    rimedio = (" Per risolvere: " + "; oppure ".join(hints) + ".") if hints else ""
    dettagli = (" [" + ", ".join(errors) + "]") if errors else ""
    return (f"❌ Nessun risultato per '{query}'. I motori senza chiave possono limitare o "
            f"bloccare le richieste automatiche (anti-bot)." + rimedio + dettagli)


_SCRIPT_STYLE = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n\s*\n\s*\n+")


def _html_to_text(page: str) -> str:
    page = _SCRIPT_STYLE.sub(" ", page)
    page = re.sub(r"<(br|/p|/div|/li|/h[1-6])\b[^>]*>", "\n", page, flags=re.IGNORECASE)
    text = html.unescape(_TAGS.sub("", page))
    text = _WS.sub(" ", text)
    return _BLANKS.sub("\n\n", text).strip()


@tool(
    "web_fetch",
    ("Scarica una pagina web e ne restituisce il testo leggibile (HTML rimosso). "
     "Usalo dopo web_search per leggere il contenuto di un risultato, o su un URL noto."),
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "L'URL da scaricare (http/https)."},
            "max_chars": {"type": "integer", "description": "Lunghezza massima del testo (default: limite di lettura)."},
        },
        "required": ["url"],
    },
)
def web_fetch(ctx: ToolContext, url: str, max_chars: int | None = None) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    limit = max_chars or ctx.cfg.read_file_max_chars
    timeout = min(ctx.cfg.request_timeout, 25)
    try:
        page = _http(url, None, timeout)
    except Exception as exc:  # noqa: BLE001
        return f"❌ Impossibile scaricare {url}: {type(exc).__name__}: {exc}"
    text = _html_to_text(page)
    if not text:
        return f"(nessun testo estraibile da {url})"
    if len(text) > limit:
        text = text[:limit] + f"\n…[troncato a {limit} caratteri]"
    return f"Contenuto di {url}:\n\n{text}"


TOOLS = [web_search, web_fetch]
