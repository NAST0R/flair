"""Potatura deterministica degli output di tool SUPERATI (stadio 0 della compaction).

Tra una compaction e l'altra gli output dei tool restano integri in contesto: sono
quasi tutti cache-hit (economici in $) ma occupano la finestra e anticipano la
compaction — che costa una chiamata LLM e, soprattutto, sostituisce il dettaglio
con un riassunto. Prima di arrivare a quel punto, qui si recupera spazio GRATIS
sostituendo con uno stub i soli output **provabilmente superati**:

1. duplicati — la più vecchia di due chiamate IDENTICHE (stesso tool di lettura,
   stessi argomenti): il modello ha già la versione più recente più avanti;
2. letture rese stale — un `read_file` di un path poi **sovrascritto** da un
   `write_file` senza append: l'intero contenuto letto non esiste più su disco;
3. letture parziali coperte — un `read_file` con offset/limit dello stesso path
   riletto più avanti PER INTERO: la lettura completa contiene la parziale;
4. immagini ri-allegate — l'allegato di `view_image` per gli STESSI file quando più
   avanti gli stessi file sono stati allegati di nuovo (tipico del monitoraggio a
   screenshot ripetuti): l'immagine vecchia è la cosa più pesante che il contesto
   possa portarsi dietro, e la sua versione aggiornata è già più avanti.

Garanzie:
- si sostituisce SOLO il campo `content` dei messaggi `tool` (il pairing
  tool_call/tool resta intatto: la conversazione rimane valida per l'API);
- sopravvive sempre l'occorrenza più recente di ogni informazione;
- `edit_file`/`multi_edit`/append NON invalidano le letture precedenti (gran parte
  del contenuto letto resta valido): per prudenza non si pota in quei casi;
- mai potato l'output di un tool non in lista, né uno già piccolo (lo stub non
  farebbe risparmiare nulla).

Il modulo è puro (opera su una lista di messaggi in formato OpenAI): chi chiama
(l'Agent) gestisce il reset dei contatori di contesto, perché la prima mutazione
spezza il prefisso in cache da quel punto in poi — è il momento giusto per farlo,
visto che l'alternativa (compaction) lo spezzerebbe comunque.
"""

from __future__ import annotations

import json

# Tool di sola lettura i cui output possono essere superati da chiamate successive.
_PRUNABLE = {
    "read_file", "list_directory", "glob", "grep", "repo_map",
    "search_files", "web_fetch", "web_search",
}

# Sotto questa dimensione lo stub non ripaga (e i messaggi d'errore restano leggibili).
_MIN_CHARS = 200

STUB = ("[superseded output: the same target was re-read or rewritten later — "
        "refer to the most recent version in the conversation]")

# Marcatore del messaggio con cui il loop consegna le immagini (lo compone
# core/agent.py, che lo importa da qui: la potatura deve poterlo riconoscere e
# importare `agent` da `prune` sarebbe un ciclo).
ATTACHED_IMAGES_PREFIX = "[Images attached by view_image: "
IMAGE_STUB = ("[superseded image: the same file was attached again later — "
              "refer to the most recent attachment]")


def _norm_path(p) -> str | None:
    if not isinstance(p, str) or not p.strip():
        return None
    q = p.replace("\\", "/").strip()
    while q.startswith("./"):
        q = q[2:]
    return q or None


def _is_full_read(args: dict) -> bool:
    """True se la read_file copre l'intero file (offset assente/1, limit assente)."""
    off = args.get("offset", 1)
    lim = args.get("limit")
    try:
        off = int(off)
    except (TypeError, ValueError):
        return False
    return off <= 1 and lim in (None, "", 0)


def _sig(name: str, args: dict | None, raw: str) -> str:
    if args is None:
        return name + "|" + raw
    try:
        return name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return name + "|" + raw


def prune_superseded(messages: list[dict]) -> int:
    """Sostituisce con STUB il contenuto degli output di tool provabilmente superati.
    Muta `messages` in place e restituisce il numero di output potati (0 = no-op).
    Non solleva mai: su qualunque struttura inattesa, semplicemente non pota."""
    # 1) Indice id → (nome, argomenti) dalle tool_call dei messaggi assistant.
    calls: dict[str, tuple[str, dict | None, str]] = {}
    for m in messages:
        for tcd in (m.get("tool_calls") or []):
            fn = tcd.get("function") or {}
            name, raw = fn.get("name") or "", fn.get("arguments") or ""
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = None
            except (json.JSONDecodeError, TypeError, ValueError):
                args = None
            if tcd.get("id"):
                calls[tcd["id"]] = (name, args, raw)

    # 2) Tutte le occorrenze di output di tool, in ordine.
    entries: list[tuple[int, str, dict | None, str]] = []  # (idx_msg, nome, args, sig)
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        info = calls.get(m.get("tool_call_id") or "")
        if not info:
            continue
        name, args, raw = info
        entries.append((i, name, args, _sig(name, args, raw)))

    to_prune: set[int] = set()

    # Regola 1 — duplicati: pota tutte le occorrenze TRANNE l'ultima della stessa firma.
    last_by_sig: dict[str, int] = {
        sig: idx for idx, name, _args, sig in entries if name in _PRUNABLE
    }
    for idx, name, _args, sig in entries:
        if name in _PRUNABLE and idx != last_by_sig.get(sig):
            to_prune.add(idx)

    # Regola 2 — sovrascrittura: read_file di un path poi riscritto per intero.
    last_overwrite: dict[str, int] = {}
    for idx, name, args, _ in entries:
        if name == "write_file" and args is not None:
            path = _norm_path(args.get("path"))
            append = str(args.get("append", False)).strip().lower() in {"1", "true", "yes", "si", "sì", "y", "on"}
            if path and not append:
                last_overwrite[path] = idx
    # Regola 3 — copertura: read_file parziale superata da una lettura INTERA successiva.
    last_full_read: dict[str, int] = {}
    for idx, name, args, _ in entries:
        if name == "read_file" and args is not None and _is_full_read(args):
            path = _norm_path(args.get("path"))
            if path:
                last_full_read[path] = idx
    for idx, name, args, _ in entries:
        if name != "read_file" or args is None:
            continue
        path = _norm_path(args.get("path"))
        if not path:
            continue
        if idx < last_overwrite.get(path, -1) or idx < last_full_read.get(path, -1):
            to_prune.add(idx)

    # 3) Applica gli stub (solo content; pairing e conteggio messaggi invariati).
    pruned = 0
    for idx in to_prune:
        m = messages[idx]
        content = m.get("content") or ""
        if not isinstance(content, str) or len(content) <= _MIN_CHARS or content == STUB:
            continue
        m["content"] = STUB
        pruned += 1
    return pruned + _prune_superseded_images(messages)


def _attached_labels(content) -> str | None:
    """Le etichette dei file di un messaggio-allegato prodotto dal loop, o None se
    il messaggio non è uno di quelli (identificato dal marcatore nella parte testo)."""
    if not isinstance(content, list):
        return None
    has_image = any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
    if not has_image:
        return None
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text") or ""
            if text.startswith(ATTACHED_IMAGES_PREFIX):
                return text
    return None


def _prune_superseded_images(messages: list[dict]) -> int:
    """Regola 4: se gli STESSI file sono stati allegati più volte, tiene solo
    l'allegato più recente e sostituisce le parti immagine precedenti con uno stub
    testuale. Provabilmente superato — è la stessa logica della regola 1 applicata
    al canale multipart — e ad alto rendimento: un'immagine pesa ~1-2K token e viene
    ri-caricata a ogni richiesta successiva finché resta in contesto. Gli allegati
    dell'utente (/img, senza marcatore) NON si toccano: non c'è modo di provare che
    siano superati, e sono una scelta esplicita di chi lavora."""
    last_by_labels: dict[str, int] = {}
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        labels = _attached_labels(m.get("content"))
        if labels is not None:
            last_by_labels[labels] = i

    pruned = 0
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        labels = _attached_labels(content)
        if labels is None or not isinstance(content, list) or last_by_labels.get(labels) == i:
            continue
        m["content"] = [
            {"type": "text", "text": IMAGE_STUB}
            if isinstance(p, dict) and p.get("type") == "image_url" else p
            for p in content
        ]
        pruned += 1
    return pruned
