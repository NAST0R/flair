"""Immagini per gli endpoint con vision (content multipart OpenAI-style).

Il canale dei tool è SOLO TESTO per protocollo (i messaggi role:"tool" non
portano parti immagine in modo portabile): il tool `view_image` quindi valida e
codifica l'immagine, la DEPOSITA nella staging area del ToolContext e risponde
con una conferma testuale; è il loop dell'agente (core/agent.py,
`_flush_pending_images`) ad accodare subito dopo UN messaggio utente multipart
con le immagini vere — append-only, stesso pattern di nudge e inventario. Al
passo successivo il modello se le trova nel contesto e le vede.

Le validazioni sono deliberatamente PRE-invio (a differenza delle web UI che
spediscono e lasciano esplodere il server): estensione, dimensione, firma reale
del contenuto (magic bytes, stdlib puro) e — con Pillow presente —
decodificabilità vera. Un file corrotto o travestito viene rifiutato con un
errore azionabile PRIMA che un solo byte entri nella conversazione append-only.

Il ridimensionamento pre-invio è opzionale e feature-detected: con Pillow
installato le immagini oltre `cfg.image_max_side` vengono ridotte (meno tempo
di encoding lato server, meno token di contesto, payload più piccolo); senza,
si invia l'originale com'è. Nessuna dipendenza nuova obbligatoria.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from ..core.tool import ToolContext, ToolError
from . import fs

# Formati accettati = quelli decodificabili da llama.cpp (stb) e dagli endpoint
# OpenAI-compatibili. L'estensione dichiara l'intento; la firma reale (sotto)
# deve confermarlo.
_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
# Tetto sul file ORIGINALE: oltre, meglio un errore azionabile che un payload
# da decine di MB in conversazione (che poi resterebbe nel prefisso a ogni turno).
_MAX_BYTES = 8_000_000


def _sniff(data: bytes) -> str | None:
    """MIME dalla firma reale del contenuto (magic bytes, stdlib puro).
    None = nessuna firma d'immagine riconosciuta."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _downscale(data: bytes, suffix: str, max_side: int) -> tuple[bytes, str, str]:
    """Riduce l'immagine a `max_side` px sul lato lungo, SE Pillow è installato e
    serve. Ritorna (bytes, mime, nota): senza Pillow o senza necessità, l'originale
    passa intatto e la nota lo dice. JPEG resta JPEG (foto: pesa meno), il resto
    diventa PNG (lossless, gestisce l'alpha). Con Pillow presente, un file che non
    si DECODIFICA è corrotto: rifiuto pre-invio, non pass-through — un'immagine
    avvelenata in conversazione costa più di un errore chiaro adesso."""
    mime = _MIME[suffix]
    try:
        from PIL import Image  # opzionale: feature-detect, mai richiesto
    except ImportError:
        return data, mime, ""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ToolError(f"Cannot decode the image (corrupted file?): {exc}") from exc
    w, h = img.size
    if max(w, h) <= max_side or max_side <= 0:
        return data, mime, f" ({w}x{h})"
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    if suffix in (".jpg", ".jpeg"):
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return buf.getvalue(), mime, f" ({w}x{h} → {img.size[0]}x{img.size[1]})"


def load_image_part(cfg, path: Path) -> tuple[dict, str]:
    """(parte image_url OpenAI-style, nota descrittiva) per il file dato.
    Solleva ToolError con messaggi azionabili su formato/dimensione/contenuto."""
    suffix = path.suffix.lower()
    if suffix not in _MIME:
        raise ToolError(
            f"Unsupported image format '{suffix or path.name}': "
            f"use one of {', '.join(sorted(_MIME))}."
        )
    if not path.is_file():
        raise ToolError(f"Image not found: {path}")
    data = path.read_bytes()
    if len(data) > _MAX_BYTES:
        raise ToolError(
            f"Image too large ({len(data) // 1_000_000} MB > {_MAX_BYTES // 1_000_000} MB): "
            "resize it first."
        )
    sniffed = _sniff(data)
    if sniffed != _MIME[suffix]:
        raise ToolError(
            f"'{path.name}' does not look like a valid {suffix} image "
            f"(content signature: {sniffed or 'unknown'}): the file is corrupted or misnamed."
        )
    data, mime, note = _downscale(data, suffix, getattr(cfg, "image_max_side", 1536))
    b64 = base64.b64encode(data).decode("ascii")
    part = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    return part, f"{path.name}{note}, ~{len(data) // 1024} KB"


def view_image_impl(ctx: ToolContext, path: str, root: Path | None) -> str:
    """Corpo condiviso di `view_image` (coding: confinato alla radice; general:
    tutta la macchina — stesso pattern di read_file). Gate a runtime sul flag
    vision dello slot ATTIVO: così vale anche dopo un /provider a metà sessione."""
    if not getattr(ctx.cfg.active, "vision", False):
        return (
            "❌ The current provider/endpoint has no vision support. Enable it only "
            "if the endpoint really accepts images (e.g. llama-server with --mmproj): "
            "set DEEPSEEK_VISION / OPENAI_VISION / LOCAL_VISION=true in .env."
        )
    full = fs.resolve(root, path)
    part, note = load_image_part(ctx.cfg, full)
    if ctx.pending_images is None:
        ctx.pending_images = []
    ctx.pending_images.append({"part": part, "label": fs.display(root, full)})
    return (
        f"✅ Image loaded: {note}. It is attached to the conversation right after "
        "this result: you will SEE it at your next step — describe or use what is "
        "actually in the image, do not guess."
    )
