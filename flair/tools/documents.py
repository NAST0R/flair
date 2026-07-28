"""Estrazione testo dai formati documento più diffusi — SOLO stdlib.

Il criterio del "fattibile senza dipendenze": DOCX/XLSX/PPTX/ODT sono ZIP+XML e
si estraggono con `zipfile` + ElementTree a qualità piena; il PDF ha un
estrattore best-effort (gli stream FlateDecode si aprono con `zlib`, gli
operatori di testo Tj/TJ si leggono direttamente) protetto da un CANCELLO DI
QUALITÀ: i PDF scansionati, cifrati o a encoding CID producono spazzatura, e
in quel caso si restituisce un errore onesto invece di darla in pasto al
modello. L'estrazione ALIMENTA la pipeline di read_file (offset/limit/budget/
header onesto): qui si produce solo il testo.
"""

from __future__ import annotations

import re
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree

from ..core.tool import ToolError

DOCUMENT_EXTS = {".docx", ".xlsx", ".pptx", ".odt", ".pdf"}


def extract_text(p: Path) -> tuple[str, str]:
    """Ritorna (testo, etichetta formato). Solleva ToolError con messaggio
    azionabile quando l'estrazione affidabile non è possibile."""
    ext = p.suffix.lower()
    try:
        if ext == ".docx":
            return _from_docx(p), "DOCX"
        if ext == ".odt":
            return _from_odt(p), "ODT"
        if ext == ".pptx":
            return _from_pptx(p), "PPTX"
        if ext == ".xlsx":
            return _from_xlsx(p), "XLSX"
        if ext == ".pdf":
            return _from_pdf(p), "PDF"
    except ToolError:
        raise
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ToolError(
            f"Cannot extract text from {p.name}: not a valid {ext[1:].upper()} "
            f"or an unsupported variant ({type(exc).__name__})."
        ) from exc
    raise ToolError(f"Unsupported document format: {ext}")


def _tag(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _from_docx(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    out: list[str] = []
    for para in root.iter():
        if _tag(para) == "p":
            line = "".join(t.text or "" for t in para.iter() if _tag(t) == "t")
            out.append(line)
    return "\n".join(out).strip() or _empty(p)


def _from_odt(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        root = ElementTree.fromstring(z.read("content.xml"))
    out = ["".join(el.itertext()) for el in root.iter() if _tag(el) in ("p", "h")]
    return "\n".join(out).strip() or _empty(p)


def _from_pptx(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        slides = sorted(
            (n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),   # type: ignore[union-attr]
        )
        out: list[str] = []
        for i, name in enumerate(slides, 1):
            root = ElementTree.fromstring(z.read(name))
            texts = [t.text or "" for t in root.iter() if _tag(t) == "t"]
            out.append(f"── slide {i} ──")
            out.extend(x for x in texts if x.strip())
    return "\n".join(out).strip() or _empty(p)


def _from_xlsx(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sroot = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(si.itertext()) for si in sroot if _tag(si) == "si"]
        names: list[str] = []
        if "xl/workbook.xml" in z.namelist():
            wb = ElementTree.fromstring(z.read("xl/workbook.xml"))
            names = [s.get("name", "") for s in wb.iter() if _tag(s) == "sheet"]
        sheets = sorted(
            (n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),   # type: ignore[union-attr]
        )
        out: list[str] = []
        for i, name in enumerate(sheets):
            label = names[i] if i < len(names) else f"sheet {i + 1}"
            out.append(f"── {label} ──")
            root = ElementTree.fromstring(z.read(name))
            for row in root.iter():
                if _tag(row) != "row":
                    continue
                cells: list[str] = []
                for c in row:
                    v = next((el for el in c.iter() if _tag(el) in ("v", "t")), None)
                    val = (v.text or "") if v is not None else ""
                    if c.get("t") == "s" and val.isdigit() and int(val) < len(shared):
                        val = shared[int(val)]
                    cells.append(val)
                if any(x.strip() for x in cells):
                    out.append(" | ".join(x.strip() for x in cells).rstrip(" |"))
    return "\n".join(out).strip() or _empty(p)


# ── PDF best-effort ───────────────────────────────────────────────────────────

_PDF_TOKEN = re.compile(
    r"\((?:\\.|[^\\()])*\)"      # stringa letterale (con escape)
    r"|<[0-9A-Fa-f\s]+>"         # stringa esadecimale
    r"|T[dD*]|ET|Tj|TJ"          # operatori di posizionamento/uscita testo
)
_PDF_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f",
                "(": "(", ")": ")", "\\": "\\"}


def _pdf_unescape(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(s):
            break
        nxt = s[i]
        if nxt in _PDF_ESCAPES:
            out.append(_PDF_ESCAPES[nxt])
            i += 1
        elif nxt.isdigit():                     # ottale \d{1,3}
            j = i
            while j < len(s) and j - i < 3 and s[j].isdigit():
                j += 1
            out.append(chr(int(s[i:j], 8) & 0xFF))
            i = j
        elif nxt == "\n":                       # continuazione di riga
            i += 1
        else:
            out.append(nxt)
            i += 1
    return "".join(out)


def _from_pdf(p: Path) -> str:
    raw = p.read_bytes()
    if b"/Encrypt" in raw:
        raise ToolError(f"{p.name} is an encrypted PDF: decrypt it (or export to text) first.")
    pieces: list[str] = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.DOTALL):
        data = m.group(1)
        try:
            data = zlib.decompress(data)
        except zlib.error:
            pass                                # stream non compresso (o non Flate)
        content = data.decode("latin-1", errors="replace")
        if "BT" not in content:
            continue
        buf: list[str] = []
        for tok in _PDF_TOKEN.finditer(content):
            t = tok.group(0)
            if t.startswith("("):
                buf.append(_pdf_unescape(t[1:-1]))
            elif t.startswith("<"):
                hexs = re.sub(r"\s", "", t[1:-1])
                if len(hexs) % 2:
                    hexs += "0"
                try:
                    buf.append(bytes.fromhex(hexs).decode("latin-1"))
                except ValueError:
                    pass
            elif t in ("Td", "TD", "T*", "ET"):
                buf.append("\n")
        piece = "".join(buf)
        if piece.strip():
            pieces.append(piece)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(pieces)).strip()
    _pdf_quality_gate(p, text)
    return text


def _pdf_quality_gate(p: Path, text: str) -> None:
    """I PDF scansionati o a font CID producono stringhe di glifi, non testo:
    meglio un errore onesto che spazzatura plausibile nel contesto del modello."""
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)
    good = sum(ch.isalnum() or ch.isspace() or ch in ".,;:!?'\"()[]{}|/\\-–—_*#@%&+=<>~`^$" for ch in text)
    if len(text) < 20 or not words or len(words) < 3 or good / max(1, len(text)) < 0.75:
        raise ToolError(
            f"Cannot reliably extract text from {p.name}: it looks scanned, image-only "
            "or CID-encoded (beyond the built-in extractor). Convert it to text/markdown "
            "first (e.g. with a PDF tool), then read the converted file."
        )


def _empty(p: Path) -> str:
    raise ToolError(f"No extractable text found in {p.name} (empty or unsupported layout).")
