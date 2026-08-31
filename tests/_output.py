"""Output a prova di encoding per gli script di test/eval.

Problema reale, visto in CI: su Windows uno `stdout` NON interattivo usa il code
page locale (cp1252 negli Actions), che non sa codificare i simboli con cui questi
script scrivono a video (`✓`, `─`, `✅`, em dash). Il `print` solleva
UnicodeEncodeError e lo script muore — a volte prima di eseguire un solo test,
altre volte a metà tabella, sempre senza dire nulla di utile. In locale non si vede
perché la console è in UTF-8.

Questo modulo esiste perché la protezione stava dentro UN file: l'altro entry point
(il runner degli eval) è morto allo stesso modo il giorno dopo. Ora la si importa,
non la si riscrive.

Strategia, in ordine:
1. si porta lo stream a UTF-8 con `errors="replace"` (i log di CI lo gestiscono, e
   i simboli si vedono correttamente);
2. se la riconfigurazione dell'encoding non è possibile, si imposta ALMENO
   `errors="replace"`: i simboli diventano '?', ma nessuno script muore più per un
   carattere;
3. si ritornano i marcatori adatti allo stream risultante, così chi vuole può
   degradare a ASCII invece di stampare punti di domanda.
"""

from __future__ import annotations

import sys


def prepare_output() -> tuple[str, str]:
    """Prepara stdout/stderr e ritorna (marcatore_ok, suffisso_finale) adatti
    all'encoding effettivo. Idempotente: chiamarla più volte è innocuo."""
    for stream in (sys.stdout, sys.stderr):
        for kwargs in ({"encoding": "utf-8", "errors": "replace"}, {"errors": "replace"}):
            try:
                stream.reconfigure(**kwargs)   # type: ignore[union-attr]
                break
            except (AttributeError, ValueError, OSError):
                continue                       # stream sostituito o non riconfigurabile
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✓ ✅ ─ —".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return "[ok]", "PASSATI"
    return "✓", "PASSATI ✅"
