"""Provider per inference LOCALE OpenAI-compatibile (llama.cpp llama-server,
vLLM, LM Studio, ...).

Differenze rispetto alla base, pensate per llama-server:
- nessuna API key richiesta (default "local": il server non la verifica);
- prezzi a zero: i costi mostrati sono onesti ($0.0000), non stime da listino;
- `temperature` NON inviata di default: comandano i sampling del server
  (--temp/--top-p/--top-k/--min-p), che per Qwen3.6 sono quelli raccomandati.
  Inviare il default 0.0 li sovrascriverebbe col greedy — e il greedy coi
  modelli thinking Qwen produce loop di ripetizione. LOCAL_TEMPERATURE la
  invia esplicitamente se impostata (>= 0).
- `keeps_reasoning_history` resta False (default base): coerente con
  `preserve_thinking:false` nel template — il thinking passato non torna.
Il reasoning arriva come `reasoning_content` (llama-server con --jinja lo
estrae dai blocchi <think> di default): pannello e spinner funzionano già.
"""

from __future__ import annotations

from .base import OpenAICompatProvider


class LocalProvider(OpenAICompatProvider):
    name = "local"
    token_param = "max_tokens"

    def _apply_reasoning(self, params: dict, model: str, think: bool) -> None:
        pc = self.cfg.active
        if pc.temperature >= 0:   # sentinella: -1.0 = non inviare, vince il server
            params["temperature"] = pc.temperature
