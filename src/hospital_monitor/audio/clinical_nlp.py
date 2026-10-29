"""NLP clínico em português para transcrições de consultas.

* Extração de termos clínicos por dicionário (sintomas, sinais de alarme,
  medicamentos), com normalização de acentos e variações.
* Detecção de negação e incerteza no estilo NegEx (Chapman et al., 2001),
  adaptada ao pt-BR: "não tenho dor no peito" não gera alerta.
* Sentimento: ``pysentimiento`` (BERTweet-pt) quando instalado; senão léxico
  próprio com negação e intensificadores.

O resultado vira um evento ``TEXT`` cujo escore cresce com a gravidade dos
termos críticos afirmados e com sentimento negativo intenso (medo, angústia).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

# termo canônico → (categoria, gravidade 0–1, variantes)
CLINICAL_LEXICON: dict[str, tuple[str, float, list[str]]] = {
    "dor torácica": ("sinal_alarme", 0.95, ["dor no peito", "pressao no peito", "aperto no peito", "dor toracica"]),
    "dispneia": ("sinal_alarme", 0.9, ["falta de ar", "nao consigo respirar", "dificuldade para respirar",
                                       "dispneia", "sem folego", "cansaco para respirar"]),
    "síncope": ("sinal_alarme", 0.9, ["desmaio", "desmaiei", "sincope", "perdi a consciencia", "apaguei"]),
    "disartria": ("sinal_alarme", 0.95, ["fala enrolada", "fala esta enrolada", "falando enrolado", "fala arrastada", "dificuldade para falar",
                                         "disartria", "enrolando a lingua"]),
    "déficit motor": ("sinal_alarme", 0.9, ["fraqueza no braco", "fraqueza na perna", "nao sinto o braco",
                                            "lado do corpo fraco"]),
    "parestesia facial": ("sinal_alarme", 0.85, ["rosto dormente", "rosto meio dormente", "rosto esta meio dormente", "rosto esta dormente", "boca torta",
                                                 "formigamento no rosto"]),
    "confusão mental": ("sinal_alarme", 0.85, ["confuso", "confusao", "desorientado", "nao sei onde estou"]),
    "hemorragia": ("sinal_alarme", 0.9, ["sangramento", "sangrando", "vomitei sangue", "sangue nas fezes"]),
    "fadiga": ("sintoma", 0.5, ["cansaco", "cansado", "cansada", "fadiga", "exausto", "exausta", "fraqueza"]),
    "tontura": ("sintoma", 0.55, ["tontura", "tonto", "tonta", "vertigem"]),
    "palpitação": ("sintoma", 0.6, ["palpitacao", "coracao acelerado", "coracao disparado"]),
    "febre": ("sintoma", 0.5, ["febre", "febril", "calafrio"]),
    "tosse": ("sintoma", 0.35, ["tosse", "tossindo"]),
    "cefaleia": ("sintoma", 0.45, ["dor de cabeca", "cefaleia", "enxaqueca"]),
    "náusea": ("sintoma", 0.35, ["nausea", "enjoo", "enjoado", "vomito", "vomitando"]),
    "dor": ("sintoma", 0.4, ["dor forte", "muita dor", "dor intensa"]),
    "insônia": ("sintoma", 0.25, ["dormi mal", "nao consigo dormir", "insonia"]),
    "ansiedade": ("sintoma", 0.3, ["medo", "ansioso", "ansiosa", "angustia", "nervoso"]),
}

MEDICATIONS = [
    "dipirona", "paracetamol", "ibuprofeno", "morfina", "tramadol", "enoxaparina", "heparina", "varfarina",
    "losartana", "captopril", "enalapril", "metformina", "insulina", "omeprazol", "furosemida",
    "ceftriaxona", "amoxicilina", "azitromicina", "salbutamol", "prednisona", "dexametasona",
]

NEG_PRE = ["nao", "nega", "negou", "sem", "nenhum", "nenhuma", "nunca", "ausencia de", "nem", "jamais", "parou de"]
UNCERTAIN_PRE = ["talvez", "acho que", "parece", "possivel", "nao sei se", "as vezes"]
TERMINATORS = ["mas", "porem", "entretanto", "contudo", "exceto", "e sim", "so que", ".", ";", "!", "?"]
NEG_WINDOW = 6  # tokens

POS_WORDS = {"bem", "melhor", "otimo", "otima", "animado", "animada", "feliz", "tranquilo", "tranquila",
             "aliviado", "aliviada", "confiante", "consegui", "bom", "boa"}
NEG_WORDS = {"medo", "pior", "piorou", "dor", "angustia", "triste", "preocupado", "preocupada", "mal",
             "ruim", "horrivel", "insuportavel", "nao consigo", "desesperado", "cansado", "cansada", "fraqueza"}
INTENSIFIERS = {"muito", "muita", "bastante", "demais", "extremamente", "forte"}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[.;!?]", text)


@dataclass
class ClinicalFinding:
    term: str
    category: str
    severity: float
    matched_text: str
    negated: bool = False
    uncertain: bool = False
    start_token: int = 0

    @property
    def effective_severity(self) -> float:
        if self.negated:
            return 0.0
        return self.severity * (0.5 if self.uncertain else 1.0)


@dataclass
class TextAnalysis:
    findings: list[ClinicalFinding]
    medications: list[str]
    sentiment: dict
    critical_terms: list[str] = field(default_factory=list)
    anomaly_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "findings": [asdict(f) | {"effective_severity": round(f.effective_severity, 3)} for f in self.findings],
            "medications": self.medications,
            "sentiment": self.sentiment,
            "critical_terms": self.critical_terms,
            "anomaly_score": round(self.anomaly_score, 3),
        }


def _context(tokens: list[str], start: int, triggers: list[str]) -> bool:
    """True se um gatilho aparece até NEG_WINDOW tokens antes, sem terminador no meio."""
    lo = max(0, start - NEG_WINDOW)
    window = tokens[lo:start]
    # corta no último terminador
    for i in range(len(window) - 1, -1, -1):
        if window[i] in TERMINATORS or (i + 1 < len(window) and f"{window[i]} {window[i+1]}" in TERMINATORS):
            window = window[i + 1 :]
            break
    joined = " " + " ".join(window) + " "
    return any(f" {t} " in joined for t in triggers)


def extract_findings(text: str) -> list[ClinicalFinding]:
    norm = normalize(text)
    tokens = _tokens(norm)
    joined = " ".join(tokens)
    # mapa posição de caractere → índice de token
    starts, pos = [], 0
    for tk in tokens:
        starts.append(pos)
        pos += len(tk) + 1
    found: list[ClinicalFinding] = []
    taken: set[int] = set()
    variants = sorted(
        ((v, term, cat, sev) for term, (cat, sev, vs) in CLINICAL_LEXICON.items() for v in vs),
        key=lambda x: -len(x[0]),  # mais longo primeiro: "fraqueza no braco" antes de "fraqueza"
    )
    for v, term, cat, sev in variants:
        for m in re.finditer(rf"(?<!\w){re.escape(v)}(?!\w)", joined):
            ti = max(i for i, s in enumerate(starts) if s <= m.start())
            span = set(range(ti, ti + len(v.split())))
            if span & taken:
                continue
            taken |= span
            found.append(ClinicalFinding(
                term=term, category=cat, severity=sev, matched_text=v,
                negated=_context(tokens, ti, NEG_PRE), uncertain=_context(tokens, ti, UNCERTAIN_PRE), start_token=ti,
            ))
    return sorted(found, key=lambda f: f.start_token)


def lexicon_sentiment(text: str) -> dict:
    tokens = _tokens(normalize(text))
    score, hits = 0.0, 0
    for i, tk in enumerate(tokens):
        bigram = f"{tk} {tokens[i+1]}" if i + 1 < len(tokens) else ""
        pol = -1 if (tk in NEG_WORDS or bigram in NEG_WORDS) else (1 if tk in POS_WORDS else 0)
        if not pol:
            continue
        w = 1.5 if i > 0 and tokens[i - 1] in INTENSIFIERS else 1.0
        if _context(tokens, i, NEG_PRE) and bigram != "nao consigo":
            pol = -pol * 0.5
        score += pol * w
        hits += 1
    compound = max(-1.0, min(1.0, score / max(3.0, hits)))
    label = "NEG" if compound < -0.15 else ("POS" if compound > 0.15 else "NEU")
    return {"label": label, "compound": round(compound, 3), "backend": "lexicon"}


_ANALYZER = None


def sentiment(text: str, backend: str = "auto") -> dict:
    global _ANALYZER
    if backend in ("auto", "pysentimiento"):
        try:
            if _ANALYZER is None:
                from pysentimiento import create_analyzer

                _ANALYZER = create_analyzer(task="sentiment", lang="pt")
            out = _ANALYZER.predict(text)
            p = out.probas
            return {"label": out.output, "compound": round(p.get("POS", 0) - p.get("NEG", 0), 3),
                    "probas": {k: round(v, 3) for k, v in p.items()}, "backend": "pysentimiento"}
        except Exception:
            if backend == "pysentimiento":
                raise
    return lexicon_sentiment(text)


def analyze_text(text: str, sentiment_backend: str = "auto") -> TextAnalysis:
    findings = extract_findings(text)
    norm = normalize(text)
    meds = [m for m in MEDICATIONS if re.search(rf"\b{m}\b", norm)]
    sent = sentiment(text, sentiment_backend)
    affirmed = [f for f in findings if f.effective_severity > 0]
    critical = sorted({f.term for f in affirmed if f.category == "sinal_alarme"})
    # escore: termo mais grave + reforço por termos adicionais + sentimento muito negativo
    sev = sorted((f.effective_severity for f in affirmed), reverse=True)
    score = sev[0] if sev else 0.0
    score += 0.05 * max(0, len(sev) - 1)
    if sent["compound"] < -0.4 and sev:
        score += 0.05
    return TextAnalysis(findings, meds, sent, critical, min(1.0, score))
