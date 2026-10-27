"""Phase 5 — rapport de tests à partir du journal des échanges (EF-12).

    python tools/rapport_tests.py [logs/echanges-*.jsonl ...] [--out docs/rapport_mesures.md]

Calcule les latences par modèle et leur conformité à ENF-01, le taux de bascule,
la répartition des statuts et la plus longue série d'échanges réussis enchaînés
(critère « 10 échanges sans intervention »).
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from jone.metrics import summarize

TARGETS_MS = {"claude": 2500, "groq": 2500, "qwen3-local": 4000}  # ENF-01


def load(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    records.sort(key=lambda r: r.get("horodatage", ""))
    return records


def fmt(v, unit=" ms"):
    return "–" if v is None else f"{v:.0f}{unit}"


def longest_ok_streak(records: list[dict], max_gap_s: float = 120) -> int:
    """Échanges « ok » consécutifs, sans pause de plus de `max_gap_s` entre deux."""
    best = cur = 0
    last = None
    for r in records:
        if r.get("statut") in ("rejete", "ignore"):
            continue  # bruit filtré : n'interrompt pas la conversation
        t = datetime.fromisoformat(r["horodatage"])
        if r.get("statut") == "ok" and (last is None or (t - last).total_seconds() <= max_gap_s):
            cur += 1
        elif r.get("statut") == "ok":
            cur = 1
        else:
            cur = 0
        last = t
        best = max(best, cur)
    return best


def report(records: list[dict]) -> str:
    lines = ["# Rapport de mesures — pipeline vocal J-ONE", ""]
    if not records:
        return "\n".join(lines + ["Aucun échange journalisé."])

    first, last = records[0]["horodatage"], records[-1]["horodatage"]
    statuses = Counter(r.get("statut") for r in records)
    answered = [r for r in records if r.get("statut") in ("ok", "erreur_llm", "reponse_vide")]
    lines += [
        f"Période : {first} → {last}  ",
        f"Énoncés détectés : {len(records)}  ",
        "Statuts : " + ", ".join(f"{k} = {v}" for k, v in statuses.most_common()),
        "",
        "## Latence de bout en bout (ENF-01)",
        "",
        "Mesurée de la fin de parole estimée (détection VAD moins le silence de fin) au premier son émis.",
        "",
        "| Modèle | n | moyenne | p50 | p90 | max | cible | conformité (p90) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for backend, target in TARGETS_MS.items():
        vals = [r["latences_ms"]["bout_en_bout"] for r in answered
                if r.get("llm") == backend and r["latences_ms"].get("bout_en_bout") is not None]
        s = summarize(vals)
        ok = "–" if s["p90"] is None else ("oui" if s["p90"] <= target else "NON")
        lines.append(f"| {backend} | {s['n']} | {fmt(s['moyenne'])} | {fmt(s['p50'])} | "
                     f"{fmt(s['p90'])} | {fmt(s['max'])} | ≤ {target} ms | {ok} |")

    lines += ["", "## Détail par étape", "", "| Étape | n | moyenne | p50 | p90 |", "|---|---|---|---|---|"]
    for key, label in (("stt", "Transcription Whisper"), ("llm_premier_jeton", "LLM, premier jeton"),
                       ("tts_premiere_phrase", "Piper, première phrase")):
        vals = [r["latences_ms"][key] for r in answered if r["latences_ms"].get(key) is not None]
        s = summarize(vals)
        lines.append(f"| {label} | {s['n']} | {fmt(s['moyenne'])} | {fmt(s['p50'])} | {fmt(s['p90'])} |")

    by_llm = Counter(r.get("llm") for r in answered)
    fallbacks = sum(1 for r in answered if r.get("bascule"))
    lines += [
        "",
        "## Bascule cloud → local (EF-07)",
        "",
        "Réponses par modèle : " + ", ".join(f"{k or 'aucun'} = {v}" for k, v in by_llm.items()) + "  ",
        f"Taux de bascule : {fallbacks}/{len(answered)} "
        f"({fallbacks / len(answered):.0%})" if answered else "Taux de bascule : –",
        "",
        "## Robustesse et enchaînement",
        "",
        f"Plus longue série d'échanges réussis enchaînés : {longest_ok_streak(records)} (critère : 10)  ",
        f"Transcriptions rejetées par le filtre (EF-03) : {statuses.get('rejete', 0)}",
    ]
    reasons = Counter(r.get("motif") for r in records if r.get("statut") == "rejete")
    if reasons:
        lines.append("Motifs : " + ", ".join(f"{k} = {v}" for k, v in reasons.most_common()))
    errors = Counter(e for r in answered for e in r.get("erreurs_llm", []))
    if errors:
        lines += ["", "Erreurs LLM rencontrées :", ""] + [f"- {e} ({n}×)" for e, n in errors.most_common(10)]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*")
    p.add_argument("--out")
    args = p.parse_args()
    paths = args.files or sorted(glob.glob(str(ROOT / "logs" / "echanges-*.jsonl")))
    text = report(load(paths))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Rapport écrit dans {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
