"""Évaluation scientifique du front-end vocal : WER, reconnaissance des commandes,
latence, FAR / FRR (docs/protocole_evaluation.md).

    python tools/evaluation.py --corpus corpus/eval_local [--protocole tests/protocole_evaluation.tsv]
                               [--logs "logs/echanges-*.jsonl"] [--depuis 2026-11-27T22:15]
                               [--llm local] [--out docs/...]

Chaque enregistrement passe par le VAD, Whisper, le filtre et le mot d'éveil, avec
la configuration du robot (config.yaml + config.local.yaml). La latence de bout en
bout est lue dans le journal des échanges : elle se mesure en conversation réelle.
Résultat : un rapport Markdown et le détail par enregistrement en JSON (logs/).
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from bench_stt import load_wav_16k
from jone.config import load_config, resolve_path
from jone.evaluation import ItemResult, command_recognized, compute_metrics, load_protocol
from jone.metrics import summarize, wer
from jone.stt import create_stt
from jone.text_filter import TranscriptFilter, WakeWordGate
from jone.vad import SileroVAD, SpeechSegmenter

TARGETS_MS = {"claude": 2500, "groq": 2500, "local": 4000}  # ENF-01


def segments(segmenter: SpeechSegmenter, audio: np.ndarray, frame: int, rate: int):
    """Énoncés découpés par le VAD, comme dans le pipeline. Une seconde de silence
    est ajoutée à la fin pour clore une phrase qui finirait avec l'enregistrement."""
    segmenter.reset()
    padded = np.concatenate([audio, np.zeros(rate, dtype=np.int16)])
    for start in range(0, len(padded) - frame + 1, frame):
        utterance = segmenter.process(padded[start : start + frame])
        if utterance is not None:
            yield utterance


def evaluate(cfg: dict, items, corpus: Path) -> list[ItemResult]:
    a = cfg["audio_input"]
    segmenter = SpeechSegmenter.from_config(SileroVAD(a["sample_rate"]), cfg)
    stt = create_stt(cfg)
    stt.warmup()
    text_filter = TranscriptFilter.from_config(cfg)
    wake_cfg = dict(cfg["wake_word"], enabled=True)  # évalué même si désactivé en écoute
    wake = WakeWordGate.from_config({"wake_word": wake_cfg})

    results = []
    for item in items:
        path = corpus / f"{item.id}.wav"
        if not path.exists():
            print(f"  {item.id} : absent, ignoré")
            continue
        r = ItemResult(item)
        texts, accepted = [], []
        for utt in segments(segmenter, load_wav_16k(path), a["frame_samples"], a["sample_rate"]):
            r.segments += 1
            tr = stt.transcribe(utt)
            r.stt_ms.append(tr.elapsed_ms)
            r.language = r.language or tr.language
            texts.append(tr.text)
            verdict = text_filter.check(tr)
            if verdict.accepted:
                accepted.append(verdict.text)
                r.wake = r.wake or wake.admit(verdict.text, 0.0) is not None
            else:
                r.rejections.append(verdict.reason)
        r.transcription = " ".join(t for t in texts if t).strip()
        r.accepted = " ".join(accepted).strip()
        results.append(r)
        print(f"  {item.id} {item.type:10s} {r.segments} énoncé(s) → {r.accepted or '∅'}")
    return results


def end_to_end(log_paths: list[str], backend: str, since: str = "") -> dict:
    """Latence de bout en bout des échanges réussis de `backend`, depuis `since`
    (début d'horodatage ISO, par exemple 2026-11-27T22:15)."""
    records = []
    for path in log_paths:
        with open(path, encoding="utf-8") as f:
            records += [json.loads(l) for l in f if l.strip()]
    vals = [r["latences_ms"]["bout_en_bout"] for r in records
            if r.get("llm") == backend and r.get("statut") == "ok"
            and r.get("horodatage", "") >= since
            and r.get("latences_ms", {}).get("bout_en_bout") is not None]
    return summarize(vals)


def pct(rate: dict) -> str:
    if rate["taux"] is None:
        return "–"
    lo, hi = rate["ic95"]
    return f"{100 * rate['taux']:.1f} % ({rate['k']}/{rate['n']}, IC95 {100 * lo:.0f}–{100 * hi:.0f} %)"


def ms(v) -> str:
    return "–" if v is None else f"{v:.0f} ms"


def report(cfg: dict, metrics: dict, e2e: dict, backend: str, results: list[ItemResult],
           since: str = "") -> str:
    s = cfg["stt"]
    moteur = (f"whisper-server, {Path(s['server']['model']).name}" if s.get("backend") == "whisper-server"
              else f"faster-whisper {s['model']} ({s['device']})")
    w, lat = metrics["wer"], metrics["latence_stt_ms"]
    target = TARGETS_MS.get(backend)
    ok = e2e["p90"] is not None and target and e2e["p90"] <= target
    lines = [
        f"# Évaluation du système vocal — mode {backend}",
        "",
        f"Date : {datetime.now():%Y-%m-%d %H:%M}  ",
        f"Transcription : {moteur} ; LLM : {cfg['llm']['local']['model'] if backend == 'local' else backend}  ",
        f"Enregistrements évalués : {len(results)} (protocole : docs/protocole_evaluation.md)",
        "",
        "## 1. WER — qualité de transcription",
        "",
        "| | WER |",
        "|---|---|",
        f"| Global ({w['n']} phrases) | {100 * w['global']:.1f} % |",
    ]
    lines += [f"| {lang} | {100 * v:.1f} % |" for lang, v in w.items() if lang not in ("global", "n")]
    lines += [
        "",
        f"Identification de la langue : {pct(metrics['identification_langue'])}",
        "",
        "## 2. Recognition Accuracy — reconnaissance des commandes",
        "",
        "Une commande est reconnue si le mot d'éveil est détecté et que ses mots-clés figurent dans la transcription acceptée.",
        "",
        f"- Précision de reconnaissance : {pct(metrics['reconnaissance_commandes'])}",
        f"- Mots-clés présents (sans condition sur le mot d'éveil ni le filtre) : {pct(metrics['mots_cles_seuls'])}",
        "",
        "## 3. Latency — rapidité",
        "",
        "| Mesure | n | p50 | p90 | max |",
        "|---|---|---|---|---|",
        f"| Transcription (par énoncé) | {lat['n']} | {ms(lat['p50'])} | {ms(lat['p90'])} | {ms(lat['max'])} |",
        f"| Bout en bout, fin de parole → premier son ({backend}) | {e2e['n']} | {ms(e2e['p50'])} | "
        f"{ms(e2e['p90'])} | {ms(e2e['max'])} |",
        "",
        f"Bout en bout : échanges du journal{' depuis ' + since if since else ''}. "
        + (f"Cible ENF-01 : p90 ≤ {target} ms → {'conforme' if ok else 'non conforme'}." if target else ""),
        "",
        "## 4. FAR / FRR — fiabilité de la détection vocale",
        "",
        "| Niveau | FAR (fausses acceptations) | FRR (faux rejets) |",
        "|---|---|---|",
        f"| Parole (VAD + filtre) | {pct(metrics['parole']['FAR'])} | {pct(metrics['parole']['FRR'])} |",
        f"| Mot d'éveil « Sarah » | {pct(metrics['mot_eveil']['FAR'])} | {pct(metrics['mot_eveil']['FRR'])} |",
        "",
        "Parole : FAR = bruits qui auraient déclenché une réponse ; FRR = phrases sans aucun énoncé accepté.  ",
        "Mot d'éveil : FAR = phrases hors cible et bruits acceptés ; FRR = commandes où « Sarah » n'est pas reconnu.",
        "",
        "## Détail",
        "",
        "| id | type | langue | transcription acceptée | WER | commande |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        it = r.item
        w_item = f"{100 * wer(it.texte, r.transcription):.0f} %" if it.is_speech else ""
        cmd = ("oui" if command_recognized(r) else "non") if it.type == "commande" else ""
        text = (r.accepted or "∅").replace("|", "/")
        lines.append(f"| {it.id} | {it.type} | {r.language or '–'} | {text} | {w_item} | {cmd} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--protocole", default="tests/protocole_evaluation.tsv")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--logs", default="logs/echanges-*.jsonl")
    p.add_argument("--llm", default="local", help="modèle dont on lit la latence de bout en bout")
    p.add_argument("--depuis", default="", help="ne compter que les échanges à partir de cet horodatage")
    p.add_argument("--out", help="rapport Markdown (défaut : docs/evaluation_<llm>-<date>.md)")
    args = p.parse_args()

    cfg = load_config(args.config)
    items = load_protocol(ROOT / args.protocole)
    results = evaluate(cfg, items, Path(args.corpus))
    metrics = compute_metrics(results)
    e2e = end_to_end(sorted(glob.glob(str(ROOT / args.logs))), args.llm, args.depuis)

    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    out = Path(args.out) if args.out else ROOT / "docs" / f"evaluation_{args.llm}-{stamp}.md"
    out.write_text(report(cfg, metrics, e2e, args.llm, results, args.depuis), encoding="utf-8")
    detail = resolve_path(cfg["logging"]["dir"]) / f"evaluation-{stamp}.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(json.dumps({
        "metriques": metrics, "bout_en_bout_ms": e2e,
        "enregistrements": [{"id": r.item.id, "type": r.item.type, "langue": r.item.langue,
                             "reference": r.item.texte, "transcription": r.transcription,
                             "acceptee": r.accepted, "rejets": r.rejections, "mot_eveil": r.wake,
                             "langue_detectee": r.language, "stt_ms": r.stt_ms} for r in results],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRapport : {out}\nDétail : {detail}")


if __name__ == "__main__":
    main()
