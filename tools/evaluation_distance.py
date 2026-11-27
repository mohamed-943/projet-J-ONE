"""Distance d'écoute : WER, FRR et reconnaissance des commandes selon la distance au
robot (docs/protocole_evaluation.md, section 5).

    python tools/evaluation_distance.py --corpus corpus/distance/personne1
                                        [--seuils-vad 0.5 0.4 0.3] [--seuil-wer 0.10] [--seuil-frr 0.10]

Le corpus contient un dossier par distance, nommé d'après elle : 0.5m, 1m, 2m, 3m…
(enregistrés avec tests/protocole_distance.tsv). La distance maximale retenue est la
plus grande pour laquelle cette distance et toutes les plus courtes respectent les
seuils de WER et de FRR. --seuils-vad compare plusieurs réglages du détecteur de
parole sur les mêmes enregistrements, pour voir lequel porte le plus loin.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from evaluation import Frontend, apply_vad_threshold, db, evaluate, pct
from jone.config import load_config, resolve_path
from jone.evaluation import compute_metrics, load_protocol
from jone.vad import SileroVAD, SpeechSegmenter

_DIST = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(cm|m)$", re.IGNORECASE)


def distance_dirs(root: Path) -> list[tuple[float, Path]]:
    found = []
    for d in root.iterdir():
        m = _DIST.match(d.name) if d.is_dir() else None
        if m:
            value = float(m.group(1).replace(",", "."))
            found.append((value / 100 if m.group(2).lower() == "cm" else value, d))
    return sorted(found)


def max_distance(rows: list[dict], max_wer: float, max_frr: float) -> float | None:
    """Plus grande distance telle que toutes les distances jusqu'à elle sont dans les seuils."""
    best = None
    for row in rows:  # triées par distance croissante
        if row["wer"] > max_wer or (row["frr"] or 0) > max_frr:
            break
        best = row["distance_m"]
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="dossier contenant un sous-dossier par distance")
    p.add_argument("--protocole", default="tests/protocole_distance.tsv")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--seuils-vad", type=float, nargs="*", help="seuils de détection de parole à comparer")
    p.add_argument("--seuil-wer", type=float, default=0.10)
    p.add_argument("--seuil-frr", type=float, default=0.10)
    p.add_argument("--out", help="rapport Markdown (défaut : docs/evaluation_distance-<date>.md)")
    args = p.parse_args()

    dirs = distance_dirs(Path(args.corpus))
    if not dirs:
        raise SystemExit(f"Aucun dossier de distance (0.5m, 1m, 2m…) dans {args.corpus}")
    cfg = load_config(args.config)
    items = load_protocol(ROOT / args.protocole)
    fe = Frontend(cfg)
    thresholds = args.seuils_vad or [cfg["vad"]["threshold"]]

    rows, details = [], {}
    for threshold in thresholds:
        apply_vad_threshold(cfg, threshold)
        fe.segmenter = SpeechSegmenter.from_config(SileroVAD(fe.rate), cfg)
        for distance, folder in dirs:
            print(f"Seuil VAD {threshold}, {distance:g} m : {folder}")
            results = evaluate(fe, items, folder, verbose=False)
            m = compute_metrics(results)
            noise = [r for r in results if r.item.type == "bruit"]
            rows.append({
                "seuil_vad": threshold, "distance_m": distance, "n": m["wer"]["n"],
                "wer": m["wer"]["global"], "frr": m["parole"]["FRR"]["taux"],
                "commandes": m["reconnaissance_commandes"]["taux"],
                "far": m["parole"]["FAR"]["taux"] if noise else None,
                "voix_db": m["niveau_voix_db"], "snr_db": m["snr_db"],
            })
            details[(threshold, distance)] = m

    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    csv_path = resolve_path(cfg["logging"]["dir"]) / f"evaluation_distance-{stamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def pc(v):
        return "–" if v is None else f"{100 * v:.0f} %"

    lines = [
        "# Distance d'écoute du robot",
        "",
        f"Date : {datetime.now():%Y-%m-%d %H:%M}  ",
        f"Corpus : {args.corpus} ({len(items)} enregistrements par distance, {args.protocole})  ",
        f"Critères : WER ≤ {100 * args.seuil_wer:.0f} % et FRR ≤ {100 * args.seuil_frr:.0f} %",
        "",
    ]
    for threshold in thresholds:
        sub = [r for r in rows if r["seuil_vad"] == threshold]
        best = max_distance(sub, args.seuil_wer, args.seuil_frr)
        lines += [
            f"## Seuil VAD {threshold}",
            "",
            f"**Distance maximale utilisable : {'aucune' if best is None else f'{best:g} m'}**",
            "",
            "| Distance | WER | FRR parole | Commandes reconnues | Niveau voix | SNR | Silence → réponse |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sub:
            m = details[(threshold, r["distance_m"])]
            lines.append(
                f"| {r['distance_m']:g} m | {pc(r['wer'])} | {pct(m['parole']['FRR'])} | "
                f"{pct(m['reconnaissance_commandes'])} | {db(r['voix_db'])} | {db(r['snr_db'])} | "
                f"{'oui' if r['far'] else 'non' if r['far'] is not None else '–'} |")
        lines.append("")
    lines += [
        "Niveau de la voix et bruit de fond en dBFS (0 = pleine échelle) ; SNR = voix − bruit de fond.  ",
        f"Données pour les graphiques : {csv_path}",
    ]

    out = Path(args.out) if args.out else ROOT / "docs" / f"evaluation_distance-{stamp}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nRapport : {out}\nDonnées : {csv_path}")


if __name__ == "__main__":
    main()
