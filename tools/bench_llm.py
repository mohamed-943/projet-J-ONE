"""Phase 3 — latence du premier jeton de chaque LLM, et test de bascule.

    python tools/bench_llm.py [--n 5] [--simulate-outage]

--simulate-outage remplace l'URL de l'API par une adresse injoignable pour
vérifier que le routeur bascule bien sur Qwen3 local (EF-07, ENF-04).
"""

from __future__ import annotations

import argparse
import os
import time

import _bootstrap  # noqa: F401
from jone.config import load_config, resolve_path
from jone.conversation import load_system_prompt
from jone.llm.router import RouteInfo, create_router
from jone.metrics import summarize

QUESTIONS = [
    "Bonjour, qui es-tu ?",
    "Qu'est-ce que tu sais faire ?",
    "Peux-tu m'expliquer ce qu'est l'intelligence artificielle ?",
    "Raconte-moi une blague sur les robots.",
    "Tu peux marcher jusqu'à la porte ?",
]


def run(router, system, n):
    firsts, totals, backends = [], [], []
    for q in QUESTIONS[:n]:
        info = RouteInfo()
        t0 = time.perf_counter()
        text = "".join(router.stream(system, [{"role": "user", "content": q}], info))
        totals.append((time.perf_counter() - t0) * 1000)
        first_ms = info.first_token_s * 1000 if info.first_token_s is not None else float("nan")
        if info.first_token_s is not None:
            firsts.append(first_ms)
        backends.append(info.backend)
        extra = f" [bascule, erreurs : {info.errors}]" if info.fallback else ""
        print(f"[{info.backend}] {first_ms:.0f} ms : {q}\n    -> {text.strip()}{extra}")
    print("\nPremier jeton (ms) :", summarize(firsts))
    print("Réponse complète (ms) :", summarize(totals))
    print("Modèles utilisés :", {b: backends.count(b) for b in set(backends)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--n", type=int, default=len(QUESTIONS))
    p.add_argument("--no-cloud", action="store_true")
    p.add_argument("--simulate-outage", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.no_cloud:
        cfg["llm"]["cloud"]["enabled"] = False
    if args.simulate_outage:
        os.environ["ANTHROPIC_BASE_URL"] = "http://10.255.255.1:9"  # adresse non routable
        cfg["llm"]["cloud"]["retry_cooldown_s"] = 0  # retenter à chaque question
    system = load_system_prompt(resolve_path(cfg["llm"]["system_prompt_file"]))
    router = create_router(cfg)
    router.warmup()
    run(router, system, args.n)


if __name__ == "__main__":
    main()
