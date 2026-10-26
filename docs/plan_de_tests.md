# Plan de tests — pipeline vocal J-ONE

Ce plan couvre la phase 5 du cahier des charges et chaque critère de validation.
Les mesures sont tirées du journal `logs/echanges-AAAAMMJJ.jsonl`, que le pipeline
remplit à chaque énoncé (EF-12). Le rapport se génère avec :

```bash
python tools/rapport_tests.py --out docs/rapport_mesures.md
```

## Avant chaque session

- Batterie chargée, session prévue pour moins de 2 h.
- Robot suspendu sur son cadre et en mode développeur (L2+R2) pour les tests SDK.
- Au moins 2 m entre le robot et les personnes. Arrêt d'urgence (L2+B maintenu plus de 5 s) à portée de l'opérateur.
- `ollama ps` montre `qwen3:4b` chargé, `.env` contient la clé API.

## 1. Tests unitaires (sans robot)

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Ils couvrent le VAD (doublure de Silero), le filtre de transcriptions, le mot d'éveil, le découpage en phrases, le nettoyage pour l'oral, le routeur et sa bascule, l'historique, la conversion audio, le WER et la boucle complète avec doublures.

## 2. Briques isolées sur la Jetson (phase 2)

| Test | Commande | Attendu |
|---|---|---|
| Haut-parleur et LED | `python tools/test_g1_audio.py --say "Bonjour"` | bip, voix audible, 4 couleurs dans l'ordre vert, bleu, blanc, rouge |
| Micro du G1 | même script (fin) | trames reçues, niveau RMS qui monte quand on parle |
| Whisper : WER (ENF-02) | `python tools/enregistrer_corpus.py --phrases tests/corpus_phrases.txt --out corpus/calme --input g1_multicast`, puis `python tools/bench_stt.py --corpus corpus/calme` | WER global ≤ 10 %, latence p90 notée |
| Piper : latence | `python tools/bench_tts.py` | facteur temps réel < 1 |
| LLM et bascule | `python tools/bench_llm.py` puis `--simulate-outage` | premier jeton Claude < 1 s ; avec panne, toutes les réponses viennent de `qwen3-local` |

Enregistrer le corpus avec 3 à 5 voix différentes (hommes, femmes, accents), à environ 1 m du robot, dans une salle calme. Refaire ensuite un corpus `corpus/bruit` avec un fond sonore (ventilation, conversation) pour documenter la dégradation.

## 3. Critères de validation

### C1 — 10 échanges enchaînés sans intervention

1. `python -m jone`
2. Mener une conversation de 10 questions qui se suivent (exemple : présentation, questions de culture générale, « tu te souviens de mon prénom ? » pour vérifier la mémoire EF-05).
3. Ne toucher ni au clavier ni au robot.

Réussi si le rapport indique une « plus longue série d'échanges réussis enchaînés » d'au moins 10.

### C2 — Latence conforme à ENF-01

Sur au moins 30 échanges avec Claude, puis 30 avec `--no-cloud`, lire le tableau « Latence de bout en bout » du rapport : p90 ≤ 2 500 ms avec Claude, ≤ 4 000 ms avec Qwen3.
La mesure va de la fin de parole estimée (détection VAD moins `min_silence_ms`) jusqu'au premier son envoyé au haut-parleur. Pour la valider une fois, filmer une conversation et comparer avec la piste audio de la vidéo.

Leviers si la cible n'est pas atteinte : baisser `vad.min_silence_ms` (500 ms), garder `stt.beam_size: 1`, vérifier que Whisper tourne bien sur GPU (journal au démarrage), réduire `audio_output.lead_seconds`.

### C3 — Fonctionnement Wi-Fi coupé (ENF-03)

1. Lancer `python -m jone` avec le Wi-Fi actif, faire 2 échanges (Claude).
2. Couper le Wi-Fi du robot.
3. Faire 5 échanges : le robot doit répondre à chaque fois. Le premier prend au plus `first_token_timeout_s` de plus, les suivants vont directement au LLM local pendant `retry_cooldown_s`.
4. Rétablir le Wi-Fi : après le délai de 30 s, les réponses reviennent à Claude.

Dans le rapport, les échanges de l'étape 3 doivent avoir `llm = qwen3-local`, `bascule = true` pour le premier.

### C4 — LED conforme à l'état réel

Filmer la tête du robot pendant 5 échanges et vérifier :

| Moment | Couleur attendue |
|---|---|
| en attente de parole | vert |
| après la fin de la phrase de l'utilisateur, avant la réponse | bleu |
| pendant que le robot parle | blanc |
| LLM indisponible (Wi-Fi coupé et Ollama arrêté) | rouge après le message d'excuse |
| retour à l'écoute | vert |

### C5 — Aucune auto-réponse sur 30 minutes

1. Conversation libre de 30 minutes, avec des réponses longues (demander des explications).
2. Le rapport ne doit montrer aucun échange dont la transcription reprend une réponse précédente du robot.

Mécanismes testés : écoute suspendue de la fin de la parole de l'utilisateur jusqu'à `tail_margin_s` après la fin réelle du son (EF-10), tampons micro vidés à la reprise, annulation d'écho matérielle du réseau de micros.

Cette même session sert au test de robustesse ENF-06 : le processus ne doit pas s'arrêter (vérifier `logs/jone.log`, aucune trace `Traceback` non rattrapée).

### C6 — Intelligibilité (ENF-05)

5 testeurs écoutent les phrases de `python tools/bench_tts.py --play` et notent pour chacune « compris / pas compris ». Réussi si au moins 4 testeurs sur 5 jugent la voix correctement compréhensible.

## 4. Fiche de résultats

| Critère | Mesure | Cible | Résultat | Date |
|---|---|---|---|---|
| C1 enchaînement | série max | ≥ 10 | | |
| C2 latence Claude | p90 | ≤ 2 500 ms | | |
| C2 latence Qwen3 | p90 | ≤ 4 000 ms | | |
| C3 hors ligne | réponses Wi-Fi coupé | 5/5 | | |
| C4 LED | états conformes | 5/5 | | |
| C5 auto-réponse | sur 30 min | 0 | | |
| ENF-02 WER salle calme | WER global | ≤ 10 % | | |
| ENF-05 intelligibilité | testeurs | ≥ 4/5 | | |
| ENF-06 robustesse | arrêts sur 30 min | 0 | | |
